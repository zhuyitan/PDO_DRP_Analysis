"""
uno_finetune_improve.py
-----------------------
Transfer learning for UNO: load a model.h5 pretrained on cancer cell line
(CCL) data, freeze a chosen set of layers, and fine-tune the rest on PDO
data with mean squared error loss.

Freeze groups (--freeze_group):
    none       all layers trainable (full fine-tune)
    canc       freeze the cancer encoder (dense, dense_1, dense_2)
    drug       freeze the drug encoder (dense_3, dense_4, dense_5)
    canc_drug  freeze both encoders
    last_3     freeze everything except the last 3 interaction layers
               (dense_8, dense_9, dense_10) and the output layer
    last_2     freeze everything except the last 2 interaction layers
               (dense_9, dense_10) and the output layer
    last_1     freeze everything except the last interaction layer
               (dense_10) and the output layer
    all        freeze everything except the output layer
Each group also freezes the dropout layers paired with its dense layers.

--freeze_post_concat_n N additionally freezes the first N interaction
layers after the concatenation (dense_6 first), on top of --freeze_group.
It only adds freezing. Valid values are 0 to 5.

--dropout overrides the rate of every Dropout layer in the loaded model.
If omitted, the pretrained model's own rates are kept.

Stopping modes:
    default        epoch-based training with val_loss early stopping
                   (--early_stopping_patience)
    --max_updates  train for exactly this many parameter updates (batches)
                   with no early stopping, e.g. to retrain with an update
                   count chosen during hyperparameter optimization
In both modes, the best update count is saved to val_scores.json as
"best_update_count".

--ccl_holdout_dir (optional) scores the fine-tuned model on a preprocessed
CCL holdout set after training and adds val_mse_ccl, val_r2_ccl and
val_pcc_ccl to val_scores.json. It is a read-only check and never affects
training.

Outputs written to --output_dir:
    model.h5                  fine-tuned model
    val_scores.json           settings and validation metrics
    val_y_data_predicted.csv  validation true and predicted values

Layer index reference:
  [0]  all_input
  [1]  lambda        (cancer slice)
  [2]  lambda_1      (drug slice)
  [3]  dense         cancer layer 1
  [4]  dropout
  [5]  dense_1       cancer layer 2
  [6]  dropout_1
  [7]  dense_2       cancer layer 3
  [8]  dropout_2
  [9]  dense_3       drug layer 1
  [10] dropout_3
  [11] dense_4       drug layer 2
  [12] dropout_4
  [13] dense_5       drug layer 3
  [14] dropout_5
  [15] concatenate
  [16] dense_6       interaction layer 1
  [17] dropout_6
  [18] dense_7       interaction layer 2
  [19] dropout_7
  [20] dense_8       interaction layer 3
  [21] dropout_8
  [22] dense_9       interaction layer 4
  [23] dropout_9
  [24] dense_10      interaction layer 5
  [25] dropout_10
  [26] dense_11      output
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Required to load model.h5 files containing Lambda layers
keras.config.enable_unsafe_deserialization()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uno_utils_improve import (
    data_merge_generator,
    batch_predict,
    calculate_sstot,
    R2Callback_efficient,
)

# ── Layer name groups ──────────────────────────────────────────────────────────
CANC_LAYERS = {"dense", "dropout", "dense_1", "dropout_1", "dense_2", "dropout_2"}
DRUG_LAYERS = {"dense_3", "dropout_3", "dense_4", "dropout_4", "dense_5", "dropout_5"}

FREEZE_GROUPS = {
    "none": set(),
    "all": CANC_LAYERS | DRUG_LAYERS | {
        "dense_6", "dropout_6",
        "dense_7", "dropout_7",
        "dense_8", "dropout_8",
        "dense_9", "dropout_9",
        "dense_10", "dropout_10",
    },
    "canc": CANC_LAYERS,
    "drug": DRUG_LAYERS,
    "canc_drug": CANC_LAYERS | DRUG_LAYERS,
    "last_1": CANC_LAYERS | DRUG_LAYERS | {
        "dense_6", "dropout_6",
        "dense_7", "dropout_7",
        "dense_8", "dropout_8",
        "dense_9", "dropout_9",
    },
    "last_2": CANC_LAYERS | DRUG_LAYERS | {
        "dense_6", "dropout_6",
        "dense_7", "dropout_7",
        "dense_8", "dropout_8",
    },
    "last_3": CANC_LAYERS | DRUG_LAYERS | {
        "dense_6", "dropout_6",
        "dense_7", "dropout_7",
    },
}

# Interaction layers after the concatenation, in order from the concat side.
POST_CONCAT_LAYER_PAIRS = [
    ("dense_6",  "dropout_6"),
    ("dense_7",  "dropout_7"),
    ("dense_8",  "dropout_8"),
    ("dense_9",  "dropout_9"),
    ("dense_10", "dropout_10"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",               required=True,
                   help="Preprocessed PDO data directory (ge/md/rsp "
                        "train and val parquet files).")
    p.add_argument("--input_model_dir",         required=True,
                   help="Directory containing the pretrained model.h5.")
    p.add_argument("--output_dir",              required=True)
    p.add_argument("--freeze_group",            default="canc",
                   choices=list(FREEZE_GROUPS.keys()))
    p.add_argument("--freeze_post_concat_n",    type=int, default=0,
                   help="Freeze the first N post-concat interaction layers "
                        "(0-5), on top of --freeze_group.")
    p.add_argument("--dropout",                 type=float, default=None,
                   help="Override the rate of every Dropout layer. If "
                        "omitted, keeps the pretrained model's own rates.")
    p.add_argument("--ccl_holdout_dir",         default=None,
                   help="Optional preprocessed CCL holdout set (parquet "
                        "files; the 'test' partition holds the CCL ids to "
                        "score). Read-only check after training.")
    p.add_argument("--finetune_lr",             type=float, default=1e-3)
    p.add_argument("--finetune_epochs",         type=int,   default=50)
    p.add_argument("--batch_size",              type=int,   default=32)
    p.add_argument("--generator_batch_size",    type=int,   default=32)
    p.add_argument("--early_stopping_patience", type=int,   default=10)
    p.add_argument("--max_updates",             type=int,   default=None,
                   help="If given, train for exactly this many parameter "
                        "updates (batches) instead of using early stopping.")
    p.add_argument("--reduce_lr_patience",      type=int,   default=5)
    p.add_argument("--reduce_lr_factor",        type=float, default=0.5)
    p.add_argument("--canc_col_name",  default="improve_sample_id")
    p.add_argument("--drug_col_name",  default="improve_chem_id")
    p.add_argument("--y_col_name",     default="auc")
    p.add_argument("--data_format",    default=".parquet")
    p.add_argument("--metric_type",    default="regression")
    return p.parse_args()


def load_data(input_dir, stage_names=("train", "val")):
    """Load preprocessed parquet files for the given stages."""
    def read(prefix, stage):
        return pd.read_parquet(Path(input_dir) / f"{prefix}_{stage}_data.parquet")

    out = {}
    for stage in stage_names:
        out[f"{stage}_ge"]  = read("ge",  stage)
        out[f"{stage}_md"]  = read("md",  stage)
        out[f"{stage}_rsp"] = read("rsp", stage)
    return out


def apply_freeze(model, freeze_group_name):
    """Set trainable=False for all layers in the chosen freeze group."""
    to_freeze = FREEZE_GROUPS[freeze_group_name]
    frozen, trainable = [], []
    for layer in model.layers:
        if layer.name in to_freeze:
            layer.trainable = False
            frozen.append(layer.name)
        else:
            layer.trainable = True
            trainable.append(layer.name)
    print(f"\n[freeze] group='{freeze_group_name}'")
    print(f"  Frozen    ({len(frozen)}): {frozen}")
    print(f"  Trainable ({len(trainable)}): {trainable}\n")
    return model


def apply_post_concat_freeze(model, n):
    """Freeze the first n post-concat interaction layer pairs, in addition
    to the freeze group. Never unfreezes anything."""
    if n <= 0:
        return model

    n = min(n, len(POST_CONCAT_LAYER_PAIRS))
    names_to_freeze = set()
    for dense_name, dropout_name in POST_CONCAT_LAYER_PAIRS[:n]:
        names_to_freeze.add(dense_name)
        names_to_freeze.add(dropout_name)

    newly_frozen = []
    for layer in model.layers:
        if layer.name in names_to_freeze and layer.trainable:
            layer.trainable = False
            newly_frozen.append(layer.name)

    print(f"[freeze_post_concat_n={n}] additionally frozen: {newly_frozen}\n")
    return model


def apply_dropout_override(model, dropout_rate):
    """Set the rate on every Dropout layer. Safe after loading, since the
    rate is not part of the saved weights."""
    if dropout_rate is None:
        return model

    changed = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dropout):
            layer.rate = dropout_rate
            changed.append(layer.name)

    print(f"[dropout override={dropout_rate}] applied to: {changed}\n")
    return model


class StopAtUpdateCount(keras.callbacks.Callback):
    """Stop training after exactly max_updates parameter updates (batches)."""
    def __init__(self, max_updates):
        super().__init__()
        self.max_updates = max_updates
        self.updates_so_far = 0

    def on_train_batch_end(self, batch, logs=None):
        self.updates_so_far += 1
        if self.updates_so_far >= self.max_updates:
            self.model.stop_training = True


def load_ccl_holdout_test(ccl_holdout_dir):
    """Load the 'test' partition of a preprocessed CCL holdout directory."""
    def read(prefix):
        return pd.read_parquet(Path(ccl_holdout_dir) / f"{prefix}_test_data.parquet")
    return read("ge"), read("md"), read("rsp")


def evaluate_ccl_holdout(model, ccl_holdout_dir, args, gen_params):
    """Score the fine-tuned model on the CCL holdout set (read-only).
    Returns None values if no holdout directory was given."""
    if ccl_holdout_dir is None:
        return {"val_mse_ccl": None, "val_r2_ccl": None, "val_pcc_ccl": None}

    ccl_ge, ccl_md, ccl_rsp = load_ccl_holdout_test(ccl_holdout_dir)

    ccl_steps = int(np.ceil(len(ccl_rsp) / args.generator_batch_size))
    ccl_gen_ordered = data_merge_generator(
        ccl_rsp, ccl_ge, ccl_md, args.generator_batch_size,
        gen_params, merge_preserve_order=True, verbose=False
    )
    ccl_pred, ccl_true = batch_predict(model, ccl_gen_ordered, ccl_steps)

    ccl_mse = float(np.mean((ccl_true - ccl_pred) ** 2))
    ccl_r2  = r2_score(ccl_true, ccl_pred)
    ccl_pcc = pearsonr(ccl_true, ccl_pred)[0]

    return {
        "val_mse_ccl": round(ccl_mse, 6),
        "val_r2_ccl":  round(float(ccl_r2), 4),
        "val_pcc_ccl": round(float(ccl_pcc), 4),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and prepare the pretrained model ─────────────────────────────────
    model_path = Path(args.input_model_dir) / "model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.h5 found at {model_path}")

    print(f"Loading pretrained model from: {model_path}")
    model = keras.models.load_model(str(model_path), compile=False, safe_mode=False)

    model = apply_freeze(model, args.freeze_group)
    model = apply_post_concat_freeze(model, args.freeze_post_concat_n)
    model = apply_dropout_override(model, args.dropout)

    optimizer = keras.optimizers.Adam(learning_rate=args.finetune_lr)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    model.summary()

    # ── Load PDO data ─────────────────────────────────────────────────────────
    print(f"\nLoading PDO data from: {args.input_dir}")
    try:
        data = load_data(args.input_dir, stage_names=("train", "val"))
    except Exception as e:
        raise RuntimeError(
            f"Data loading failed: {e}\n"
            f"Expected parquet files in {args.input_dir}"
        )
    tr_ge, tr_md, tr_rsp = data["train_ge"], data["train_md"], data["train_rsp"]
    vl_ge, vl_md, vl_rsp = data["val_ge"],   data["val_md"],   data["val_rsp"]

    print(f"  PDO train samples: {len(tr_rsp)}")
    print(f"  PDO val   samples: {len(vl_rsp)}")

    gen_params = {
        "canc_col_name": args.canc_col_name,
        "drug_col_name": args.drug_col_name,
        "y_col_name":    args.y_col_name,
        "train_debug":   False,
    }

    train_gen = data_merge_generator(
        tr_rsp, tr_ge, tr_md, args.batch_size, gen_params, shuffle=True, peek=True
    )
    val_gen = data_merge_generator(
        vl_rsp, vl_ge, vl_md, args.generator_batch_size, gen_params, shuffle=False, peek=True
    )

    steps_per_epoch  = int(np.ceil(len(tr_rsp) / args.batch_size))
    validation_steps = int(np.ceil(len(vl_rsp) / args.generator_batch_size))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=args.reduce_lr_factor,
        patience=args.reduce_lr_patience,
        min_lr=args.finetune_lr / 1000, verbose=1,
    )

    stop_at_update_count = None
    if args.max_updates is not None:
        # Epoch count is only a safety cap; StopAtUpdateCount ends training
        # at the exact update.
        epochs = int(np.ceil(args.max_updates / steps_per_epoch)) + 1
        stop_at_update_count = StopAtUpdateCount(args.max_updates)
        stopping_callbacks = [stop_at_update_count]
        print(f"[stopping] training for exactly {args.max_updates} updates "
              f"(safety cap: {epochs} epochs); early stopping disabled.\n")
    else:
        epochs = args.finetune_epochs
        stopping_callbacks = [EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience,
            mode="min", verbose=1, restore_best_weights=True,
        )]

    train_ss_tot = calculate_sstot(tr_rsp[args.y_col_name])
    val_ss_tot   = calculate_sstot(vl_rsp[args.y_col_name])
    r2_callback  = R2Callback_efficient(train_ss_tot=train_ss_tot, val_ss_tot=val_ss_tot)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    start = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[r2_callback, reduce_lr] + stopping_callbacks,
    )
    elapsed = time.time() - start
    print(f"\nFine-tuning complete in {elapsed:.1f}s  ({len(history.history['loss'])} epochs)")

    if stop_at_update_count is not None:
        best_update_count = min(args.max_updates, stop_at_update_count.updates_so_far)
    else:
        val_loss_history = history.history.get("val_loss", [])
        if val_loss_history:
            best_epoch_idx = int(np.argmin(val_loss_history))
            best_update_count = (best_epoch_idx + 1) * steps_per_epoch
        else:
            best_update_count = None

    ft_model_path = output_dir / "model.h5"
    model.save(str(ft_model_path))
    print(f"Fine-tuned model saved to: {ft_model_path}")

    # ── Validation scores ─────────────────────────────────────────────────────
    val_gen_ordered = data_merge_generator(
        vl_rsp, vl_ge, vl_md, args.generator_batch_size,
        gen_params, merge_preserve_order=True, verbose=False
    )
    val_pred, val_true = batch_predict(model, val_gen_ordered, validation_steps)

    r2   = r2_score(val_true, val_pred)
    pcc  = pearsonr(val_true, val_pred)[0]
    scc  = spearmanr(val_true, val_pred)[0]
    mse  = float(np.mean((val_true - val_pred) ** 2))
    rmse = float(np.sqrt(mse))

    val_scores = {
        "freeze_group":         args.freeze_group,
        "freeze_post_concat_n": args.freeze_post_concat_n,
        "dropout":              args.dropout,
        "finetune_lr":          args.finetune_lr,
        "batch_size":           args.batch_size,
        "finetune_epochs":      len(history.history["loss"]),
        "steps_per_epoch":      steps_per_epoch,
        "best_update_count":    best_update_count,
        "max_updates":          args.max_updates,
        "val_r2":               round(r2,   4),
        "val_pcc":              round(pcc,  4),
        "val_scc":              round(scc,  4),
        "val_mse":              round(mse,  6),
        "val_rmse":             round(rmse, 6),
    }

    print("\nEvaluating CCL holdout set...")
    val_scores.update(evaluate_ccl_holdout(model, args.ccl_holdout_dir, args, gen_params))

    print(f"\nValidation scores: {val_scores}")

    with open(output_dir / "val_scores.json", "w") as f:
        json.dump(val_scores, f, indent=2)
    print(f"Scores saved to: {output_dir / 'val_scores.json'}")

    pd.DataFrame({"y_true": val_true, "y_pred": val_pred}).to_csv(
        output_dir / "val_y_data_predicted.csv", index=False
    )


if __name__ == "__main__":
    main()