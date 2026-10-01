##############################################################################
# transfer_learning: UNO fine-tuning with hyperparameter optimization.
#
# Conventional transfer learning: fine-tune the cell_line_only_training
# pretrained UNO model on each study's PDO train split, pick
# hyperparameters by val_mse on the study's val split with differential
# evolution, then evaluate on the study test split.
#
# For each (split_type, study, CCL split i, PDO split j):
#   pretrained model: cell_line_only_training UNO model trained on ccl_split_{i}
#   train:            {study}_split_{j}_train_ids.txt
#   val:              {study}_split_{j}_val_ids.txt
#   test:             {study}_split_{j}_test_ids.txt
#   mixed test:       {study}_ccl_ids.txt   (skipped for Pilot_1)
#
# Cancer-blind PDO split counts are detected per study; random uses 10.
#
# Requirements:
#   - cell_line_only_training UNO results (run_experiments/cell_line_only_training/run_uno.py)
#     for every split type, including both preprocessing scalers.
#   - models/UNO/uno_finetune_improve.py with a "canc_drug" entry
#     (CANC_LAYERS | DRUG_LAYERS) in FREEZE_GROUPS.
#   - scipy
#
# Usage (from the repository root):
#   python run_experiments/transfer_learning/run_uno.py
#
# Runs every split type in SPLIT_TYPES in order. Safe to stop and rerun:
# jobs that are already complete are skipped.
##############################################################################

import os
import sys
import json
import math
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Manager

from scipy.optimize import differential_evolution

############################################################
# CONFIG
############################################################

PILOT_MODE     = False
PILOT_CCL_FOLD = 0
PILOT_PDO_FOLD = 0

# Run in this order automatically, in a single process.
SPLIT_TYPES = ["cancer-blind", "random"]

STUDIES = [
    "IMPROVE dataset 1",
    "Pilot_1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
]

# Studies that do NOT have a {study}_ccl_ids.txt file.
# The mixed (study CCL) inference is skipped for these.
SEPERATE_STUDIES = {
    "Pilot_1",
}

# All paths are relative to the repository root and converted to
# absolute paths so they stay valid wherever a command runs from.
RAW_DATA_DIR = os.path.abspath("pdo_drp_data")
SPLITS_ROOT  = os.path.abspath("pdo_drp_data/splits")

BASE_OUTPUT_ROOT = os.path.abspath("transfer_learning_results_uno")

# Pretrained cell_line_only_training UNO model for CCL split i. By default this is the
# output of run_experiments/cell_line_only_training/run_uno.py.
PRETRAIN_DIR_TEMPLATE = os.path.abspath(
    "cell_line_only_results_uno/{split_type}/split_{i}/processed_ccl"
)
PRETRAIN_N_SPLITS = 10
PRETRAIN_SCALER_NAMES = [
    "x_data_gene_expression_scaler.gz",
    "x_data_mordred_scaler.gz",
]

UNO_DIR = os.path.abspath("models/UNO")

TRAIN_SCRIPT      = os.path.join(UNO_DIR, "uno_finetune_improve.py")
PREPROCESS_SCRIPT = os.path.join(UNO_DIR, "uno_preprocess_improve.py")
INFER_SCRIPT      = os.path.join(UNO_DIR, "uno_infer_improve.py")

MODEL_FILE_NAME = "model.h5"

# Training length (HPO trials AND final training)
FINETUNE_EPOCHS         = 200
EARLY_STOPPING_PATIENCE = 20


def pretrained_model_dir(split_type, i):
    """Directory holding the cell_line_only_training UNO model and scalers for CCL split i."""
    return PRETRAIN_DIR_TEMPLATE.format(split_type=split_type, i=i)


def get_run_paths(split_type):
    """
    Resolve every split_type-dependent path in one place.

    Returns (base_split_dir, base_output_dir, pretrain_dir, global_summary_file).
    pretrain_dir still contains an "{i}" placeholder for the CCL split.
    """
    base_split_dir = os.path.join(SPLITS_ROOT, split_type)

    base_output_dir = os.path.join(BASE_OUTPUT_ROOT, split_type)

    pretrain_dir = pretrained_model_dir(split_type, "{i}")

    global_summary_file = os.path.join(base_output_dir, "results_summary.jsonl")

    return base_split_dir, base_output_dir, pretrain_dir, global_summary_file

############################################################
# GPU CONFIG
############################################################

def detect_gpus():
    """
    Return the GPUs available on this machine.

    If CUDA_VISIBLE_DEVICES is set, only those GPUs are used.
    Otherwise every GPU reported by nvidia-smi is used.
    If no GPU is found, a single CPU worker ("cpu") is used.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if visible is not None:
        gpus = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
        return gpus or ["cpu"]

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        )
        count = len([
            line for line in result.stdout.splitlines()
            if line.startswith("GPU")
        ])
    except OSError:
        count = 0

    if count == 0:
        return ["cpu"]

    return [str(gpu) for gpu in range(count)]


GPUS = detect_gpus()

# Number of jobs that share each GPU. Defaults to 1.
# Set the JOBS_PER_GPU environment variable to run more.
JOBS_PER_GPU    = int(os.environ.get("JOBS_PER_GPU", "1"))
MAX_ACTIVE_JOBS = len(GPUS) * JOBS_PER_GPU

############################################################
# FIXED FREEZE CONFIG
############################################################
# Both the cancer encoder and the drug encoder are ALWAYS frozen.
# Requires a "canc_drug": CANC_LAYERS | DRUG_LAYERS entry in
# FREEZE_GROUPS in uno_finetune_improve.py.
FIXED_FREEZE_GROUP = "canc_drug"

############################################################
# DIFFERENTIAL EVOLUTION CONFIG
############################################################
# Search space (4 dims, every job):
#   - batch size:     categorical  [4, 8, 16]
#   - learning rate:  continuous   1e-5 to 1e-3 (log10 scale)
#   - freeze_num:     categorical  [0, 1, 2, 3] = number of last
#                     post-concat interaction layers left UNFROZEN.
#                     Realized via freeze_post_concat_n = 5 - freeze_num.
#   - dropout rate:   categorical  [0, 0.1, 0.2, 0.4, 0.6, 0.9]

BATCH_SIZE_CHOICES = [4, 8, 16]
BATCH_INDEX_BOUNDS = (0.0, float(len(BATCH_SIZE_CHOICES)))

LOG10_LEARNING_RATE_BOUNDS = (-5.0, -3.0)   # 1e-5 to 1e-3

FREEZE_NUM_CHOICES = [0, 1, 2, 3]
FREEZE_NUM_INDEX_BOUNDS = (0.0, float(len(FREEZE_NUM_CHOICES)))

POST_CONCAT_TOTAL = 5  # dense_6..dense_10 in uno_finetune_improve.py

DROPOUT_CHOICES = [0, 0.1, 0.2, 0.4, 0.6, 0.9]
DROPOUT_INDEX_BOUNDS = (0.0, float(len(DROPOUT_CHOICES)))

HPO_METRIC = "val_mse"      # minimised by DE; must be a key in val_scores.json

DE_MAXITER     = 5
DE_POPSIZE     = 5
DE_TOL         = 0.01
DE_SEED        = 42
DE_STRATEGY    = "best1bin"
DE_MUTATION    = (0.5, 1.0)
DE_RECOMBINATION = 0.7

DELETE_HPO_TRIALS = True

############################################################
# HELPERS
############################################################

# The Python of the environment running this script.
PYTHON = sys.executable


def get_de_bounds():
    return [
        BATCH_INDEX_BOUNDS,
        LOG10_LEARNING_RATE_BOUNDS,
        FREEZE_NUM_INDEX_BOUNDS,
        DROPOUT_INDEX_BOUNDS,
    ]


def check_pretrain_scalers(pretrain_dir, split_type):
    """
    Fail fast (before any GPU job) if any cell_line_only_training split is missing its
    gene-expression or Mordred scaler.
    """
    missing = []
    for i in range(PRETRAIN_N_SPLITS):
        split_dir = pretrained_model_dir(split_type, i)
        for name in PRETRAIN_SCALER_NAMES:
            if not os.path.isfile(os.path.join(split_dir, name)):
                missing.append(os.path.join(split_dir, name))

    if missing:
        raise FileNotFoundError(
            "Pretrain scalers missing for split_type={!r}:\n  {}\n"
            "Run cell_line_only_training/run_uno.py first.".format(split_type, "\n  ".join(missing))
        )

    print(
        "[PRETRAIN SCALERS OK] split_type={!r}: all {} splits have both scalers.".format(
            split_type, PRETRAIN_N_SPLITS
        ),
        flush=True,
    )


def detect_pdo_split_count(study, base_split_dir):
    """
    Count how many PDO folds exist for a study by looking for
    {study}_split_{j}_val_ids.txt files in base_split_dir.
    """
    j = 0
    while os.path.isfile(
        os.path.join(base_split_dir, "{}_split_{}_val_ids.txt".format(study, j))
    ):
        j += 1

    if j == 0:
        raise FileNotFoundError(
            "No PDO split val files found for study '{}' in {}".format(
                study, base_split_dir
            )
        )

    return j


def setup_raw_dir(raw_dir, train_ids, val_ids, test_ids, data_root):
    """
    Build the raw_data/splits symlink structure that uno_preprocess_improve.py
    expects:
        raw_dir/splits/gCSI_split_0_{train,val,test}.txt  (symlinks)
        raw_dir/x_data  ->  data_root/x_data
        raw_dir/y_data  ->  data_root/y_data
    """
    split_dir = os.path.join(raw_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    for src, name in [
        (train_ids, "gCSI_split_0_train.txt"),
        (val_ids,   "gCSI_split_0_val.txt"),
        (test_ids,  "gCSI_split_0_test.txt"),
    ]:
        dst = os.path.join(split_dir, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    for sub in ("x_data", "y_data"):
        dst = os.path.join(raw_dir, sub)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(data_root, sub), dst)


def build_train_cmd(input_dir, pretrained_model, output_dir,
                    freeze_post_concat_n, learning_rate, batch_size, dropout_rate):
    return [
        PYTHON, "-u", TRAIN_SCRIPT,
        "--input_dir",               input_dir,
        "--input_model_dir",         pretrained_model,
        "--output_dir",              output_dir,
        "--freeze_group",            FIXED_FREEZE_GROUP,
        "--freeze_post_concat_n",    str(freeze_post_concat_n),
        "--finetune_lr",             "{:.12g}".format(learning_rate),
        "--batch_size",              str(batch_size),
        "--dropout",                 "{:.12g}".format(dropout_rate),
        "--finetune_epochs",         str(FINETUNE_EPOCHS),
        "--early_stopping_patience", str(EARLY_STOPPING_PATIENCE),
    ]


def run_cmd(cmd, log_file=None, env=None):
    """Run a subprocess, stream output to the console, and optionally log it."""

    start = time.time()

    if log_file is None:
        subprocess.run(cmd, check=True, env=env)
        return time.time() - start

    log_parent = os.path.dirname(log_file)
    if log_parent:
        os.makedirs(log_parent, exist_ok=True)

    with open(log_file, "a") as log_handle:
        header = (
            "\n"
            + "=" * 80
            + "\n"
            + "TIME: {}\n".format(datetime.now())
            + "COMMAND: {}\n".format(" ".join(cmd))
            + "=" * 80
            + "\n"
        )

        print(header, end="", flush=True)
        log_handle.write(header)
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )

        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
        finally:
            if process.stdout is not None:
                process.stdout.close()

        return_code = process.wait()
        elapsed = time.time() - start

        footer = (
            "\nELAPSED: {:.2f} sec\n".format(elapsed)
            + "RETURN CODE: {}\n".format(return_code)
            + "=" * 80
            + "\n"
        )

        print(footer, end="", flush=True)
        log_handle.write(footer)
        log_handle.flush()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    return elapsed


def read_validation_score(score_file):
    with open(score_file, "r") as f:
        scores = json.load(f)

    if HPO_METRIC not in scores:
        raise KeyError(
            "Metric {!r} not found in {}. Available metrics: {}".format(
                HPO_METRIC, score_file, list(scores.keys())
            )
        )

    score = float(scores[HPO_METRIC])

    if not math.isfinite(score):
        raise ValueError(
            "Non-finite {} in {}: {}".format(HPO_METRIC, score_file, score)
        )

    return score


def load_scores_dict(score_file):
    """
    Load a full *_scores.json file as a dict. Returns None if the file
    doesn't exist (e.g. no "mixed" split for Pilot_1).
    """
    if not os.path.exists(score_file):
        return None

    with open(score_file, "r") as f:
        return json.load(f)


def append_hpo_history(history_file, record):
    with open(history_file, "a") as f:
        json.dump(record, f)
        f.write("\n")


def decode_params(x):
    """Decode the continuous 4-dim DE vector into UNO hyperparameters."""

    batch_index = int(math.floor(float(x[0])))
    batch_index = max(0, min(len(BATCH_SIZE_CHOICES) - 1, batch_index))
    batch_size  = BATCH_SIZE_CHOICES[batch_index]

    log10_learning_rate = float(x[1])
    learning_rate = 10.0 ** log10_learning_rate

    freeze_num_index = int(math.floor(float(x[2])))
    freeze_num_index = max(0, min(len(FREEZE_NUM_CHOICES) - 1, freeze_num_index))
    freeze_num = FREEZE_NUM_CHOICES[freeze_num_index]
    freeze_post_concat_n = POST_CONCAT_TOTAL - freeze_num

    dropout_index = int(math.floor(float(x[3])))
    dropout_index = max(0, min(len(DROPOUT_CHOICES) - 1, dropout_index))
    dropout_rate  = DROPOUT_CHOICES[dropout_index]

    return {
        "batch_index":           batch_index,
        "batch_size":            batch_size,
        "learning_rate":         learning_rate,
        "log10_learning_rate":   log10_learning_rate,
        "freeze_num_index":      freeze_num_index,
        "freeze_num":            freeze_num,
        "freeze_post_concat_n":  freeze_post_concat_n,
        "dropout_index":         dropout_index,
        "dropout_rate":          dropout_rate,
        "freeze_group":          FIXED_FREEZE_GROUP,
    }


def job_complete(run_dir, study):
    required = [
        MODEL_FILE_NAME,
        "best_hyperparameters.json",
        "timing.json",
        "study/test_scores.json",
        "train/test_scores.json",
        "performance_summary.json",
    ]

    if study not in SEPERATE_STUDIES:
        required.append("mixed/test_scores.json")

    return all(
        os.path.exists(os.path.join(run_dir, path))
        for path in required
    )


############################################################
# PERFORMANCE SUMMARY HELPERS
############################################################


def build_performance_summary(
    run_dir, study, ccl_fold, pdo_fold, pdo_split_index, best_params, split_type,
):
    """
    Collect train/val/test(/mixed) performance for one run into a single
    dict, and write it to run_dir/performance_summary.json.
    """
    split_files = {
        "train": os.path.join(run_dir, "train", "test_scores.json"),
        "val":   os.path.join(run_dir, "val_scores.json"),
        "test":  os.path.join(run_dir, "study", "test_scores.json"),
    }

    if study not in SEPERATE_STUDIES:
        split_files["test_mixed"] = os.path.join(run_dir, "mixed", "test_scores.json")

    performance = {
        split_name: load_scores_dict(path)
        for split_name, path in split_files.items()
    }

    summary = {
        "study":                     study,
        "split_type":                split_type,
        "ccl_fold":                  ccl_fold,
        "pdo_fold":                  pdo_fold,
        "pdo_split_index":           pdo_split_index,
        "best_batch_size":           best_params["batch_size"],
        "best_learning_rate":        best_params["learning_rate"],
        "freeze_group":              best_params["freeze_group"],
        "best_freeze_num":           best_params["freeze_num"],
        "best_freeze_post_concat_n": best_params["freeze_post_concat_n"],
        "best_dropout_rate":         best_params["dropout_rate"],
        "performance":               performance,
    }

    summary_file = os.path.join(run_dir, "performance_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def append_global_summary(summary, summary_lock, global_summary_file):
    """
    Flatten one performance_summary dict into one row per available split
    (train/val/test/test_mixed) and append those rows to the shared
    results_summary.jsonl for this split_type.
    """
    rows = []
    for split_name, metrics in summary["performance"].items():
        if metrics is None:
            continue

        row = {
            "study":                     summary["study"],
            "split_type":                summary["split_type"],
            "ccl_fold":                  summary["ccl_fold"],
            "pdo_fold":                  summary["pdo_fold"],
            "pdo_split_index":           summary["pdo_split_index"],
            "split":                     split_name,
            "best_batch_size":           summary["best_batch_size"],
            "best_learning_rate":        summary["best_learning_rate"],
            "freeze_group":              summary["freeze_group"],
            "best_freeze_num":           summary["best_freeze_num"],
            "best_freeze_post_concat_n": summary["best_freeze_post_concat_n"],
            "best_dropout_rate":         summary["best_dropout_rate"],
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        return

    summary_parent = os.path.dirname(global_summary_file)
    if summary_parent:
        os.makedirs(summary_parent, exist_ok=True)

    with summary_lock:
        with open(global_summary_file, "a") as f:
            for row in rows:
                json.dump(row, f)
                f.write("\n")


############################################################
# HYPERPARAMETER OPTIMIZATION
############################################################


def saved_best_params_matches_search_space(saved):
    """
    Defensive check for the HPO resume shortcut. A best_hyperparameters.json
    from a DIFFERENT search space or training setup must never be trusted.
    """
    required_keys = (
        "batch_size", "learning_rate", "freeze_group", "freeze_num",
        "freeze_post_concat_n", "dropout_rate",
    )
    if not all(key in saved for key in required_keys):
        return False

    if saved["batch_size"] not in BATCH_SIZE_CHOICES:
        return False

    if saved["freeze_group"] != FIXED_FREEZE_GROUP:
        return False

    if saved["freeze_num"] not in FREEZE_NUM_CHOICES:
        return False

    if saved["freeze_post_concat_n"] != POST_CONCAT_TOTAL - saved["freeze_num"]:
        return False

    if saved["dropout_rate"] not in DROPOUT_CHOICES:
        return False

    if saved.get("finetune_epochs") != FINETUNE_EPOCHS:
        return False

    if saved.get("early_stopping_patience") != EARLY_STOPPING_PATIENCE:
        return False

    return True


def optimize_hyperparameters(
    run_dir,
    processed_study_dir,
    pretrained_model,
    gpu_id,
    gpu_env,
):
    best_params_file = os.path.join(run_dir, "best_hyperparameters.json")

    # Resume: skip HPO if best params already saved AND they match the
    # CURRENT search space / training setup.
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            saved = json.load(f)

        if saved_best_params_matches_search_space(saved):
            print(
                "[HPO RESUME] batch_size={} learning_rate={} "
                "freeze_group={!r} freeze_num={} freeze_post_concat_n={} "
                "dropout_rate={}".format(
                    saved["batch_size"],
                    saved["learning_rate"],
                    saved["freeze_group"],
                    saved["freeze_num"],
                    saved["freeze_post_concat_n"],
                    saved["dropout_rate"],
                ),
                flush=True,
            )

            return saved

        print(
            "[HPO RESUME SKIPPED] {} does not match the current search "
            "space / training setup — re-running HPO from scratch.".format(
                best_params_file
            ),
            flush=True,
        )

    hpo_dir      = os.path.join(run_dir, "hpo_trials")
    history_file = os.path.join(run_dir, "hpo_history.jsonl")

    if os.path.exists(hpo_dir):
        shutil.rmtree(hpo_dir)

    if os.path.exists(history_file):
        os.remove(history_file)

    os.makedirs(hpo_dir, exist_ok=True)

    evaluation_number = 0

    def objective(x):
        nonlocal evaluation_number
        evaluation_number += 1

        decoded               = decode_params(x)
        batch_index           = decoded["batch_index"]
        batch_size            = decoded["batch_size"]
        learning_rate         = decoded["learning_rate"]
        log10_learning_rate   = decoded["log10_learning_rate"]
        freeze_num_index      = decoded["freeze_num_index"]
        freeze_num            = decoded["freeze_num"]
        freeze_post_concat_n  = decoded["freeze_post_concat_n"]
        dropout_index         = decoded["dropout_index"]
        dropout_rate          = decoded["dropout_rate"]

        trial_dir = os.path.join(
            hpo_dir, "eval_{:04d}".format(evaluation_number)
        )
        os.makedirs(trial_dir, exist_ok=True)

        print(
            "[HPO] eval={} batch_size={} learning_rate={:.12g} "
            "log10_lr={:.6f} freeze_group={!r} freeze_num={} "
            "freeze_post_concat_n={} dropout_rate={} GPU={}".format(
                evaluation_number,
                batch_size,
                learning_rate,
                log10_learning_rate,
                FIXED_FREEZE_GROUP,
                freeze_num,
                freeze_post_concat_n,
                dropout_rate,
                gpu_id,
            ),
            flush=True,
        )

        base_record = {
            "evaluation":            evaluation_number,
            "batch_size":            batch_size,
            "batch_index":           batch_index,
            "learning_rate":         learning_rate,
            "log10_learning_rate":   log10_learning_rate,
            "freeze_group":          FIXED_FREEZE_GROUP,
            "freeze_num_index":      freeze_num_index,
            "freeze_num":            freeze_num,
            "freeze_post_concat_n":  freeze_post_concat_n,
            "dropout_index":         dropout_index,
            "dropout_rate":          dropout_rate,
        }

        trial_elapsed = None

        try:
            trial_elapsed = run_cmd(
                build_train_cmd(
                    processed_study_dir, pretrained_model, trial_dir,
                    freeze_post_concat_n, learning_rate,
                    batch_size, dropout_rate,
                ),
                log_file=os.path.join(trial_dir, "subprocess.log"),
                env=gpu_env,
            )

            score = read_validation_score(os.path.join(trial_dir, "val_scores.json"))

            record = dict(base_record)
            record.update({
                HPO_METRIC:        score,
                "elapsed_seconds": trial_elapsed,
                "status":          "ok",
            })
            append_hpo_history(history_file, record)

            print(
                "[HPO RESULT] eval={} {}={}".format(
                    evaluation_number, HPO_METRIC, score
                ),
                flush=True,
            )

            return score

        except Exception as e:
            record = dict(base_record)
            record.update({
                HPO_METRIC:        None,
                "elapsed_seconds": trial_elapsed,
                "status":          "error",
                "error":           str(e),
            })
            append_hpo_history(history_file, record)

            print(
                "[HPO ERROR] eval={}: {}".format(evaluation_number, e),
                flush=True,
            )

            return float("inf")

        finally:
            if DELETE_HPO_TRIALS:
                shutil.rmtree(trial_dir, ignore_errors=True)

    try:
        result = differential_evolution(
            objective,
            bounds=get_de_bounds(),
            strategy=DE_STRATEGY,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            mutation=DE_MUTATION,
            recombination=DE_RECOMBINATION,
            polish=False,
            init="latinhypercube",
            seed=DE_SEED,
            disp=False,
        )
    finally:
        if DELETE_HPO_TRIALS:
            shutil.rmtree(hpo_dir, ignore_errors=True)

    if not math.isfinite(result.fun):
        raise RuntimeError(
            "Differential evolution did not produce a finite validation score."
        )

    decoded_best = decode_params(result.x)

    best_params = {
        "batch_size":              decoded_best["batch_size"],
        "batch_index":             decoded_best["batch_index"],
        "learning_rate":           decoded_best["learning_rate"],
        "log10_learning_rate":     decoded_best["log10_learning_rate"],
        "freeze_group":            decoded_best["freeze_group"],
        "freeze_num_index":        decoded_best["freeze_num_index"],
        "freeze_num":              decoded_best["freeze_num"],
        "freeze_post_concat_n":    decoded_best["freeze_post_concat_n"],
        "dropout_index":           decoded_best["dropout_index"],
        "dropout_rate":            decoded_best["dropout_rate"],
        HPO_METRIC:                float(result.fun),
        "finetune_epochs":         FINETUNE_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "nfev":                    int(result.nfev),
        "nit":                     int(result.nit),
        "success":                 bool(result.success),
        "message":                 str(result.message),
        "search_space": {
            "batch_size_choices":         BATCH_SIZE_CHOICES,
            "batch_encoded_bounds":       list(BATCH_INDEX_BOUNDS),
            "log10_learning_rate":        list(LOG10_LEARNING_RATE_BOUNDS),
            "learning_rate": [
                10.0 ** LOG10_LEARNING_RATE_BOUNDS[0],
                10.0 ** LOG10_LEARNING_RATE_BOUNDS[1],
            ],
            "fixed_freeze_group":         FIXED_FREEZE_GROUP,
            "freeze_num_choices":         FREEZE_NUM_CHOICES,
            "freeze_num_encoded_bounds":  list(FREEZE_NUM_INDEX_BOUNDS),
            "post_concat_total":          POST_CONCAT_TOTAL,
            "dropout_choices":            DROPOUT_CHOICES,
            "dropout_encoded_bounds":     list(DROPOUT_INDEX_BOUNDS),
        },
        "de_settings": {
            "strategy":      DE_STRATEGY,
            "maxiter":       DE_MAXITER,
            "popsize":       DE_POPSIZE,
            "tol":           DE_TOL,
            "mutation":      list(DE_MUTATION),
            "recombination": DE_RECOMBINATION,
            "seed":          DE_SEED,
            "polish":        False,
            "workers":       1,
        },
    }

    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=2)

    print(
        "[HPO BEST] batch_size={} learning_rate={:.12g} "
        "freeze_group={!r} freeze_num={} freeze_post_concat_n={} "
        "dropout_rate={} {}={}".format(
            best_params["batch_size"],
            best_params["learning_rate"],
            best_params["freeze_group"],
            best_params["freeze_num"],
            best_params["freeze_post_concat_n"],
            best_params["dropout_rate"],
            HPO_METRIC,
            best_params[HPO_METRIC],
        ),
        flush=True,
    )

    return best_params


############################################################
# SINGLE FOLD JOB
############################################################


def run_transfer_job(
    study, ccl_fold, pdo_fold,
    split_type, base_split_dir, base_output_dir, pretrain_dir, global_summary_file,
    gpu_queue, summary_lock,
):
    """
    Run transfer learning for a single (study, ccl_fold, pdo_fold) triple
    under the given split_type.

    Output directory: <base_output_dir>/{study}/ccl{i}_j{j}/

    Resume: if run_dir already satisfies job_complete(), returns
    immediately without redoing any work.
    """

    i = ccl_fold
    j = pdo_fold
    p = j   # index used for PDO split files

    gpu_id = gpu_queue.get()  # blocks until a GPU slot is free

    try:
        run_dir = os.path.join(
            base_output_dir,
            study,
            "ccl{}_j{}".format(i, j),
        )
        os.makedirs(run_dir, exist_ok=True)

        if job_complete(run_dir, study):
            print(
                "[SKIP DONE] {} split_type={} ccl={} pdo={}".format(
                    study, split_type, i, j
                ),
                flush=True,
            )
            return

        start_time = time.time()

        preprocess_study_time = None
        hpo_time              = None
        final_train_time      = None
        study_infer_time      = None
        preprocess_mixed_time = None
        mixed_infer_time      = None
        preprocess_train_time = None
        train_infer_time      = None

        timing_file = os.path.join(run_dir, "timing.json")

        try:
            print(
                "[START] {} split_type={} ccl={} pdo={} (pdo split files: {}) GPU={}".format(
                    study, split_type, i, j, p, gpu_id
                ),
                flush=True,
            )

            gpu_env = os.environ.copy()
            if gpu_id != "cpu":
                gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_env["PYTHONUNBUFFERED"]     = "1"
            gpu_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

            # ── Pretrained model ───────────────────────────────────────────
            pretrained_model = pretrained_model_dir(split_type, i)

            # ── Split file paths ───────────────────────────────────────────
            train_split = os.path.join(
                base_split_dir, "{}_split_{}_train_ids.txt".format(study, p),
            )
            val_split = os.path.join(
                base_split_dir, "{}_split_{}_val_ids.txt".format(study, p),
            )
            test_split = os.path.join(
                base_split_dir, "{}_split_{}_test_ids.txt".format(study, p),
            )

            required_inputs = [
                os.path.join(pretrained_model, MODEL_FILE_NAME),
                train_split, val_split, test_split,
            ]
            required_inputs += [
                os.path.join(pretrained_model, name) for name in PRETRAIN_SCALER_NAMES
            ]

            mixed_split = None
            if study not in SEPERATE_STUDIES:
                mixed_split = os.path.join(
                    base_split_dir,
                    "{}_ccl_ids.txt".format(study),
                )
                required_inputs.append(mixed_split)

            missing_inputs = [path for path in required_inputs if not os.path.exists(path)]
            if missing_inputs:
                raise FileNotFoundError(
                    "Missing required input files: {}".format(missing_inputs)
                )

            processed_root      = os.path.join(run_dir, "processed")
            processed_study_dir = os.path.join(processed_root, "study")
            processed_mixed_dir = os.path.join(processed_root, "mixed")
            processed_train_dir = os.path.join(processed_root, "train")

            os.makedirs(processed_study_dir, exist_ok=True)

            ####################################################
            # PREPROCESS STUDY TRAIN / VAL / TEST DATA
            # (scalers LOADED from the pretrained model dir, never refit)
            ####################################################

            study_raw_dir = os.path.join(processed_study_dir, "raw_data")
            setup_raw_dir(
                study_raw_dir,
                train_split, val_split, test_split,
                RAW_DATA_DIR,
            )

            preprocess_study_time = run_cmd(
                [
                    PYTHON, "-u", PREPROCESS_SCRIPT,
                    "--input_dir",             study_raw_dir,
                    "--output_dir",            processed_study_dir,
                    "--pretrained_model_dir",  pretrained_model,
                ],
                log_file=os.path.join(run_dir, "preprocess_study_subprocess.log"),
                env=gpu_env,
            )

            ####################################################
            # HYPERPARAMETER OPTIMIZATION
            ####################################################

            hpo_start = time.time()

            best_params = optimize_hyperparameters(
                run_dir=run_dir,
                processed_study_dir=processed_study_dir,
                pretrained_model=pretrained_model,
                gpu_id=gpu_id,
                gpu_env=gpu_env,
            )

            hpo_time = time.time() - hpo_start

            ####################################################
            # FINAL TRAINING WITH THE BEST CONFIGURATION
            ####################################################

            final_train_time = run_cmd(
                build_train_cmd(
                    processed_study_dir, pretrained_model, run_dir,
                    best_params["freeze_post_concat_n"],
                    best_params["learning_rate"],
                    best_params["batch_size"],
                    best_params["dropout_rate"],
                ),
                log_file=os.path.join(run_dir, "transfer_train_subprocess.log"),
                env=gpu_env,
            )

            ####################################################
            # STUDY TEST INFERENCE
            ####################################################

            study_dir = os.path.join(run_dir, "study")
            os.makedirs(study_dir, exist_ok=True)

            study_infer_time = run_cmd(
                [
                    PYTHON, "-u", INFER_SCRIPT,
                    "--input_dir",         processed_study_dir,
                    "--input_data_dir",    processed_study_dir,
                    "--input_model_dir",   run_dir,
                    "--output_dir",        study_dir,
                    "--calc_infer_scores", "true",
                ],
                log_file=os.path.join(study_dir, "infer_subprocess.log"),
                env=gpu_env,
            )

            ####################################################
            # TRAINING SET PERFORMANCE
            ####################################################

            os.makedirs(processed_train_dir, exist_ok=True)

            train_raw_dir = os.path.join(processed_train_dir, "raw_data")
            setup_raw_dir(
                train_raw_dir,
                train_split, val_split, train_split,  # train plays the "test" role
                RAW_DATA_DIR,
            )

            preprocess_train_time = run_cmd(
                [
                    PYTHON, "-u", PREPROCESS_SCRIPT,
                    "--input_dir",             train_raw_dir,
                    "--output_dir",            processed_train_dir,
                    "--pretrained_model_dir",  pretrained_model,
                ],
                log_file=os.path.join(run_dir, "preprocess_train_subprocess.log"),
                env=gpu_env,
            )

            train_dir = os.path.join(run_dir, "train")
            os.makedirs(train_dir, exist_ok=True)

            train_infer_time = run_cmd(
                [
                    PYTHON, "-u", INFER_SCRIPT,
                    "--input_dir",         processed_train_dir,
                    "--input_data_dir",    processed_train_dir,
                    "--input_model_dir",   run_dir,
                    "--output_dir",        train_dir,
                    "--calc_infer_scores", "true",
                ],
                log_file=os.path.join(train_dir, "infer_subprocess.log"),
                env=gpu_env,
            )

            ####################################################
            # MIXED TEST INFERENCE  (study CCL IDs)
            ####################################################

            if study not in SEPERATE_STUDIES:
                os.makedirs(processed_mixed_dir, exist_ok=True)

                mixed_raw_dir = os.path.join(processed_mixed_dir, "raw_data")
                setup_raw_dir(
                    mixed_raw_dir,
                    train_split, val_split, mixed_split,
                    RAW_DATA_DIR,
                )

                preprocess_mixed_time = run_cmd(
                    [
                        PYTHON, "-u", PREPROCESS_SCRIPT,
                        "--input_dir",             mixed_raw_dir,
                        "--output_dir",            processed_mixed_dir,
                        "--pretrained_model_dir",  pretrained_model,
                    ],
                    log_file=os.path.join(run_dir, "preprocess_mixed_subprocess.log"),
                    env=gpu_env,
                )

                mixed_dir = os.path.join(run_dir, "mixed")
                os.makedirs(mixed_dir, exist_ok=True)

                mixed_infer_time = run_cmd(
                    [
                        PYTHON, "-u", INFER_SCRIPT,
                        "--input_dir",         processed_mixed_dir,
                        "--input_data_dir",    processed_mixed_dir,
                        "--input_model_dir",   run_dir,
                        "--output_dir",        mixed_dir,
                        "--calc_infer_scores", "true",
                    ],
                    log_file=os.path.join(mixed_dir, "infer_subprocess.log"),
                    env=gpu_env,
                )

            ####################################################
            # PERFORMANCE SUMMARY
            ####################################################

            performance_summary = build_performance_summary(
                run_dir=run_dir,
                study=study,
                ccl_fold=i,
                pdo_fold=j,
                pdo_split_index=p,
                best_params=best_params,
                split_type=split_type,
            )

            append_global_summary(performance_summary, summary_lock, global_summary_file)

            total_elapsed = time.time() - start_time

            timing = {
                "status":                       "complete",
                "split_type":                   split_type,
                "study":                        study,
                "ccl_fold":                     i,
                "pdo_fold":                     j,
                "pdo_split_index":              p,
                "gpu_id":                       gpu_id,
                "pilot_mode":                   PILOT_MODE,
                "finetune_epochs":              FINETUNE_EPOCHS,
                "early_stopping_patience":      EARLY_STOPPING_PATIENCE,
                "preprocess_study_seconds":     preprocess_study_time,
                "hpo_elapsed_seconds":          hpo_time,
                "final_train_elapsed_seconds":  final_train_time,
                "study_infer_elapsed_seconds":  study_infer_time,
                "preprocess_train_seconds":     preprocess_train_time,
                "train_infer_elapsed_seconds":  train_infer_time,
                "preprocess_mixed_seconds":     preprocess_mixed_time,
                "mixed_infer_elapsed_seconds":  mixed_infer_time,
                "total_elapsed_seconds":        total_elapsed,
                "total_elapsed_minutes":        total_elapsed / 60.0,
                "hpo_evaluations":              best_params.get("nfev"),
                "best_batch_size":              best_params["batch_size"],
                "best_learning_rate":           best_params["learning_rate"],
                "freeze_group":                 best_params["freeze_group"],
                "best_freeze_num":              best_params["freeze_num"],
                "best_freeze_post_concat_n":    best_params["freeze_post_concat_n"],
                "best_dropout_rate":            best_params["dropout_rate"],
            }

            with open(timing_file, "w") as f:
                json.dump(timing, f, indent=2)

            print(
                "[DONE] {} ccl={} pdo={} GPU={} ({:.2f} min)".format(
                    study, i, j, gpu_id, total_elapsed / 60.0
                ),
                flush=True,
            )

        except Exception as e:
            total_elapsed = time.time() - start_time

            error_record = {
                "status":                "error",
                "split_type":            split_type,
                "study":                 study,
                "ccl_fold":              i,
                "pdo_fold":              j,
                "pdo_split_index":       p,
                "gpu_id":                gpu_id,
                "pilot_mode":            PILOT_MODE,
                "total_elapsed_seconds": total_elapsed,
                "total_elapsed_minutes": total_elapsed / 60.0,
                "error":                 str(e),
            }

            with open(timing_file, "w") as f:
                json.dump(error_record, f, indent=2)

            with open(os.path.join(run_dir, "job_error.log"), "w") as f:
                f.write(traceback.format_exc())

            print(
                "[ERROR] {} split_type={} ccl={} pdo={} GPU={}: {}".format(
                    study, split_type, i, j, gpu_id, e
                ),
                flush=True,
            )

            raise

    finally:
        gpu_queue.put(gpu_id)  # always return the GPU to the pool


############################################################
# MAIN
############################################################


def build_job_list(split_type, base_split_dir):
    """
    Build the full (study, ccl_fold, pdo_fold) job list for one split
    type. Cancer-blind PDO split counts are detected per study; random
    uses exactly 10 PDO splits per study.
    """
    if split_type == "cancer-blind":
        print(
            "Detecting per-study PDO split counts from {}...".format(base_split_dir),
            flush=True,
        )
        study_pdo_splits = {}
        for study in STUDIES:
            count = detect_pdo_split_count(study, base_split_dir)
            study_pdo_splits[study] = count
            print("  {}: {} PDO splits".format(study, count), flush=True)
        print("", flush=True)
    else:
        study_pdo_splits = {study: 10 for study in STUDIES}
        print(
            "Random splits: using 10 PDO folds per study (fixed).",
            flush=True,
        )

    jobs = []

    if PILOT_MODE:
        for study in STUDIES:
            jobs.append((study, PILOT_CCL_FOLD, PILOT_PDO_FOLD))
    else:
        for study in STUDIES:
            for ccl_fold in range(10):
                for pdo_fold in range(study_pdo_splits[study]):
                    jobs.append((study, ccl_fold, pdo_fold))

    return jobs


def run_split_type_pipeline(split_type, gpu_queue, summary_lock):
    """
    Run every (study, ccl_fold, pdo_fold) job for one split_type to
    completion. Called once per SPLIT_TYPES entry by main(), in order.
    """
    base_split_dir, base_output_dir, pretrain_dir, global_summary_file = (
        get_run_paths(split_type)
    )

    print("\n" + "#" * 70, flush=True)
    print("# SPLIT TYPE: {}".format(split_type), flush=True)
    print("#" * 70 + "\n", flush=True)

    # Fail fast on missing cell_line_only_training scalers — before any GPU work.
    check_pretrain_scalers(pretrain_dir, split_type)

    jobs = build_job_list(split_type, base_split_dir)

    already_done = 0
    for (study, ccl_fold, pdo_fold) in jobs:
        run_dir = os.path.join(
            base_output_dir, study, "ccl{}_j{}".format(ccl_fold, pdo_fold)
        )
        if job_complete(run_dir, study):
            already_done += 1

    remaining = len(jobs) - already_done

    print(
        "[{}] Total jobs: {} | already complete (resumed): {} | remaining: {}".format(
            split_type, len(jobs), already_done, remaining
        ),
        flush=True,
    )

    if remaining == 0:
        print(
            "[{}] Nothing to do — every job already complete.\n".format(split_type),
            flush=True,
        )
        return

    print(
        "Maximum concurrent jobs: {} ({} per GPU across GPUs {})".format(
            MAX_ACTIVE_JOBS, JOBS_PER_GPU, ", ".join(GPUS)
        ),
        flush=True,
    )
    print(
        "Shared results summary file: {}".format(global_summary_file),
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=MAX_ACTIVE_JOBS) as executor:
        futures = [
            executor.submit(
                run_transfer_job,
                study, ccl_fold, pdo_fold,
                split_type, base_split_dir, base_output_dir, pretrain_dir, global_summary_file,
                gpu_queue, summary_lock,
            )
            for (study, ccl_fold, pdo_fold) in jobs
        ]

        for future in as_completed(futures):
            future.result()

    print("[{}] All jobs complete.\n".format(split_type), flush=True)


def main():
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu in GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gpu)

    summary_lock = manager.Lock()

    max_evals = (DE_MAXITER + 1) * DE_POPSIZE * len(get_de_bounds())

    print("Split types to run, in order: {}".format(SPLIT_TYPES), flush=True)
    print(
        "Fixed freeze group (both encoders, every run): {!r}".format(FIXED_FREEZE_GROUP),
        flush=True,
    )
    print(
        "*** REQUIRES uno_finetune_improve.py's FREEZE_GROUPS to contain "
        "a {!r} entry (CANC_LAYERS | DRUG_LAYERS) before launching. ***".format(
            FIXED_FREEZE_GROUP
        ),
        flush=True,
    )
    print(
        "L2-SP regularization is OFF (not passed to uno_finetune_improve.py at all).",
        flush=True,
    )
    print(
        "Training: up to {} epochs, early stopping patience {}.".format(
            FINETUNE_EPOCHS, EARLY_STOPPING_PATIENCE
        ),
        flush=True,
    )
    print(
        "DE: maxiter={} popsize={} -> up to {} candidates per job.".format(
            DE_MAXITER, DE_POPSIZE, max_evals
        ),
        flush=True,
    )
    print("Batch size choices: {}".format(BATCH_SIZE_CHOICES), flush=True)
    print("Dropout choices: {}".format(DROPOUT_CHOICES), flush=True)

    for split_type in SPLIT_TYPES:
        run_split_type_pipeline(split_type, gpu_queue, summary_lock)

    print(
        "All split types complete: {}".format(SPLIT_TYPES),
        flush=True,
    )


if __name__ == "__main__":
    main()