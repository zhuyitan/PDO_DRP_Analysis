"""Train GraphDRP for drug response prediction."""

import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import (
    DRPTrainConfig,
)
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specific imports
from model_params_def import train_params
from model_utils.torch_utils import (
    build_GraphDRP_dataloader,
    determine_device,
    load_GraphDRP,
    predicting,
    set_GraphDRP,
    train_epoch,
)


filepath = Path(__file__).resolve().parent

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _load_pretrained_weights(model, pretrained_model, device):
    """Load either an IMPROVE checkpoint dict or a plain state_dict."""
    checkpoint = torch.load(pretrained_model, map_location=device)

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    print(
        f"Loaded pretrained model: {pretrained_model}",
        flush=True,
    )


def _configure_transfer_learning(model, params):
    """Apply GIN-only encoder/head freezing and print verification information."""
    freeze_mode = params.get("freeze_layers", "None")
    head_freeze_mode = params.get("freeze_head_layers", "None")
    model_arch = params["model_arch"]

    if model_arch != "GINConvNet":
        if freeze_mode != "None" or head_freeze_mode != "None":
            raise ValueError(
                "Layer freezing is implemented only for GINConvNet. "
                f"Received model_arch={model_arch!r}, "
                f"freeze_layers={freeze_mode!r}, and "
                f"freeze_head_layers={head_freeze_mode!r}."
            )

        print(
            "[Layer freezing] Non-GIN architecture selected; "
            "no layers frozen.",
            flush=True,
        )
        return

    if not hasattr(model, "configure_layer_freezing"):
        raise AttributeError(
            "GINConvNet does not define configure_layer_freezing(). "
            "Confirm that the updated ginconv.py is being imported."
        )

    model.configure_layer_freezing(
        freeze_mode=freeze_mode,
        head_freeze_mode=head_freeze_mode,
        verbose=True,
    )


def _build_optimizer(model, learning_rate):
    """Build an optimizer containing only trainable parameters."""
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        print(
            "[Layer freezing] No trainable parameters remain. "
            "Skipping optimizer creation and PDO adaptation.",
            flush=True,
        )
        return None

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    frozen_in_optimizer = [
        name
        for name, parameter in model.named_parameters()
        if (
            not parameter.requires_grad
            and id(parameter) in optimizer_parameter_ids
        )
    ]

    if frozen_in_optimizer:
        raise RuntimeError(
            "Frozen parameters were added to the optimizer: "
            + ", ".join(frozen_in_optimizer)
        )

    print(
        "[Layer freezing] Optimizer verification passed: "
        f"{len(trainable_parameters)} trainable parameter tensors; "
        "0 frozen tensors in optimizer.",
        flush=True,
    )

    return optimizer


# [Req]
def run(params: Dict) -> Dict:
    """Run model training."""
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"],
    )

    train_data_fname = frm.build_ml_data_file_name(
        data_format=params["data_format"],
        stage="train",
    )
    val_data_fname = frm.build_ml_data_file_name(
        data_format=params["data_format"],
        stage="val",
    )

    # GraphDRP-specific: remove data_format suffix.
    train_data_fname = train_data_fname.split(
        params["data_format"]
    )[0]
    val_data_fname = val_data_fname.split(
        params["data_format"]
    )[0]

    print("\nTrain data:")
    print(f"batch_size: {params['batch_size']}")
    sys.stdout.flush()

    train_loader = build_GraphDRP_dataloader(
        data_dir=params["input_dir"],
        data_fname=train_data_fname,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    print("\nVal data:")
    print(f"val_batch: {params['val_batch']}")
    sys.stdout.flush()

    val_loader = build_GraphDRP_dataloader(
        data_dir=params["input_dir"],
        data_fname=val_data_fname,
        batch_size=params["val_batch"],
        shuffle=False,
    )

    def determine_gene_dim(dataloader):
        sample_data = next(iter(dataloader))
        return sample_data.target.shape[1]

    params["num_genes"] = determine_gene_dim(train_loader)

    device = determine_device(params["cuda_name"])

    # ------------------------------------------------------
    # Prepare model, load pretrained weights, then freeze.
    # ------------------------------------------------------
    model = set_GraphDRP(params, device)

    pretrained_model = params.get("pretrained_model")
    if pretrained_model:
        _load_pretrained_weights(
            model=model,
            pretrained_model=pretrained_model,
            device=device,
        )
    elif (
        params.get("freeze_layers", "None") != "None"
        or params.get("freeze_head_layers", "None") != "None"
    ):
        print(
            "[Layer freezing] WARNING: layers are being frozen "
            "without loading a pretrained checkpoint.",
            flush=True,
        )

    # Freezing must happen after loading the checkpoint and before creating
    # the optimizer.
    _configure_transfer_learning(model, params)

    optimizer = _build_optimizer(
        model=model,
        learning_rate=params["learning_rate"],
    )
    loss_fn = torch.nn.MSELoss()

    initial_epoch = 0
    num_epoch = params["epochs"]
    log_interval = params["log_interval"]
    patience = params["patience"]

    best_score = np.inf
    best_epoch = -1
    early_stop_counter = 0
    early_stop_metric = params["early_stop_metric"]

    epoch_list = []
    val_loss_list = []
    train_loss_list = []
    log_interval_epoch = 5

    print(f"Epochs: {initial_epoch + 1} to {num_epoch}")
    sys.stdout.flush()

    if optimizer is None:
        # Fully frozen model: this is the CCL-pretrained, no-PDO-adaptation
        # condition. Save the loaded state immediately so downstream loading
        # and evaluation use the standard checkpoint path.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "in_dim": model.in_dim,
                "freeze_layers": params.get("freeze_layers", "None"),
                "freeze_head_layers": params.get(
                    "freeze_head_layers",
                    "None",
                ),
                "dropout": params.get("dropout", 0.5),
            },
            modelpath,
        )
        print(
            "[Layer freezing] Fully frozen model saved without "
            "fine-tuning.",
            flush=True,
        )
    else:
        for epoch in range(initial_epoch, num_epoch):
            print(f"Start epoch: {epoch}")

            train_loss = train_epoch(
                model,
                device,
                train_loader,
                optimizer,
                loss_fn,
                epoch + 1,
                log_interval,
            )

            val_true, val_pred = predicting(
                model,
                device,
                val_loader,
            )
            val_scores = compute_metrics(
                val_true,
                val_pred,
                params["metric_type"],
            )

            if epoch % log_interval_epoch == 0:
                epoch_list.append(epoch)
                val_loss_list.append(
                    val_scores[early_stop_metric]
                )

                train_true, train_pred = predicting(
                    model,
                    device,
                    train_loader,
                )
                train_scores = compute_metrics(
                    train_true,
                    train_pred,
                    params["metric_type"],
                )
                train_loss_list.append(
                    train_scores[early_stop_metric]
                )

            print(
                f"{early_stop_metric}, "
                f"{val_scores[early_stop_metric]}"
            )
            sys.stdout.flush()

            if val_scores[early_stop_metric] < best_score:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "in_dim": model.in_dim,
                        "freeze_layers": params.get(
                            "freeze_layers",
                            "None",
                        ),
                        "freeze_head_layers": params.get(
                            "freeze_head_layers",
                            "None",
                        ),
                        "dropout": params.get("dropout", 0.5),
                    },
                    modelpath,
                )

                best_epoch = epoch + 1
                best_score = val_scores[early_stop_metric]

                print(
                    f"{early_stop_metric} improved at epoch "
                    f"{best_epoch}; Best {early_stop_metric}: "
                    f"{best_score}; Model: {params['model_arch']}"
                )
                early_stop_counter = 0
            else:
                print(
                    f"No improvement since epoch {best_epoch}; "
                    f"Best {early_stop_metric}: {best_score}; "
                    f"Model: {params['model_arch']}"
                )
                early_stop_counter += 1

            if early_stop_counter == patience:
                print(
                    "Terminate training (model did not improve on "
                    f"val data for {params['patience']} epochs)."
                )
                print(
                    f"Best epoch: {best_epoch}; "
                    f"Best score ({early_stop_metric}): {best_score}"
                )
                break

    history = pd.DataFrame(
        {
            "epoch": epoch_list,
            "val_loss": val_loss_list,
            "train_loss": train_loss_list,
        }
    )

    model = load_GraphDRP(
        params,
        modelpath,
        device,
    )
    model.eval()

    val_true, val_pred = predicting(
        model,
        device,
        data_loader=val_loader,
    )

    frm.store_predictions_df(
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"],
    )

    val_scores = frm.compute_performance_scores(
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"],
    )

    history.to_csv(
        Path(params["output_dir"]) / "history.csv",
        index=False,
    )

    return val_scores


# [Req]
def main(args):
    additional_definitions = train_params
    cfg = DRPTrainConfig()

    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="graphdrp_params.txt",
        additional_definitions=additional_definitions,
    )

    run(params)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])