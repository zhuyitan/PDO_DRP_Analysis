""" Train LightGBM for drug response prediction.

Required outputs
----------------
All the outputs from this train script are saved in params["output_dir"].

1. Trained model.
   The model is trained with train data and validated with val data. The model
   file name and file format are specified, respectively by
   params["model_file_name"] and params["model_file_format"].
   For LightGBM, the saved model:
        model.txt

2. Predictions on val data.
   Raw model predictions calculated using the trained model on val data.
   The predictions are saved in val_y_data_predicted.csv

3. Prediction performance scores on val data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics. The scores are saved as json in
   val_scores.json
"""

import sys
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import lightgbm as lgb

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specific imports
from model_params_def import train_params  # [Req]
from model_utils.utils import extract_subset_fea

filepath = Path(__file__).resolve().parent  # [Req]


def setup_logger(output_dir):
    """Configure console + file logging."""

    log_file = Path(output_dir) / "training.log"

    logger = logging.getLogger("lgbm_train")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-running
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File logger
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    # Console logger
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# [Req]
def run(params: Dict) -> Dict:
    """Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data.
    """

    logger = setup_logger(params["output_dir"])

    logger.info("=" * 80)
    logger.info("Starting LightGBM training")
    logger.info("=" * 80)

    # ------------------------------------------------------
    # [Req] Build model path
    # ------------------------------------------------------
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )

    logger.info(f"Model output path: {modelpath}")

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(
        data_format=params["data_format"],
        stage="train"
    )

    val_data_fname = frm.build_ml_data_file_name(
        data_format=params["data_format"],
        stage="val"
    )

    logger.info(f"Train data file: {train_data_fname}")
    logger.info(f"Validation data file: {val_data_fname}")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    logger.info("Loading parquet datasets")

    tr_data = pd.read_parquet(
        Path(params["input_dir"]) / train_data_fname
    )

    vl_data = pd.read_parquet(
        Path(params["input_dir"]) / val_data_fname
    )

    logger.info(f"Train dataframe shape: {tr_data.shape}")
    logger.info(f"Validation dataframe shape: {vl_data.shape}")

    fea_list = ["ge", "mordred"]
    fea_sep = "."

    # ------------------------------------------------------
    # Extract features
    # ------------------------------------------------------
    logger.info("Extracting feature subsets")

    # Train data
    xtr = extract_subset_fea(
        tr_data,
        fea_list=fea_list,
        fea_sep=fea_sep
    )

    ytr = tr_data[[params["y_col_name"]]]

    # Validation data
    xvl = extract_subset_fea(
        vl_data,
        fea_list=fea_list,
        fea_sep=fea_sep
    )

    yvl = vl_data[[params["y_col_name"]]]

    logger.info(f"xtr shape: {xtr.shape}")
    logger.info(
        f"First 10 features: "
        f"{list(xtr.columns[:10])}"
    )
    logger.info(f"ytr shape: {ytr.shape}")

    logger.info(f"xvl shape: {xvl.shape}")
    logger.info(f"yvl shape: {yvl.shape}")

    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    logger.info("Initializing LightGBM model")

    ml_init_args = {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "n_jobs": 8,
        "random_state": 42
    }

    logger.info(f"Model init args: {ml_init_args}")

    logger.info("=" * 80)
    logger.info("Beginning training")
    logger.info("=" * 80)

    # ======================================================
    # STANDARD TRAINING
    # ======================================================
    if not params.get("pretrained_model"):

        logger.info("Training from scratch")

        model = lgb.LGBMRegressor(
            objective="regression",
            **ml_init_args
        )

        model.fit(
            xtr,
            ytr.values.ravel(),

            eval_set=[
                (xtr, ytr.values.ravel()),
                (xvl, yvl.values.ravel())
            ],

            eval_names=[
                "train",
                "val"
            ],

            eval_metric="l2",

            verbose=1,

            early_stopping_rounds=500,
        )

        logger.info("=" * 80)
        logger.info("Training complete")
        logger.info("=" * 80)

        logger.info(
            f"Best iteration: {model.best_iteration_}"
        )

        if hasattr(model, "best_score_"):
            logger.info(
                f"Best scores: {model.best_score_}"
            )

        logger.info("Saving model")

        model.booster_.save_model(str(modelpath))

        del model

    # ======================================================
    # TRANSFER LEARNING
    # ======================================================
    else:

        logger.info(
            f"Loading pretrained model: "
            f"{params['pretrained_model']}"
        )

        train_set = lgb.Dataset(
            xtr,
            label=ytr.values.ravel()
        )

        val_set = lgb.Dataset(
            xvl,
            label=yvl.values.ravel()
        )

        booster = lgb.train(
            params={
                "objective": "regression",
                "learning_rate": params["learning_rate"],
                "num_leaves": params["num_leaves"],
                "max_depth": params["max_depth"],
                "min_data_in_leaf": params["min_data_in_leaf"],
                "num_threads": 8
            },

            train_set=train_set,

            valid_sets=[val_set],

            init_model=params["pretrained_model"],

            num_boost_round=params.get(
                "transfer_rounds",
                1000
            ),

            callbacks=[
                lgb.early_stopping(50)
            ]
        )

        logger.info("Transfer learning complete")

        logger.info("Saving transferred model")

        booster.save_model(str(modelpath))

        del booster

    logger.info(f"Model saved to: {modelpath}")
    # logger.info("Initializing LightGBM model")
    
    if params.get("pretrained_model"):

        source_model = lgb.Booster(
            model_file=params["pretrained_model"]
        )

        transferred_model = lgb.Booster(
            model_file=str(modelpath)
        )

        logger.info(
            f"Source trees: {source_model.num_trees()}"
        )

        logger.info(
            f"Transferred trees: {transferred_model.num_trees()}"
        )

    

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    logger.info("Reloading saved model")

    model = lgb.Booster(
        model_file=str(modelpath)
    )

    logger.info("Generating validation predictions")

    val_pred = model.predict(xvl)
    val_true = yvl.values.squeeze()

    logger.info(
        f"Validation predictions shape: {val_pred.shape}"
    )

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    logger.info("Saving validation predictions")

    frm.store_predictions_df(
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    logger.info("Computing validation performance metrics")

    val_scores = frm.compute_performance_scores(
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    logger.info(
        f"Validation scores: {val_scores}"
    )

    logger.info("=" * 80)
    logger.info("Finished model training")
    logger.info("=" * 80)

    return val_scores


# [Req]
def main(args):

    # [Req]
    additional_definitions = train_params

    cfg = DRPTrainConfig()

    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="lgbm_params.txt",
        additional_definitions=additional_definitions
    )

    val_scores = run(params)

    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])