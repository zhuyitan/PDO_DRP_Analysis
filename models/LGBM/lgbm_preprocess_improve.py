"""
Preprocess benchmark data (e.g., CSA data) to generate datasets for the
LightGBM prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["output_dir"].

1. Model input data files.
   This script creates three data files corresponding to train, validation,
   and test data. These data files are used as inputs to the ML/DL model in
   the train and infer scripts. The file format is specified by
   params["data_format"].

2. Y data files.
   The script creates dataframes with true y values and additional metadata.
   Generated files:
        train_y_data.csv, val_y_data.csv, and test_y_data.csv.

Pretrained scaler support
-------------------------
If both of the following parameters are provided:

    pretrained_ge_scaler
    pretrained_md_scaler

the saved scaler objects from pretraining are loaded and reused instead of
fitting new scalers on the current dataset.

If neither is provided, new scalers are fit using train + validation data,
which preserves the original preprocessing behavior.
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import joblib

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import (
    DRPPreprocessConfig,
)
import improvelib.utils as frm

# Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specific imports
from model_params_def import preprocess_params
from model_utils.utils import gene_selection, scale_df


filepath = Path(__file__).resolve().parent


def _optional_path(value):
    """
    Convert an optional config value to a Path.

    Config files sometimes represent None as a string, so handle those values
    explicitly.
    """
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"", "none", "null"}:
        return None

    return Path(value).expanduser()


def _align_features_to_scaler(
    df: pd.DataFrame,
    id_col: str,
    scaler,
    scaler_name: str,
) -> pd.DataFrame:
    """
    Align dataframe feature columns with the features used when a pretrained
    scaler was originally fit.

    If the sklearn scaler contains feature_names_in_, those names define the
    canonical feature set and order.

    Extra features in the current dataset are dropped. Missing features cause
    an error because the pretrained scaler/model cannot safely operate without
    them.

    If feature_names_in_ is unavailable, n_features_in_ is used as a weaker
    compatibility check.
    """

    feature_cols = [col for col in df.columns if col != id_col]

    # ------------------------------------------------------
    # Best case: sklearn stored the actual feature names.
    # ------------------------------------------------------
    if hasattr(scaler, "feature_names_in_"):

        expected_features = list(scaler.feature_names_in_)

        missing_features = [
            col for col in expected_features
            if col not in df.columns
        ]

        if missing_features:
            preview = missing_features[:20]

            raise ValueError(
                f"{scaler_name} scaler expects "
                f"{len(expected_features)} features, but "
                f"{len(missing_features)} are missing from the current data.\n"
                f"First missing features: {preview}"
            )

        extra_features = [
            col for col in feature_cols
            if col not in expected_features
        ]

        if extra_features:
            print(
                f"{scaler_name}: dropping "
                f"{len(extra_features)} features that were not present "
                f"during pretraining."
            )

        print(
            f"{scaler_name}: using {len(expected_features)} "
            f"pretraining features."
        )

        # Critically, preserve the exact order seen during pretraining.
        return df[[id_col] + expected_features].copy()

    # ------------------------------------------------------
    # Older scaler or scaler fit on a NumPy array.
    # We can verify feature count, but not feature names/order.
    # ------------------------------------------------------
    if hasattr(scaler, "n_features_in_"):

        expected_count = scaler.n_features_in_

        if len(feature_cols) != expected_count:
            raise ValueError(
                f"{scaler_name} scaler expects {expected_count} features, "
                f"but the current dataframe contains "
                f"{len(feature_cols)} features."
            )

        print(
            f"{scaler_name}: scaler has no feature_names_in_; "
            f"verified feature count ({expected_count}) only."
        )

    else:
        print(
            f"WARNING: {scaler_name} scaler does not expose "
            f"feature_names_in_ or n_features_in_. "
            f"Feature compatibility cannot be verified."
        )

    return df


# [Req]
def run(params: Dict):
    """
    Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed
            ML data files.
    """

    # ------------------------------------------------------
    # [Req] Load X data
    # ------------------------------------------------------
    print("\nLoad omics data.")

    omics_obj = omics_utils.OmicsLoader(params)
    ge = omics_obj.dfs["cancer_gene_expression.tsv"]

    print("\nLoad drugs data.")

    drugs_obj = drugs_utils.DrugsLoader(params)
    md = drugs_obj.dfs["drug_mordred.tsv"]

    # TODO: implement reset_index() inside loader
    md = md.reset_index()

    # ------------------------------------------------------
    # Further preprocess X data
    # ------------------------------------------------------

    # Gene selection
    if params["use_lincs"]:
        genes_fpath = filepath / "model_utils/landmark_genes.txt"

        ge = gene_selection(
            ge,
            genes_fpath,
            canc_col_name=params["canc_col_name"],
        )

    # Prefix gene-expression columns with "ge."
    fea_sep = "."
    fea_prefix = "ge"

    ge = ge.rename(
        columns={
            fea: f"{fea_prefix}{fea_sep}{fea}"
            for fea in ge.columns[1:]
        }
    )

    # ------------------------------------------------------
    # Determine scaler mode
    # ------------------------------------------------------
    pretrained_ge_scaler_path = _optional_path(
        params.get("pretrained_ge_scaler")
    )

    pretrained_md_scaler_path = _optional_path(
        params.get("pretrained_md_scaler")
    )

    # Require both pretrained scalers or neither.
    if (
        pretrained_ge_scaler_path is None
    ) != (
        pretrained_md_scaler_path is None
    ):
        raise ValueError(
            "Pretrained scaler configuration is incomplete.\n"
            "Provide both:\n"
            "  pretrained_ge_scaler\n"
            "  pretrained_md_scaler\n"
            "or provide neither."
        )

    use_pretrained_scalers = (
        pretrained_ge_scaler_path is not None
        and pretrained_md_scaler_path is not None
    )

    # ======================================================
    # PRETRAINED SCALERS
    # ======================================================
    if use_pretrained_scalers:

        print("\nUse pretrained feature scalers.")

        if not pretrained_ge_scaler_path.exists():
            raise FileNotFoundError(
                "Pretrained gene-expression scaler not found: "
                f"{pretrained_ge_scaler_path}"
            )

        if not pretrained_md_scaler_path.exists():
            raise FileNotFoundError(
                "Pretrained Mordred scaler not found: "
                f"{pretrained_md_scaler_path}"
            )

        print(
            "Load gene-expression scaler:",
            pretrained_ge_scaler_path,
        )

        ge_scaler = joblib.load(pretrained_ge_scaler_path)

        print(
            "Load Mordred scaler:        ",
            pretrained_md_scaler_path,
        )

        md_scaler = joblib.load(pretrained_md_scaler_path)

        # --------------------------------------------------
        # Match current features to the features/order used
        # during pretraining.
        # --------------------------------------------------
        ge = _align_features_to_scaler(
            df=ge,
            id_col=params["canc_col_name"],
            scaler=ge_scaler,
            scaler_name="Gene expression",
        )

        md = _align_features_to_scaler(
            df=md,
            id_col=params["drug_col_name"],
            scaler=md_scaler,
            scaler_name="Mordred",
        )

    # ======================================================
    # NEW SCALERS
    # ======================================================
    else:

        print("\nCreate feature scalers from current train + val data.")

        # Load train and validation responses.
        rsp_tr = drp.DrugResponseLoader(
            params,
            split_file=params["train_split_file"],
            verbose=False,
        ).dfs["response.tsv"]

        rsp_vl = drp.DrugResponseLoader(
            params,
            split_file=params["val_split_file"],
            verbose=False,
        ).dfs["response.tsv"]

        rsp = pd.concat(
            [rsp_tr, rsp_vl],
            axis=0,
        )

        # --------------------------------------------------
        # Retain feature rows represented in response data.
        # --------------------------------------------------
        rsp = rsp.merge(
            ge[[params["canc_col_name"]]],
            on=params["canc_col_name"],
            how="inner",
        )

        rsp = rsp.merge(
            md[[params["drug_col_name"]]],
            on=params["drug_col_name"],
            how="inner",
        )

        ge_sub = ge[
            ge[params["canc_col_name"]].isin(
                rsp[params["canc_col_name"]]
            )
        ].reset_index(drop=True)

        md_sub = md[
            md[params["drug_col_name"]].isin(
                rsp[params["drug_col_name"]]
            )
        ].reset_index(drop=True)

        # --------------------------------------------------
        # Fit gene-expression scaler
        # --------------------------------------------------
        _, ge_scaler = scale_df(
            ge_sub,
            scaler_name=params["scaling"],
        )

        # --------------------------------------------------
        # Fit Mordred scaler
        # --------------------------------------------------
        _, md_scaler = scale_df(
            md_sub,
            scaler_name=params["scaling"],
        )

        del rsp
        del rsp_tr
        del rsp_vl
        del ge_sub
        del md_sub

    # ------------------------------------------------------
    # Save the scalers used for this preprocessing run.
    #
    # For transfer learning these are copies of the loaded
    # pretrained scalers, not newly fitted scalers.
    # ------------------------------------------------------
    ge_scaler_fpath = (
        Path(params["output_dir"])
        / params["ge_scaler_fname"]
    )

    md_scaler_fpath = (
        Path(params["output_dir"])
        / params["md_scaler_fname"]
    )

    joblib.dump(
        ge_scaler,
        ge_scaler_fpath,
    )

    joblib.dump(
        md_scaler,
        md_scaler_fpath,
    )

    print(
        "\nScaler object for gene expression:",
        ge_scaler_fpath,
    )

    print(
        "Scaler object for Mordred:        ",
        md_scaler_fpath,
    )

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage
    # ------------------------------------------------------
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }

    for stage, split_file in stages.items():

        print("\n" + "=" * 70)
        print(f"Process stage: {stage}")
        print("=" * 70)

        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        rsp = drp.DrugResponseLoader(
            params,
            split_file=split_file,
            verbose=False,
        ).dfs["response.tsv"]

        # --------------------------------
        # Retain responses for which both
        # omics and drug features exist.
        # --------------------------------
        rsp = rsp.merge(
            ge[[params["canc_col_name"]]],
            on=params["canc_col_name"],
            how="inner",
        )

        rsp = rsp.merge(
            md[[params["drug_col_name"]]],
            on=params["drug_col_name"],
            how="inner",
        )

        ge_sub = ge[
            ge[params["canc_col_name"]].isin(
                rsp[params["canc_col_name"]]
            )
        ].reset_index(drop=True)

        md_sub = md[
            md[params["drug_col_name"]].isin(
                rsp[params["drug_col_name"]]
            )
        ].reset_index(drop=True)

        # --------------------------------
        # Scale using the selected scalers.
        #
        # These may either be:
        #   1. newly fit scalers, or
        #   2. pretrained scalers.
        #
        # scale_df must NOT fit again when
        # scaler= is supplied.
        # --------------------------------
        ge_sc, _ = scale_df(
            ge_sub,
            scaler=ge_scaler,
        )

        md_sc, _ = scale_df(
            md_sub,
            scaler=md_scaler,
        )

        # --------------------------------
        # [Req] Build data filename
        # --------------------------------
        data_fname = frm.build_ml_data_file_name(
            data_format=params["data_format"],
            stage=stage,
        )

        # --------------------------------
        # Merge response + scaled features
        # --------------------------------
        print("Merge data")

        data = rsp.merge(
            ge_sc,
            on=params["canc_col_name"],
            how="inner",
        )

        data = data.merge(
            md_sc,
            on=params["drug_col_name"],
            how="inner",
        )

        # Shuffle
        data = data.sample(
            frac=1.0
        ).reset_index(drop=True)

        # --------------------------------
        # Save model input data
        # --------------------------------
        print("Save data")

        # to_parquet() throws error because
        # "study" can contain mixed values.
        data = data.drop(columns=["study"])

        output_path = (
            Path(params["output_dir"])
            / data_fname
        )

        data.to_parquet(output_path)

        print(
            f"{stage} data shape: {data.shape}"
        )

        print(
            f"{stage} data saved: {output_path}"
        )

        # --------------------------------
        # Prepare Y dataframe
        # --------------------------------
        fea_list = [
            "ge",
            "mordred",
        ]

        fea_cols = [
            col
            for col in data.columns
            if col.split(fea_sep)[0] in fea_list
        ]

        meta_cols = [
            col
            for col in data.columns
            if col.split(fea_sep)[0] not in fea_list
        ]

        ydf = data[meta_cols]

        # --------------------------------
        # [Req] Save Y dataframe
        # --------------------------------
        frm.save_stage_ydf(
            ydf,
            stage,
            params["output_dir"],
        )

        del rsp
        del ge_sub
        del md_sub
        del ge_sc
        del md_sc
        del data
        del ydf

    return params["output_dir"]


# [Req]
def main(args):

    additional_definitions = preprocess_params

    cfg = DRPPreprocessConfig()

    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="lgbm_params.txt",
        additional_definitions=additional_definitions,
    )

    ml_data_outdir = run(params)

    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])