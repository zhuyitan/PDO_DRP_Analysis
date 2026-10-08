""" Preprocessing of raw data to generate datasets for UNO Model. """
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Union
from uno_utils_improve import print_duration, get_common_samples, get_column_ranges, subset_data, lincs_gene_filtering
# Script Dependencies: pandas, numpy, joblib, scikit-learn

# [Req] Import params
from params import app_preproc_params, model_preproc_params, app_train_params, model_train_params

import numpy as np
import pandas as pd
import joblib
import textwrap

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    PowerTransformer,
)

filepath = Path(__file__).resolve().parent  # [Req]

# IMPROVE imports
import improvelib.utils as frm
import improvelib.applications.drug_response_prediction.drp_utils as drp
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils

# ---------------------
# [Req] Parameter lists
# ---------------------
preprocess_params = app_preproc_params + model_preproc_params
train_params = app_train_params + model_train_params

def scale_df(
    df: pd.DataFrame, scaler_name: str = "std", scaler=None, verbose: bool = False
):
    """Returns a dataframe with scaled data."""
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return df, None

    # Scale data
    df_num = df.select_dtypes(include="number")

    if scaler is None:  # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "maxabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        elif scaler_name in ["l1", "l2", "max"]:
            scaler = Normalizer(norm=scaler_name)
        elif scaler_name == "power_yj":
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            print(
                f"The specified scaler ({scaler_name}) is not implemented (no df scaling)."
            )
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else:  # Apply passed scikit scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm

    # Remove rows with NaN or inf values and print proportion of rows removed
    rows_before = df.shape[0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    proportion_removed = (rows_before - rows_after) / rows_before
    print(f"Proportion of rows removed for corrupted data: {proportion_removed:.3%}")

    return df, scaler

def run(params: Dict, pretrained_model_dir: str = None):
    """Execute data pre-processing for UNO model.

    Args:
        params: IMPROVE params dict (as before).
        pretrained_model_dir: Optional. If given, the gene-expression and
            drug (Mordred) scalers are LOADED from this directory
            (`<pretrained_model_dir>/<ge_scaler_fname>` and
            `.../<md_scaler_fname>`) and only *applied* (`.transform`) to
            this study's data — never refit. This guarantees that
            finetuning/inference preprocessing matches exactly what the
            pretrained model was itself trained on. If omitted (None),
            behavior is unchanged from before: new scalers are fit on
            this run's train+val data, as appropriate for pretraining
            from scratch.
    """
    # Record start time
    preprocess_start_time = time.time()

    # [Req] Build paths and create ML data dir
    params = frm.build_paths(params)
    frm.create_outdir(outdir=params["output_dir"])

    # Reading hyperparameters
    preprocess_debug = params["preprocess_debug"]
    preprocess_subset_data = params["preprocess_subset_data"]

    temp_start_time = time.time()
    # [Req] Load omics data
    print("\nLoading omics data.")
    omics_obj = omics_utils.OmicsLoader(params)
    ge = omics_obj.dfs['cancer_gene_expression.tsv']
    ge["improve_sample_id"] = ge['improve_sample_id'].astype(str)
    first_column = ge.iloc[:, :1]
    rest_columns = ge.iloc[:, 1:].add_prefix('ge.')
    ge = pd.concat([first_column, rest_columns], axis=1)

    # Apply LINCS gene filtering if enabled
    use_lincs = params["use_lincs"]
    lincs_genes_file = params["lincs_genes_file"]

    print(f"DEBUG: use_lincs = {use_lincs}, type = {type(use_lincs)}")
    print(f"DEBUG: lincs_genes_file = {lincs_genes_file}")

    if use_lincs and not lincs_genes_file:
        raise ValueError("lincs_genes_file must be specified when use_lincs is True") 

    if use_lincs and lincs_genes_file:
        print("\nApplying LINCS gene filtering...")
        
        # Build the full path to the LINCS genes file
        # The file is in the same directory as the preprocessing script
        lincs_file_path = filepath / lincs_genes_file
        
        print(f"Looking for LINCS genes file at: {lincs_file_path}")
        
        if not lincs_file_path.exists():
            raise FileNotFoundError(f"LINCS genes file not found at: {lincs_file_path}")
        
        print(f"Gene expression shape before LINCS filtering: {ge.shape}")
        ge = lincs_gene_filtering(ge, str(lincs_file_path), params['canc_col_name'])
        print(f"Gene expression shape after LINCS filtering: {ge.shape}")
        print("LINCS gene filtering completed.")
    else:
        print("\nLINCS gene filtering disabled.")

    # [Req] Load drug data
    print("\nLoading drugs data.")
    drugs_obj = drugs_utils.DrugsLoader(params)
    md = drugs_obj.dfs['drug_mordred.tsv']
    md = md.reset_index()
    md["improve_chem_id"] = md['improve_chem_id'].astype(str)

    temp_end_time = time.time()
    print("")
    print_duration("Loading Data", temp_start_time, temp_end_time)

    if preprocess_debug:
        print("Loaded Gene Expression:")
        print(ge.head())
        print(ge.shape)
        print("")
        print("Loaded Mordred Descriptors:")
        print(md.head())
        print(md.shape)
        print("")

    temp_start_time = time.time()
    # Data prep to create scaler on
    rsp_tr = drp.DrugResponseLoader(
        params, split_file=params["train_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(
        params, split_file=params["val_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)
    rsp = rsp[[params["canc_col_name"], params["drug_col_name"], params["y_col_name"]]]

    ge_sub, md_sub, rsp_sub = get_common_samples(
        ge, md, rsp, params["canc_col_name"], params["drug_col_name"]
    )

    if preprocess_debug:
        print(textwrap.dedent(f"""
            Gene Expression Shape Before Subsetting With Response: {ge.shape}
            Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
            Mordred Shape Before Subsetting With Response: {md.shape}
            Mordred Shape After Subsetting With Response: {md_sub.shape}
            Response Shape Before Merging With Data: {rsp.shape}
            Response Shape After Merging With Data: {rsp_sub.shape}
        """))

    # Create / Reuse Feature Scalers
    #
    # If pretrained_model_dir is given, the scalers are loaded from that
    # directory and only APPLIED here (never refit on this study's
    # train+val data). This matters whenever this preprocessing run feeds
    # a pretrained model (finetuning HPO trials, final finetune runs, or
    # inference): the model's input features must be scaled the exact
    # same way they were during pretraining, or the pretrained weights are
    # being fed distribution-shifted inputs.
    if pretrained_model_dir:
        ge_scaler_src = Path(pretrained_model_dir) / params["ge_scaler_fname"]
        md_scaler_src = Path(pretrained_model_dir) / params["md_scaler_fname"]

        if not ge_scaler_src.exists() or not md_scaler_src.exists():
            raise FileNotFoundError(
                "pretrained_model_dir={!r} was given but scaler file(s) are "
                "missing: {} / {}. Preprocessing must reuse the pretrained "
                "model's scalers rather than fit new ones — refusing to "
                "silently fall back to fitting fresh scalers.".format(
                    pretrained_model_dir, ge_scaler_src, md_scaler_src
                )
            )

        print(
            "\nReusing pretrained scalers from {} (transform only, NOT "
            "refit on this study's data):\n  ge: {}\n  md: {}".format(
                pretrained_model_dir, ge_scaler_src, md_scaler_src
            )
        )
        ge_scaler = joblib.load(ge_scaler_src)
        md_scaler = joblib.load(md_scaler_src)

        _, ge_scaler = scale_df(ge_sub, scaler=ge_scaler)
        _, md_scaler = scale_df(md_sub, scaler=md_scaler)
    else:
        print("\nCreating Feature Scalers (no pretrained_model_dir given)\n")
        _, ge_scaler = scale_df(ge_sub, scaler_name=params["ge_scaling"])
        _, md_scaler = scale_df(md_sub, scaler_name=params["md_scaling"])

    ge_scaler_fpath = Path(params["output_dir"]) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    md_scaler_fpath = Path(params["output_dir"]) / params["md_scaler_fname"]
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub, md_sub
    temp_end_time = time.time()
    print_duration("Creating/Loading Scalers", temp_start_time, temp_end_time)

    # [Req] Construct ML data for every stage (train, val, test)
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }

    for stage, split_file in stages.items():
        split_start_time = time.time()
        print(f"Stage: {stage.upper()}")
        rsp = drp.DrugResponseLoader(params, split_file=split_file, verbose=False).dfs[
            "response.tsv"
        ]
        rsp = rsp[[params["canc_col_name"], params["drug_col_name"], params["y_col_name"]]]

        ge_sub, md_sub, rsp_sub = get_common_samples(
            ge, md, rsp, params["canc_col_name"], params["drug_col_name"]
        )

        if preprocess_debug:
            print(textwrap.dedent(f"""
                Gene Expression Shape Before Subsetting With Response: {ge.shape}
                Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
                Mordred Shape Before Subsetting With Response: {md.shape}
                Mordred Shape After Subsetting With Response: {md_sub.shape}
                Response Shape Before Merging With Data: {rsp.shape}
                Response Shape After Merging With Data: {rsp_sub.shape}
            """))

        temp_start_time = time.time()
        print("\nScaling data")
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)
        temp_end_time = time.time()
        print_duration(f"Applying Scaler to {stage.capitalize()}", temp_start_time, temp_end_time)

        if preprocess_debug:
            print("Gene Expression Scaled:")
            print(ge_sc.head())
            print(ge_sc.shape)
            print("")
            print("Mordred Descriptors Scaled:")
            print(md_sc.head())
            print(md_sc.shape)
            print("")

        if preprocess_subset_data:
            total_num_samples = 5000
            stage_proportions = {"train": 0.8, "val": 0.1, "test": 0.1}
            rsp_sub = subset_data(rsp_sub, stage, total_num_samples, stage_proportions)

        temp_start_time = time.time()
        print(f"Saving {stage.capitalize()} Data (unmerged) to Parquet")
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)
        ge_fname = f"ge_{data_fname}"
        md_fname = f"md_{data_fname}"
        rsp_fname = f"rsp_{data_fname}"
        ge_sc.to_parquet(Path(params["output_dir"]) / ge_fname)
        md_sc.to_parquet(Path(params["output_dir"]) / md_fname)
        rsp_sub.to_parquet(Path(params["output_dir"]) / rsp_fname)
        ydf = rsp_sub[['improve_sample_id', 'improve_chem_id', params["y_col_name"]]]
        frm.save_stage_ydf(ydf, stage, params["output_dir"])
        temp_end_time = time.time()
        print_duration(f"Saving {stage.capitalize()} Dataframes", temp_start_time, temp_end_time)

        split_end_time = time.time()
        print_duration(f"Processing {stage.capitalize()} Data", split_start_time, split_end_time)

    preprocess_end_time = time.time()
    print_duration(
        f"Preprocessing Data (All)", preprocess_start_time, preprocess_end_time
    )

    return params["output_dir"]

# [Req]
def main(args):
    # Pull out an optional --pretrained_model_dir flag before handing
    # control to the IMPROVE config system, since it is not a registered
    # IMPROVE param (it only exists to make this script reuse a
    # pretrained model's scalers — see run()'s docstring).
    #
    # IMPORTANT: cfg.initialize_parameters() below does NOT take an argv
    # list — it parses directly from sys.argv internally. Stripping the
    # flag out of a local `args` copy (as an earlier version of this
    # script did) has no effect on that parse and the IMPROVE argparse
    # still sees (and rejects) --pretrained_model_dir as unrecognized.
    # We therefore mutate sys.argv itself here, then restore it after so
    # the rest of the process still sees the original argv.
    args = list(args)
    pretrained_model_dir = None
    if "--pretrained_model_dir" in args:
        idx = args.index("--pretrained_model_dir")
        try:
            pretrained_model_dir = args[idx + 1]
        except IndexError:
            raise ValueError("--pretrained_model_dir was given with no value")
        del args[idx:idx + 2]

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + args

    # [Req]
    additional_definitions = preprocess_params
    cfg = DRPPreprocessConfig()
    try:
        params = cfg.initialize_parameters(
            pathToModelDir=filepath,
            default_config="uno_default_model.txt",
            additional_definitions=additional_definitions,
            required=None,
        )
    finally:
        sys.argv = original_argv

    ml_data_outdir = run(params, pretrained_model_dir=pretrained_model_dir)
    print(
        "\nFinished UNO pre-processing (transformed raw DRP data to model input ML data)."
    )

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])