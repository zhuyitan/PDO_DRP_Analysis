import sys
from pathlib import Path
from typing import Dict

# [MODEL] Model-specific imports, as needed
import subprocess
import joblib
import pandas as pd
import numpy as np
# import pickle
import os
from Step2_DataEncoding import DataEncoding
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# [Req] Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
# [Req] Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent  # [Req]

def process_gene_expression(data, gene_expression_columns, dtype=None):
    df_gene_expression = data[gene_expression_columns]
    if dtype is not None:
        if dtype not in ["float32", "float16"]:
            raise ValueError("dtype must be 'float32' or 'float16'")
        df_gene_expression = df_gene_expression.astype(dtype)
    return df_gene_expression

def preprocess(args, rna_data, drug_data, response_data, response_metric='AUC'):
    obj = DataEncoding(args, args["input_supp_data_dir"], args["canc_col_name"],
                       args["sample_col_name"], args["y_col_name"], args["drug_col_name"])
    drug_smiles = drug_data

    drugid2smile = dict(
        zip(drug_smiles[args["drug_col_name"]], drug_smiles['SMILES']))
    smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
        obj._drug2emb_encoder)
    uniq_smile_dict = dict(
        zip(drug_smiles['SMILES'].unique(), smile_encode))

    drug_data['smiles'] = [drugid2smile[i]
                           for i in drug_data[args["drug_col_name"]]]
    drug_data['drug_encoding'] = [uniq_smile_dict[i]
                                  for i in drug_data['smiles']]
    drug_data = drug_data.reset_index()

    response_data = response_data[[
        args["canc_col_name"], args["drug_col_name"], response_metric]]
    response_data.columns = [args["canc_col_name"],
                             args["drug_col_name"], 'Label']
    drug_data = pd.merge(response_data, drug_data,
                         on=args["drug_col_name"], how='inner')

    drug_data.index = range(drug_data.shape[0])
    rna_data.index = range(rna_data.shape[0])

    print('Preprocessing...!!!')
    print(np.shape(rna_data), np.shape(drug_data))
    return rna_data, drug_data


def build_common_data(params: Dict):
    omics_loader = omics_utils.OmicsLoader(params)
    drugs_loader = drugs_utils.DrugsLoader(params)

    gene_expression = omics_loader.dfs['cancer_gene_expression.tsv']
    df_drug = drugs_loader.dfs['drug_SMILES.tsv']
    df_drug = df_drug.reset_index()
    df_drug = df_drug[[params["drug_col_name"], "SMILES"]].copy()
    df_drug.columns = [params["drug_col_name"], "smiles"]
    params['drug_id'] = params["drug_col_name"]
    df_drug["SMILES"] = df_drug["smiles"]

    def gene_selection(df, genes_fpath, canc_col_name):
        with open(genes_fpath) as f:
            genes = [str(line.rstrip()) for line in f]
        genes = sorted(list(set(genes).intersection(set(df.columns[1:]))))
        cols = [canc_col_name] + genes
        return df[cols]

    if params["use_lincs"]:
        genes_fpath = filepath/"landmark_genes"
        gene_expression = gene_selection(
            gene_expression, genes_fpath, canc_col_name=params["canc_col_name"])

    return df_drug, gene_expression


def _download_default_dataset(default_data_url):
    url = default_data_url
    improve_data_dir = '.'
    if improve_data_dir is None:
        improve_data_dir = '.'

    OUT_DIR = improve_data_dir
    print('outdir after: {}'.format(OUT_DIR))

    url_length = len(url.split('/'))-5
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    url = url.strip('\'')
    try:
        subprocess.run(['rm', '*index*'])
    except:
        pass
    command = ['wget', '--recursive', '--no-clobber', '-nH',
               f'--cut-dirs={url_length}', '--no-parent', f'--directory-prefix={OUT_DIR}', f'{url}']
    subprocess.run(command)
    try:
        subprocess.run(['rm', '*index*'])
    except:
        pass


def download_model_data(params):
    _download_default_dataset(params["default_data_url"])


def download_dataset(params):
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"])
    command = f'wget --directory-prefix={mainpath} --cut-dirs=7 -nH -np -m ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data'
    tokens = command.split(' ')
    subprocess.run(tokens)


def prepare_dataframe(args, gene_expression, smiles, responses):
    gene_expression, drug_data = preprocess(args,
                                            gene_expression, smiles, responses, args["y_col_name"])
    drug_data = drug_data.drop(['index'], axis=1)
    drug_columns = [
        x for x in drug_data.columns if x not in [args["canc_col_name"], args["drug_col_name"]]]
    data = pd.merge(gene_expression, drug_data,
                    on=args["canc_col_name"], how='inner')
    gene_expression = gene_expression.drop([args["canc_col_name"]], axis=1)
    gene_expression_columns = gene_expression.columns

    return data, gene_expression_columns, drug_columns


def get_common_samples(df1, df2, ref_col):
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def scale_df(dataf, scaler_name="std", scaler=None, verbose=False):
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return dataf, None

    df_num = dataf.select_dtypes(include="number")

    if scaler is None:
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(f"The specified scaler {scaler_name} is not implemented (no df scaling).")
            return dataf, None
        df_norm = scaler.fit_transform(df_num)
    else:
        df_norm = scaler.transform(df_num)

    dataf[df_num.columns] = df_norm
    return dataf, scaler


def build_stage_dependent_data(params, stage, df_drug, df_cell_all, scaler):
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for key in params:
        if type(params[key]) == str:
            params[key] = params[key].strip('"')

    df_response = drp.DrugResponseLoader(params,
                                         split_file=stages[stage],
                                         verbose=False).dfs["response.tsv"]

    df_y, df_cell = get_common_samples(df1=df_response,
                                       df2=df_cell_all,
                                       ref_col=params["canc_col_name"])
    print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

    if stage == "train":
        df_cell, scaler = scale_df(df_cell, scaler_name=params["scaling"])
        if params["scaling"] is not None and params["scaling"] != "none":
            scaler_fname = os.path.join(params["output_dir"], "cell_xdata_scaler.gz")
            joblib.dump(scaler, scaler_fname)
            print("Scaling object created is stored in: ", scaler_fname)
    else:
        df_cell, _ = scale_df(df_cell, scaler=scaler)

    df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]

    data, gene_expression_columns, drug_columns = prepare_dataframe(params, df_cell, df_drug, df_y)

    df_gene_expression = process_gene_expression(data, gene_expression_columns, params["gene_dtype"])

    # FIX: include drug_col_name (improve_chem_id) in saved drug dataframe
    # so entropy loss can group predictions per drug at train time
    df_drug_out = data[drug_columns + [params["drug_col_name"]]]

    out_path = os.path.join(params["output_dir"], frm.build_ml_data_file_name(
        params["data_format"], stage=stage))
    print(out_path)
    df_output = {'drug': df_drug_out, 'gene_expression': df_gene_expression}
    for key in df_output:
        df_output[key].to_hdf(out_path, key)

    data[params['y_col_name']] = data['Label']
    y_df = pd.DataFrame(
        data[['Label', params['y_col_name'], params['canc_col_name'], params['drug_col_name']]])
    frm.save_stage_ydf(y_df, stage, params['output_dir'])

    return scaler


def run(params: Dict):
    df_drug, df_cell_all = build_common_data(params)
    stages = ["train", "val", "test"]
    scaler = None
    for st in stages:
        print(f"Building stage: {st}")
        scaler = build_stage_dependent_data(params, st, df_drug, df_cell_all, scaler)
    return params["output_dir"]


def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        filepath,
        default_config="deepttc_params.txt",
        additional_definitions=preprocess_params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])