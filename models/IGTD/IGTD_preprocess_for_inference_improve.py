"""
IGTD_preprocess_for_inference_improve.py
----------------------------------------
Preprocess an inference dataset for IGTD using the settings saved during
training preprocessing (IGTD_preprocess_improve.py).

IGTD converts each sample's features into an image, and a trained model
only works on images built exactly the same way as its training data. This
script loads the saved preprocessor parameters (preprocessor_param_file)
and reuses them instead of fitting new ones:
    - the same selected features
    - the same min-max scaling values
    - the same feature-to-pixel image layout

Inputs (from IGTD_params.txt or the command line):
    inference_split_file     sample/drug pairs to preprocess
    preprocessor_param_file  saved preprocessing settings, in output_dir
    inference_data_name      subfolder name for this dataset's output

Output is saved under output_dir/inference_data_name with test-stage file
names, so IGTD_infer_improve.py can read it directly.
"""

import os
from pathlib import Path
import _pickle as cp

import numpy as np
import pandas as pd

from Table2Image_Functions import min_max_transform, load_data, generate_image_data

from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.utils as frm
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp
from model_params_def import preprocess_params


filepath = Path(__file__).resolve().parent


def run(params):

    # [Req] Build paths and create output dir
    processed_outdir = params["output_dir"]

    print("\nLoading response data...")
    rr_inference = drp.DrugResponseLoader(params, split_file=params["inference_split_file"], verbose=True)
    df_response = rr_inference.dfs["response.tsv"]

    print("\nLoading omics data...")
    oo = omics_utils.OmicsLoader(params)
    print(oo)
    ge = oo.dfs['cancer_gene_expression.tsv']  # get the needed canc x data
    # Use the first column (sample IDs) as the index
    ge.index = ge.iloc[:, 0]
    ge = ge.iloc[:, 1:]
    ge = ge.loc[np.unique(df_response.improve_sample_id), :]

    print("\nLoading drugs data...")
    dd = drugs_utils.DrugsLoader(params)
    print(dd)
    md = dd.dfs['drug_mordred.tsv']  # get the needed drug x data
    md = md.loc[np.unique(df_response.improve_chem_id), :]

    print("\nLoading data preprocessor parameters...")
    pkl_file = open(os.path.join(params["output_dir"], params["preprocessor_param_file"]), 'rb')
    preprocessor_params = cp.load(pkl_file)
    pkl_file.close()

    print("\nGenerate gene expression image data...")
    ge = ge.loc[:, preprocessor_params['gene_expression']['input_data'].columns].iloc[:, preprocessor_params['gene_expression']['feature_id']]
    norm_ge, min_v, max_v = min_max_transform(ge.values, preprocessor_params['gene_expression']['min_max_param'][0], 
                                              preprocessor_params['gene_expression']['min_max_param'][1])
    norm_ge = pd.DataFrame(norm_ge, columns=ge.columns, index=ge.index)
    result_dir = os.path.join(processed_outdir, params['inference_data_name'], 'Image_Data', 'Cancer')
    os.makedirs(name=result_dir, exist_ok=True)

    data, samples = generate_image_data(data=norm_ge, index=preprocessor_params['gene_expression']['feature_swap_index'], 
                                        num_row=preprocessor_params['gene_expression']['num_row'], 
                                        num_column=preprocessor_params['gene_expression']['num_col'],
                                        coord=preprocessor_params['gene_expression']['image_coordinate'], 
                                        image_folder=os.path.join(result_dir, 'data'), file_name='')

    output = open(result_dir + '/inference_data_results.pkl', 'wb')
    cp.dump(norm_ge, output)
    cp.dump(data, output)
    cp.dump(samples, output)
    output.close()

    print("\nGenerate drug descriptor image data...")
    md = md.loc[:, preprocessor_params['drug_descriptor']['input_data'].columns].iloc[:, preprocessor_params['drug_descriptor']['feature_id']]
    norm_md, min_v, max_v = min_max_transform(md.values, preprocessor_params['drug_descriptor']['min_max_param'][0], 
                                              preprocessor_params['drug_descriptor']['min_max_param'][1])
    norm_md = pd.DataFrame(norm_md, columns=md.columns, index=md.index)
    result_dir = os.path.join(processed_outdir, params['inference_data_name'], 'Image_Data', 'Drug')
    os.makedirs(name=result_dir, exist_ok=True)

    data, samples = generate_image_data(data=norm_md, index=preprocessor_params['drug_descriptor']['feature_swap_index'], 
                                        num_row=preprocessor_params['drug_descriptor']['num_row'], 
                                        num_column=preprocessor_params['drug_descriptor']['num_col'],
                                        coord=preprocessor_params['drug_descriptor']['image_coordinate'], 
                                        image_folder=os.path.join(result_dir, 'data'), file_name='')

    output = open(result_dir + '/inference_data_results.pkl', 'wb')
    cp.dump(norm_md, output)
    cp.dump(data, output)
    cp.dump(samples, output)
    output.close()

    cancer_image_data_filepath = os.path.join(processed_outdir, params['inference_data_name'], 'Image_Data', 'Cancer', 'inference_data_results.pkl')
    drug_image_data_filepath = os.path.join(processed_outdir, params['inference_data_name'], 'Image_Data', 'Drug', 'inference_data_results.pkl')
    data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    data = load_data(rr_inference.dfs["response.tsv"], cancer_image_data_filepath, drug_image_data_filepath, 
                     params['canc_col_name'], params['drug_col_name'], params['y_col_name'])
    output = open(os.path.join(processed_outdir, params['inference_data_name'], data_fname), 'wb')
    cp.dump(data, output, protocol=4)
    output.close()

    # [Req] Save y dataframe for the current stage
    res_to_save = pd.DataFrame({params['canc_col_name']: [i.split('|')[0] for i in data['sample']],
                                params['drug_col_name']: [i.split('|')[1] for i in data['sample']],
                                params['y_col_name']: data['label']}, index=None)
    frm.save_stage_ydf(ydf=res_to_save, stage="test", output_dir=os.path.join(processed_outdir, params['inference_data_name']))

    return processed_outdir


def main():
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="IGTD_params.txt",
        additional_definitions=preprocess_params)
    processed_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main()
