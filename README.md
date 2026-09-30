# PDO_DRP_Analysis
This repository includes code to use data of patient-derived organoids and immortalized cell lines to develop and evaluate anti-cancer drug response prediction models.

The project evaluates several drug response prediction models across multiple experimental settings, including joint training and transfer learning.

## Repository Structure

```text
PDO_DRP_Analysis/
├── analyze_results/       # Summarize experiment results
├── dataset_analysis/      # Dataset statistics and exploratory analyses
├── models/                # Model preprocessing, training, and inference code
├── run_experiments/       # Experiment orchestration scripts
└── pdo_drp_data/          # Input data and experimental splits
```

See the README within each directory for additional details.

## Data

The data required to run the analyses are available from Zenodo:

https://doi.org/10.5281/zenodo.22920452

Download the data and place it in:

```text
PDO_DRP_Analysis/pdo_drp_data/
```

The analysis and experiment scripts expect this directory structure when run from the repository root.

## Models

The model implementations used in this study are based on the **IMPROVE** versions of GraphDRP, HiDRA, and LGBM, with modifications made for the experiments in this repository.

Instructions for setting up the model environments, dependencies, and running the underlying models can be found in the corresponding IMPROVE repositories:

- GraphDRP: https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop
- HiDRA: https://github.com/JDACS4C-IMPROVE/HiDRA/tree/branch-v0.1.0
- LGBM: https://github.com/JDACS4C-IMPROVE/LGBM/tree/develop
- IMPROVE documentation: https://jdacs4c-improve.github.io/docs/

See `models/README.md` for information about the model-specific modifications used in this study.

## Running the Analysis

Experiment runners are located under:

```text
run_experiments/
```

These scripts coordinate preprocessing, training, inference, and evaluation using the model implementations under `models/`.

Results can then be summarized using the scripts under:

```text
analyze_results/
```

Refer to the READMEs in these directories for experiment-specific instructions and configuration.
