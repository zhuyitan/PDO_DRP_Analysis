# Models

This directory contains the model implementations used in the PDO drug-response prediction experiments:

```text id="t6rz3p"
models/
├── GraphDRP/
├── HiDRA/
└── LGBM/
```

Each model directory includes the **preprocessing, training, and inference scripts used by the experiment runners** under `run_experiments/`.

These implementations are based on models from the **IMPROVE (Innovative Methodologies and New Data for Predictive Oncology Model Evaluation)** project, with modifications made for the analyses in this repository.

## Upstream Models

| Model | IMPROVE Repository |
|---|---|
| GraphDRP | https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/branch-v0.1.0 |
| HiDRA | https://github.com/JDACS4C-IMPROVE/HiDRA/tree/branch-v0.1.0 |
| LGBM | https://github.com/JDACS4C-IMPROVE/LGBM/tree/branch-v0.1.0 |

Additional IMPROVE resources:

- Documentation: https://jdacs4c-improve.github.io/docs/
- GitHub organization: https://github.com/JDACS4C-IMPROVE

Users should refer to the upstream IMPROVE repositories and documentation for model-specific installation, dependencies, configuration, and standard usage instructions.

## Modifications in This Repository

The general model structure and supporting files are derived from the corresponding IMPROVE implementations. Changes were made where necessary to support the experiments in this repository.

### GraphDRP

Based on the IMPROVE GraphDRP implementation. The local preprocessing, training, and inference scripts include modifications to support the transfer-learning workflow used in this analysis.

### HiDRA

Based on the IMPROVE HiDRA implementation. The local version includes architectural modifications intended to improve computational efficiency for the experiments performed in this analysis.

### LGBM

Based on the IMPROVE LGBM implementation. The local preprocessing, training, and inference scripts include modifications to support transfer learning from pretrained cancer cell line (CCL) models to PDO data.

## Experiment Runners

The scripts in `models/` provide the model-specific preprocessing, training, and inference functionality. They are called by the experiment orchestration scripts under:

```text id="5p5e0e"
run_experiments/
├── cell_line_only_training/
├── joint_training/
└── transfer_learning/
```

For reproducing the experiments in this repository, refer to the READMEs in the corresponding `run_experiments/` directories.

For general information about running and configuring the underlying models, refer to the original IMPROVE repositories and official IMPROVE documentation.