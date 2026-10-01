# Cell Line Only Training Experiments

This directory contains experiment runners for training drug response prediction models on cancer cell line (CCL) data only, and then evaluating them on held-out CCL data and on PDO studies.

No PDO data (expression, drug, or response) is used at any point during training. The PDO evaluations are therefore **zero-shot**: they measure how well a model trained purely on cell lines transfers to patient-derived organoids without any adaptation.

This scheme is the baseline against which the joint-training experiments (`run_experiments/joint_training/`) can be compared.

## Scripts

```text
cell_line_only_training/
├── README.md
├── run_deepttc.py
├── run_graphdrp.py
├── run_hidra.py
├── run_igtd.py
├── run_lgbm.py
└── run_uno.py
```

Run each script from the repository root, for example:

```bash
python run_experiments/cell_line_only_training/run_graphdrp.py
```

These are orchestration scripts. The model-specific preprocessing, training, and inference scripts must also be available in the expected repository locations (for example, under `models/<model_name>/`).

---

# Experiment Design

The cell-line-only experiments use a single index:

- a **CCL split**, indexed by `i` (`i = 0 … 9`)

Unlike joint training, there is **no PDO split index `j`**. Because no PDO data enters training, there is nothing to partition on the PDO side.

For each CCL split `i`, one model is trained:

| Role | Split file |
|---|---|
| Train | `ccl_split_{i}_train_ids.txt` |
| Validation | `ccl_split_{i}_val_ids.txt` |

With 10 CCL splits, this produces:

```text
10 splits = 10 trained model instances per architecture, per split type
```

The CCL training data is pooled from the cell line drug screening studies (CCLE, CTRPv2, GDSCv1, GDSCv2, and gCSI). The prediction target is the area under the dose-response curve (AUC).

## Test sets

Each trained model is evaluated, without retraining, on the held-out CCL split and on every PDO study.

The studies currently used are:

```text
IMPROVE dataset 1
IMPROVE dataset 2
IMPROVE dataset 3
Pilot_1
```

| Test set | Split file | Evaluated |
|---|---|---|
| CCL | `ccl_split_{i}_test_ids.txt` | once per split `i` |
| Study PDO | `{study}_pdo_ids.txt` | once per split `i` and study |
| Study CCL | `{study}_ccl_ids.txt` | once per split `i` and study |

`Pilot_1` does not have a study-CCL file, so the study-CCL evaluation is skipped for it.

Note that the study PDO test set is the **full** PDO sample set for each study (`{study}_pdo_ids.txt`), not a held-out fold as in joint training (`{study}_split_{j}_test_ids.txt`).

---

# Workflow

For each CCL split `i`, a runner performs:

```text
CCL preprocessing (train / val)
        │
        ▼
Model training
        │
        ▼
CCL test inference
        │
        ▼
For each study:
    Study PDO inference
        │
        ▼
    Study CCL inference   (skipped for Pilot_1)
```

The model is trained **once** per split and reused for every inference set.

## Total runs

For each architecture:

```text
10 training runs
10 × 1                 CCL test inferences
10 × 4 studies         study PDO inferences
10 × 3 studies         study CCL inferences (Pilot_1 excluded)
```

---

# Configuration

Run each script from the repository root. The main settings are defined near the top of each script. For example, in `run_graphdrp.py`:

```python
STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1"
]

SPLIT_TYPES = [
    "random",
    "cancer-blind"
]

SPLITS_ROOT = "pdo_drp_data/splits"
RAW_DATA_DIR = "pdo_drp_data"
BASE_OUTPUT_DIR = "cell_line_only_results_graphdrp"

NUM_SPLITS = 10

SEPERATE_STUDIES = {
    "Pilot_1"
}
```

| Script | Notes |
|---|---|
| `run_deepttc.py` | Runs commands from `models/DeepTTC` |
| `run_graphdrp.py` | Deletes temporary `.pt` tensors after each inference |
| `run_hidra.py` | Runs commands from `models/HiDRA` |
| `run_igtd.py` | Uses `IGTD_preprocess_for_inference_improve.py` with a patched copy of `IGTD_params.txt` for each test set |
| `run_lgbm.py` | Run from the LGBM Python environment |
| `run_uno.py` | Sets `TF_DETERMINISTIC_OPS=1` |

## Split types

Each runner runs every split type in `SPLIT_TYPES`, reading the CCL split files from `pdo_drp_data/splits/{split_type}/`. With both split types and 10 CCL splits, each runner trains:

```text
2 split types × 10 splits = 20 models
```

To run only one split type, remove the other from `SPLIT_TYPES`.

## GPU scheduling

No GPU settings need to be edited. Each runner detects the available GPUs when it starts and runs one split pipeline per GPU at a time. Each command only sees its own GPU.

To limit which GPUs are used, set `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_experiments/cell_line_only_training/run_graphdrp.py
```

To run more than one split per GPU at a time, set `JOBS_PER_GPU`:

```bash
JOBS_PER_GPU=2 python run_experiments/cell_line_only_training/run_graphdrp.py
```

If no GPU is found, the splits run one at a time on the CPU.

## Output

Results are written under `cell_line_only_results_<model>/`, with one directory per CCL split:

```text
cell_line_only_results_graphdrp/
├── random/
│   └── split_{i}/
│       ├── processed_ccl/        preprocessed data and trained model
│       ├── ccl/                  CCL test inference
│       └── {study}/
│           ├── processed_pdo/
│           ├── pdo/              study PDO inference
│           ├── processed_ccl/
│           └── ccl/              study CCL inference (not for Pilot_1)
└── cancer-blind/
    └── split_{i}/                same layout as above
```

A failure in one test set is printed and the remaining test sets for that split still run.

## Logs

Each step saves its output next to its results as `preprocess.log`, `train.log` or `infer.log`. If a step fails, the last lines of its log are printed.

When the runner finishes, it prints how many splits completed without errors, and exits with code 1 if any split had an error.

## Restarting

A restarted runner skips every split whose results already exist. A split that only partly finished is rerun from the start, so all of its scores come from the same trained model.

---

# Models

All six runners follow the workflow above. The differences between them are listed here.

## DeepTTC

```bash
python run_experiments/cell_line_only_training/run_deepttc.py
```

- Model scripts: `models/DeepTTC/deepttc_{preprocess,train,infer}_improve.py`
- Commands run from inside `models/DeepTTC`

## GraphDRP

```bash
python run_experiments/cell_line_only_training/run_graphdrp.py
```

- Model scripts: `models/GraphDRP/graphdrp_{preprocess,train,infer}_improve.py`
- The device is passed with `--cuda_name`
- The temporary preprocessing tensors (`processed/train_data.pt`, `val_data.pt`, `test_data.pt`) are deleted after each inference. The model, predictions and scores are kept.

## HiDRA

```bash
python run_experiments/cell_line_only_training/run_hidra.py
```

- Model scripts: `models/HiDRA/HiDRA_{preprocess,train,infer}_improve.py`
- Commands run from inside `models/HiDRA`

## IGTD

```bash
python run_experiments/cell_line_only_training/run_igtd.py
```

- Model scripts: `models/IGTD/IGTD_{preprocess,train,infer}_improve.py` and `IGTD_preprocess_for_inference_improve.py`
- Training must produce `preprocessor_params.pkl`
- Each test set is preprocessed with `IGTD_preprocess_for_inference_improve.py`, using a copy of `IGTD_params.txt` whose `inference_split_file` points at that test set
- Commands run from inside `models/IGTD`

## LightGBM (LGBM)

```bash
python run_experiments/cell_line_only_training/run_lgbm.py
```

- Model scripts: `models/LGBM/lgbm_{preprocess,train,infer}_improve.py`
- Run from the LGBM conda environment. The runner adds `models/IMPROVE` to `PYTHONPATH` and sets `IMPROVE_DATA_DIR`.

## UNO

```bash
python run_experiments/cell_line_only_training/run_uno.py
```

- Model scripts: `models/UNO/uno_{preprocess,train,infer}_improve.py`
- Sets `TF_DETERMINISTIC_OPS=1`
- The trained models in `cell_line_only_results_uno/` are the pretrained models used by `run_experiments/transfer_learning/run_uno.py`

---

# Split Naming Convention

The runners depend on consistent split filenames.

For CCL split `i`:

```text
ccl_split_{i}_train_ids.txt
ccl_split_{i}_val_ids.txt
ccl_split_{i}_test_ids.txt
```

For each study:

```text
{study}_pdo_ids.txt
{study}_ccl_ids.txt
```

Example for `i = 7`:

```text
ccl_split_7_train_ids.txt
ccl_split_7_val_ids.txt
ccl_split_7_test_ids.txt
IMPROVE dataset 1_pdo_ids.txt
IMPROVE dataset 1_ccl_ids.txt
```

Each `.txt` file contains row indices into `y_data/response.tsv`.

---

# Output Interpretation

Each split `i` represents one trained model. The inference outputs evaluate that same model against different test sets.

### CCL

```text
ccl_split_{i}_test_ids.txt
```

Evaluates performance on held-out cancer cell lines, which are the same kind of data the model was trained on.

### Study PDO

```text
{study}_pdo_ids.txt
```

Evaluates zero-shot performance on all PDO samples from the study.

### Study CCL

```text
{study}_ccl_ids.txt
```

Evaluates performance on the cell lines screened within the study, when that file is available.

Performance is reported using the Pearson correlation coefficient (PCC) and the coefficient of determination (R²) between predicted and observed AUC, both per split and aggregated across the 10 splits.

---

# Comparison with Joint Training

| | Cell line only | Joint training |
|---|---|---|
| Training data | CCL only | CCL + part of one PDO study |
| Split indices | `i` | `i` and `j` |
| Models per architecture | 10 per split type | 10 × 10 per study |
| PDO test set | full study (`{study}_pdo_ids.txt`) | held-out fold (`{study}_split_{j}_test_ids.txt`) |

---

# Recommended Environment

A typical environment should include:

- Python 3
- the dependencies required by the corresponding model, including `improvelib`
- CUDA-compatible PyTorch or TensorFlow, depending on the model
- sufficient CPU RAM for concurrent preprocessing
- sufficient disk space for intermediate datasets and model outputs

Activate the model's conda environment before running its runner. Running from the wrong environment usually fails with `No module named 'improvelib'`.

---

# Running in tmux

`run_experiments/launch_tmux.sh` starts a runner in its own detached tmux session, so it keeps running after you disconnect. It activates the given conda environment inside the session and saves all output to `logs/`.

```bash
run_experiments/launch_tmux.sh <conda_env> <runner.py> [work_dir]
```

Example:

```bash
run_experiments/launch_tmux.sh <graphdrp_env> run_experiments/cell_line_only_training/run_graphdrp.py
```

- The session is named after the folder and model, for example `cell_line_only_training-graphdrp`
- `work_dir` is the folder holding `pdo_drp_data/` and `models/` (default: the repository root)
- `CUDA_VISIBLE_DEVICES` and `JOBS_PER_GPU` are passed through when set

Useful tmux commands:

```bash
tmux ls                          # list sessions
tmux attach -t <session>         # watch a session; detach with Ctrl-b then d
tmux kill-session -t <session>   # stop a session
```

---

# Summary

The cell-line-only runners automate the zero-shot evaluation pipeline across CCL splits:

```text
split selection
      ↓
preprocessing
      ↓
training (CCL only)
      ↓
multi-set inference (CCL test + each PDO study)
      ↓
score generation
```