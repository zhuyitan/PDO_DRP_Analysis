# Joint Training Experiments

This directory contains experiment runners for training and evaluating multiple drug response prediction models across PDO studies and cancer cell line (CCL) splits.

The runners currently support:

- **GraphDRP**
- **HiDRA**
- **LightGBM (LGBM)**

Each script coordinates preprocessing, model training, inference on multiple test sets, parallel execution, and experiment output management.

## Scripts

```text
joint_training/
├── run_graphdrp.py
├── run_hidra.py
└── run_lgbm.py
```

These are orchestration scripts. The model-specific preprocessing, training, and inference scripts must also be available in the expected repository locations.

---

# Experiment Design

The joint-training experiments use combinations of:

- a **PDO study split**, indexed by `j`
- a **CCL split**, indexed by `i`

The studies currently used are:

```text
IMPROVE dataset 1
IMPROVE dataset 2
IMPROVE dataset 3
Pilot_1
```

For a given `(study, i, j)` combination, the training data is defined by a split file such as:

```text
{study}_split_{j}_ccl_split_{i}_train_ids.txt
```

Validation is performed using:

```text
{study}_split_{j}_val_ids.txt
```

Depending on the model runner, the trained model is then evaluated on up to three test populations:

| Test set | Split file |
|---|---|
| CCL | `ccl_split_{i}_test_ids.txt` |
| Study/PDO | `{study}_split_{j}_test_ids.txt` |
| Mixed / study CCL | `{study}_ccl_ids.txt` |

`Pilot_1` does not use the mixed / study-CCL test set.

---

# GraphDRP

Run:

```bash
python run_experiments/joint_training/run_graphdrp.py
```

## Configuration

The main settings are defined near the top of `run_graphdrp.py`.

Example:

```python
STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1",
]

BASE_SPLIT_DIR = "pdo_drp_data/splits/random"
RAW_DATA_DIR = "pdo_drp_data"
BASE_OUTPUT_DIR = "joint_training_results_graphdrp"

NUM_GPUS = 8
SLOTS_PER_GPU = 4
MAX_ACTIVE_FOLDS = 32
```

GraphDRP creates jobs across all configured study, PDO split, and CCL split combinations.

With 4 studies, 10 PDO splits, and 10 CCL splits, this corresponds to:

```text
4 × 10 × 10 = 400 experiment folds
```

## GPU scheduling

GraphDRP maintains a shared GPU queue.

For example:

```python
NUM_GPUS = 8
SLOTS_PER_GPU = 4
```

allows up to:

```text
8 × 4 = 32
```

GPU-assigned jobs to run concurrently.

`MAX_ACTIVE_FOLDS` additionally limits how many complete pipelines may be active at once.

Adjust these values according to available GPU memory, CPU memory, and system I/O capacity.

## Workflow

For each `(study, i, j)` job, GraphDRP performs:

```text
CCL preprocessing
        │
        ▼
Model training
        │
        ▼
CCL inference
        │
        ▼
Delete temporary preprocessing tensors
        │
        ▼
Study/PDO preprocessing
        │
        ▼
Study/PDO inference
        │
        ▼
Delete temporary preprocessing tensors
        │
        ▼
Mixed preprocessing
        │
        ▼
Mixed inference
        │
        ▼
Delete temporary preprocessing tensors
```

The mixed stage is skipped for `Pilot_1`.

The same trained model is reused for the different inference sets.

## Disk cleanup

GraphDRP preprocessing generates:

```text
processed/train_data.pt
processed/val_data.pt
processed/test_data.pt
```

These tensors can consume substantial disk space.

The runner deletes only these temporary `.pt` files after the corresponding inference stage.

Important outputs such as the following are retained:

```text
model.pt
train_y_data.csv
val_y_data.csv
test_y_data.csv
val_y_data_predicted.csv
test_y_data_predicted.csv
val_scores.json
test_scores.json
history.csv
runtime_preprocess.json
runtime_train.json
runtime_infer.json
```

A final safety-cleanup block also attempts to remove remaining preprocessing tensors if a pipeline fails partway through execution.

## Output

Results are written under:

```text
joint_training_results_graphdrp/
```

with one directory per experiment combination.

A run contains separate processed and inference directories for CCL, study, and mixed evaluation.

---

# HiDRA

Run:

```bash
python run_experiments/joint_training/run_hidra.py
```

## Expected HiDRA repository

The runner expects HiDRA under:

```text
models/HiDRA/
```

with:

```text
models/HiDRA/
├── HiDRA_preprocess_improve.py
├── HiDRA_train_improve.py
└── HiDRA_infer_improve.py
```

The configured paths are:

```python
HIDRA_REPO_DIR = Path("models/HiDRA")

PREPROCESS_SCRIPT = HIDRA_REPO_DIR / "HiDRA_preprocess_improve.py"
TRAIN_SCRIPT = HIDRA_REPO_DIR / "HiDRA_train_improve.py"
INFER_SCRIPT = HIDRA_REPO_DIR / "HiDRA_infer_improve.py"
```

## Configuration

Important settings include:

```python
RAW_INPUT_DIR = Path("pdo_drp_data")
SPLIT_DIR = RAW_INPUT_DIR / "splits" / "random"

EXPERIMENT_ROOT = Path("HiDRA_joint_training_results")

GPUS = list(range(7))
CELL_LINE_SPLITS = list(range(10))
```

Training hyperparameters are configured directly in the runner:

```python
EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.001
PATIENCE = 20
LOSS = "mse"
```

## HiDRA configuration files

HiDRA uses generated configuration files rather than passing every option directly on the command line.

Each experiment creates configuration files defining:

- preprocessing input/output
- train/validation/test split files
- model hyperparameters
- training input/output directories
- inference data directory
- trained model directory
- inference output directory

This keeps the exact configuration used for each run alongside its experiment outputs.

## Split discovery

PDO split indices are discovered automatically from files matching:

```text
{study}_split_{j}_val_ids.txt
```

A PDO split is used only when both the corresponding validation and test files exist.

CCL split indices are configured through:

```python
CELL_LINE_SPLITS = list(range(10))
```

Before scheduling a job, the runner checks that the required split files exist.

Missing combinations are skipped and written to the experiment log.

## GPU scheduling

HiDRA assigns jobs to GPUs in round-robin order.

For example:

```python
GPUS = list(range(7))
```

creates one worker for each of GPUs:

```text
0 1 2 3 4 5 6
```

Each GPU worker processes its assigned jobs sequentially.

Commands are launched with:

```text
CUDA_VISIBLE_DEVICES=<gpu_id>
```

so each experiment is pinned to its assigned GPU.

## Workflow

For each fold, HiDRA first runs:

```text
PDO preprocess
    ↓
training
    ↓
PDO inference
```

The trained model is then reused for additional inference datasets.

When available:

```text
{study}_ccl_ids.txt
```

is preprocessed and evaluated.

The model is also evaluated using:

```text
ccl_split_{i}_test_ids.txt
```

without retraining.

For `Pilot_1`, the optional `{study}_ccl_ids.txt` stage is skipped when the file is unavailable.

## Resume behavior

Completed HiDRA runs create:

```text
RUN_COMPLETE.marker
```

Failed runs create:

```text
RUN_FAILED.marker
```

By default:

```python
FORCE_RERUN = False
```

causes completed jobs to be skipped when the script is restarted.

Set:

```python
FORCE_RERUN = True
```

to rerun completed experiments.

## Logging

HiDRA produces both a global experiment log and per-run terminal logs.

Global log:

```text
HiDRA_joint_training_results/experiment.log
```

Per-run log:

```text
terminal_output.log
```

Output from preprocessing, training, and inference is streamed to both the terminal and the corresponding per-run log.

---

# LightGBM

Run:

```bash
python run_experiments/joint_training/run_lgbm.py
```

## Configuration

Important settings include:

```python
STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1",
]

BASE_SPLIT_DIR = "pdo_drp_data/splits/cancer-blind"
RAW_DATA_DIR = "pdo_drp_data"

BASE_OUTPUT_DIR = "./LGBM_Joint_Training_Results"

MAX_PARALLEL_JOBS = 6
START_FROM = None
```

The maximum number of simultaneous experiment processes is controlled by:

```python
MAX_PARALLEL_JOBS
```

For example:

```python
MAX_PARALLEL_JOBS = 6
```

allows six `(study, i, j)` pipelines to execute concurrently.

## Automatic split detection

Unlike assuming a fixed number of folds, the LGBM runner scans the configured split directory for files matching:

```text
{study}_split_{j}_ccl_split_{i}_train_ids.txt
```

and extracts the available `i` and `j` values.

This allows the runner to operate only on split combinations present in the dataset.

## Workflow

Each LightGBM job performs:

```text
Preprocess
    ↓
Train
    ↓
CCL inference
    ↓
Study inference
    ↓
Mixed inference
    ↓
Cleanup
```

The mixed inference stage is skipped for `Pilot_1`.

The model is trained once for each `(study, i, j)` combination and reused for all inference variants.

## Resume behavior

Before starting a run, the runner checks for:

```text
ccl/test_scores.json
study/test_scores.json
mixed/test_scores.json
```

For `Pilot_1`, only:

```text
ccl/test_scores.json
study/test_scores.json
```

are required.

If all required output files already exist, the job is considered complete and is skipped.

Individual inference stages are also skipped when their corresponding:

```text
test_scores.json
```

already exists.

This allows interrupted experiments to be restarted without repeating completed evaluation stages.

## Starting from a specific fold

The runner includes:

```python
START_FROM = None
```

To begin scheduling from a specific experiment:

```python
START_FROM = ("IMPROVE dataset 2", 3, 5)
```

where the tuple is:

```text
(study, ccl_split_i, study_split_j)
```

All jobs appearing before this combination in the generated job list are ignored.

## Outputs

After a complete experiment, the LGBM runner preserves:

```text
model.txt
val_scores.json
param_log_file.txt
val_y_data_predicted.csv
```

and the inference directories:

```text
ccl/
study/
mixed/
```




---



# Recommended Environment

A typical environment should include:

- Python 3
- the dependencies required by the corresponding model
- CUDA-compatible PyTorch for GraphDRP
- LightGBM and its Python dependencies for LGBM
- sufficient CPU RAM for concurrent preprocessing
- sufficient disk space for intermediate datasets and model outputs

GPU availability should match the GPU configuration in the runner before starting GraphDRP or HiDRA.

For example, if only four GPUs are available, do not leave:

```python
NUM_GPUS = 8
```

or:

```python
GPUS = list(range(7))
```

unchanged.

---

# Split Naming Convention

The runners depend on consistent split filenames.

For PDO fold `j` and CCL fold `i`:

```text
{study}_split_{j}_ccl_split_{i}_train_ids.txt
{study}_split_{j}_val_ids.txt
{study}_split_{j}_test_ids.txt
ccl_split_{i}_test_ids.txt
```

For the mixed/study-CCL evaluation:

```text
{study}_ccl_ids.txt
```

Example:

```text
IMPROVE dataset 1_split_2_ccl_split_7_train_ids.txt
IMPROVE dataset 1_split_2_val_ids.txt
IMPROVE dataset 1_split_2_test_ids.txt
ccl_split_7_test_ids.txt
IMPROVE dataset 1_ccl_ids.txt
```

Here:

```text
j = 2
i = 7
```

---

# Output Interpretation

Each `(study, i, j)` combination represents a trained model.

The different inference outputs evaluate that same model against different test sets:

### CCL

```text
ccl_split_{i}_test_ids.txt
```

Evaluates performance on the held-out cancer cell line split.

### Study

```text
{study}_split_{j}_test_ids.txt
```

Evaluates performance on the held-out PDO/study test split.

### Mixed

```text
{study}_ccl_ids.txt
```

Evaluates performance on the study-associated CCL population when that split is available.


---

# Important Notes

### Split strategy

The current scripts are not all configured for the same split directory.

GraphDRP and HiDRA currently use:

```text
pdo_drp_data/splits/random
```

while the provided LGBM runner uses:

```text
pdo_drp_data/splits/cancer-blind
```

These scripts can work for either set of splits, so to run the whole experiment for both split strategies you would run the scripts once for the "random" splits directory and once for "cancer-blind"

### Concurrency

Increasing concurrency can significantly increase:

- GPU memory usage
- system RAM usage
- disk I/O
- preprocessing load
- temporary storage usage

Tune the parallelism settings for the machine being used.



---


# Summary

The joint-training runners automate the full evaluation pipeline across study and CCL folds:

```text
split selection
      ↓
preprocessing
      ↓
training
      ↓
multi-set inference
      ↓
score generation
      ↓
cleanup / resume bookkeeping
```
