# Transfer Learning Experiments

This directory contains experiment runners for **transfer learning from cancer cell line (CCL) models to PDO study data**.

The currently supported models are:

- **GraphDRP**
- **LightGBM (LGBM)**
- **UNO**

The transfer-learning experiments start from models pretrained on CCL data, fine-tune those models using PDO training data, perform hyperparameter optimization on the PDO validation split, and evaluate the resulting models on held-out study and auxiliary evaluation sets.

---

# Scripts

```text
transfer_learning/
├── run_graphdrp.py
├── run_lgbm.py
├── run_uno.py
└── README.md
```

The runners orchestrate:

```text
pretrained CCL model
        ↓
PDO preprocessing
        ↓
hyperparameter optimization
        ↓
transfer learning / fine-tuning
        ↓
study evaluation
        ↓
additional evaluation
```

---

# Experiment Design

Each transfer-learning experiment is indexed by:

- `i`: pretrained **CCL fold**
- `j`: **PDO study fold**

The studies currently used are:

```text
IMPROVE dataset 1
IMPROVE dataset 2
IMPROVE dataset 3
Pilot_1
```

For the full experiment:

```text
i = 0 ... 9
j = 0 ... 9
```

For random splits, each study can contain:

```text
10 CCL folds × 10 PDO folds = 100 transfer-learning runs
```

Across four studies, this gives up to:

```text
4 × 10 × 10 = 400 runs per model
```

Cancer-blind studies do not all have 10 PDO folds. The UNO runner detects the number for each study.


---

# Data Splits

The GraphDRP and LGBM transfer-learning runners use:

```text
pdo_drp_data/splits/random/
```

The UNO runner runs both:

```text
pdo_drp_data/splits/cancer-blind/
pdo_drp_data/splits/random/
```

PDO fine-tuning splits follow:

```text
{study}_split_{j}_train_ids.txt
{study}_split_{j}_val_ids.txt
{study}_split_{j}_test_ids.txt
```


The study training split is used for transfer learning, the validation split is used for hyperparameter selection, and the test split is reserved for final study evaluation.

Additional study-associated CCL evaluation data may be provided through:

```text
{study}_ccl_ids.txt
```

`Pilot_1` does not use this mixed evaluation set.

---

# Expected Repository Structure

The transfer-learning runners assume the model implementations and pretrained assets are organized under `models/`.

A simplified repository structure is:

```text
PDO_DRP_Analysis_GitHub/
├── models/
│   ├── GraphDRP/
│   ├── HiDRA/
│   ├── LGBM/
│   └── UNO/
│
├── run_experiments/
│   ├── cell_line_only_training/
│   ├── joint_training/
│   └── transfer_learning/
│       ├── README.md
│       ├── run_graphdrp.py
│       ├── run_lgbm.py
│       └── run_uno.py
│
└── pdo_drp_data/
    ├── splits/
    │   ├── random/
    │   └── cancer-blind/
    ├── x_data/
    └── y_data/
```

The runner paths should be configured consistently with this repository structure.

---

# GraphDRP Transfer Learning

Run:

```bash
python run_experiments/transfer_learning/run_graphdrp.py
```

## Pretrained Models

GraphDRP expects one pretrained model for each CCL fold:

```text
models/GraphDRP/pretrained_models/
├── split_0_model.pt
├── split_1_model.pt
├── split_2_model.pt
├── ...
└── split_9_model.pt
```

Each pretrained model corresponds to the same CCL fold index used by the transfer-learning job.

For example:

```text
i = 4
```

uses:

```text
models/GraphDRP/pretrained_models/split_4_model.pt
```

---

# GraphDRP Pretrained Scalers

Transfer-learning preprocessing must use the transformation fitted on the corresponding CCL pretraining fold.

The runner therefore maintains:

```text
models/GraphDRP/pretrained_scalers/
```

with one directory per CCL fold:

```text
pretrained_scalers/
├── ccl0/
│   └── x_data_gene_expression_scaler.gz
├── ccl1/
│   └── x_data_gene_expression_scaler.gz
├── ...
└── ccl9/
    └── x_data_gene_expression_scaler.gz
```

Before transfer-learning jobs begin, the runner checks whether the necessary preprocessing outputs and scalers already exist.

If they do not exist, preprocessing is run on the original CCL train/validation/test split so that the gene-expression scaler is fitted using the source CCL training data.

The resulting scaler is then reused for PDO transfer-learning preprocessing.

This is important because the PDO data should be transformed using the feature scaling learned from the corresponding source CCL fold rather than fitting a new scaler on the target PDO data.

---

# GraphDRP Transfer Strategy

The GraphDRP transfer-learning experiment keeps both pretrained encoders frozen:

```python
FIXED_FREEZE_LAYERS = "Freeze Both"
```

The experiment therefore focuses on adaptation of the prediction head rather than retraining the feature encoders.

The hyperparameter search considers different prediction-head configurations.

---

# GraphDRP Hyperparameter Optimization

Hyperparameter optimization uses SciPy's:

```python
scipy.optimize.differential_evolution
```

with validation:

```text
MSE
```

as the optimization objective.

The optimized parameters are:

### Batch size

```text
4
8
16
```

### Learning rate

The learning rate is searched on a log10 scale:

```text
10^-5 to 10^-3
```

### Prediction-head freezing

The following strategies are considered:

```text
None
Freeze FC1
Freeze FC1 FC2
Freeze Head
```

### Dropout

Prediction-head dropout is optimized continuously over:

```text
0.0 to 0.9
```

The pretrained GraphDRP encoders remain fixed during all HPO trials.

The default differential-evolution configuration includes:

```text
maxiter       = 5
popsize       = 5
tolerance     = 0.01
seed          = 42
strategy      = best1bin
mutation      = 0.5–1.0
recombination = 0.7
```

Each candidate is trained and evaluated using validation MSE.

The best configuration is written to:

```text
best_hyperparameters.json
```

A history of HPO evaluations is written to:

```text
hpo_history.jsonl
```

Temporary HPO model directories can be deleted after their scores have been recorded.

---

# GraphDRP Workflow

For each:

```text
(study, CCL fold i, PDO fold j)
```

the runner performs approximately:

```text
Verify CCL pretrained model
        ↓
Recover/generate CCL preprocessing scaler
        ↓
Preprocess PDO train/validation/test
using CCL fold scaler
        ↓
Differential-evolution HPO
        ↓
Select best transfer-learning parameters
        ↓
Final transfer training
from pretrained CCL checkpoint
        ↓
Study test inference
        ↓
Mixed CCL inference
        ↓
Save scores, predictions, model,
HPO configuration, and timing
```

The mixed evaluation stage is omitted for `Pilot_1`.

---

# GraphDRP GPU Scheduling

GraphDRP supports multiple simultaneous jobs on each GPU.

The runner defines a slot count per GPU:

```python
GPU_SLOTS = {
    0: 4,
    1: 4,
    2: 4,
    3: 4,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
}
```

The total number of active transfer jobs is:

```python
MAX_ACTIVE_JOBS = sum(GPU_SLOTS.values())
```

With the configuration above:

```text
8 GPUs × 4 slots = 32 active jobs
```

A shared GPU queue assigns a GPU to each transfer-learning process.

Each subprocess is launched with its assigned GPU exposed through:

```text
CUDA_VISIBLE_DEVICES
```

Adjust the slot counts based on GPU memory requirements.

---

# GraphDRP Output

Results are written under:

```text
GraphDRP_transfer_learning_results/
```

using a structure based on study, CCL fold, and PDO fold:

```text
GraphDRP_transfer_learning_results/
└── IMPROVE dataset 1/
    ├── ccl0_j0/
    ├── ccl0_j1/
    ├── ...
    ├── ccl1_j0/
    └── ...
```

A completed run contains important outputs including:

```text
model.pt
best_hyperparameters.json
hpo_history.jsonl
timing.json
study/
mixed/
```

For `Pilot_1`, the `mixed/` result is not required.

A run is treated as complete only when the required model, HPO, timing, and inference outputs exist.

This allows completed jobs to be skipped when the experiment is restarted.

---

# LightGBM Transfer Learning

Run:

```bash
python run_experiments/transfer_learning/run_lgbm.py
```

---

# LGBM Pretrained Assets

Each CCL fold provides three source-domain artifacts:

```text
model.txt
x_data_gene_expression_scaler.gz
x_data_mordred_scaler.gz
```

The intended structure is:

```text
models/LGBM/pretrained_models/
├── ccl_fold_0/
│   ├── model.txt
│   ├── x_data_gene_expression_scaler.gz
│   └── x_data_mordred_scaler.gz
│
├── ccl_fold_1/
│   ├── model.txt
│   ├── x_data_gene_expression_scaler.gz
│   └── x_data_mordred_scaler.gz
│
├── ...
│
└── ccl_fold_9/
    ├── model.txt
    ├── x_data_gene_expression_scaler.gz
    └── x_data_mordred_scaler.gz
```

For example:

```text
i = 6
```

uses:

```text
models/LGBM/pretrained_models/ccl_fold_6/model.txt
models/LGBM/pretrained_models/ccl_fold_6/x_data_gene_expression_scaler.gz
models/LGBM/pretrained_models/ccl_fold_6/x_data_mordred_scaler.gz
```

The corresponding gene-expression and Mordred scalers are reused when preprocessing the target PDO data.

---

# LGBM Preprocessing

LGBM transfer-learning preprocessing uses:

```text
lgbm_preprocess_improve.py
```

The preprocessing stage receives:

```text
--pretrained_ge_scaler
--pretrained_md_scaler
```

so that the PDO features are transformed using the preprocessing parameters learned from the corresponding source CCL fold.

For every `(i, j)` combination:

```text
CCL fold i
    ↓
pretrained model
pretrained GE scaler
pretrained Mordred scaler
    ↓
PDO fold j
```

This keeps the source-domain transformation consistent between CCL pretraining and PDO transfer learning.

---

# LGBM Hyperparameter Optimization

LGBM also uses differential evolution for transfer-learning hyperparameter optimization.

The optimization metric is:

```text
MSE
```

The optimized parameters are:

### `min_data_in_leaf`

Search range:

```text
2 to 64
```

### Learning rate

Search range:

```text
0.001 to 0.05
```

The learning rate is optimized in log10 space.

The default differential-evolution settings are:

```text
maxiter   = 5
popsize   = 5
tolerance = 0.01
seed      = 42
```

Each HPO candidate fine-tunes the pretrained LightGBM model and evaluates the resulting validation MSE.

Temporary trials are stored under:

```text
hpo_trials/
```

Results from individual evaluations are recorded in:

```text
hpo_history.jsonl
```

The selected parameters are stored in:

```text
best_hyperparameters.json
```

If `best_hyperparameters.json` already exists, the saved HPO result is reused rather than repeating the optimization.

---

# LGBM Transfer Training

After HPO, the pretrained CCL model is fine-tuned using the selected parameters.

The transfer training uses:

```text
--pretrained_model
```

together with the optimized:

```text
min_data_in_leaf
learning_rate
```

and:

```text
transfer_rounds = 10000
```

The resulting transferred model is written as:

```text
model.txt
```

---

# LGBM Evaluation Sets

The final transferred LightGBM model is evaluated on multiple datasets.

## 1. Study Test Set

The primary held-out PDO test set is:

```text
{study}_split_{j}_test_ids.txt
```

Results are written under:

```text
study/
```

including:

```text
study/test_scores.json
study/test_y_data_predicted.csv
```

---

## 2. Mixed Evaluation Set

For non-pilot studies, the model is also evaluated using:

```text
{study}_ccl_ids.txt
```

This dataset receives a separate preprocessing pass using the same pretrained CCL fold scalers.

The output is written under:

```text
mixed/
```

including:

```text
mixed/test_scores.json
mixed/test_y_data_predicted.csv
```

`Pilot_1` skips this stage.

---

## 3. Fine-Tuning Training Set

The LGBM runner additionally evaluates the final transferred model on its fine-tuning training set.

Because the inference script operates on the preprocessed `test` partition, the runner creates another preprocessing dataset where:

```text
train split → train
validation split → validation
train split → test
```

This allows the existing inference implementation to generate predictions for the fine-tuning training observations without modifying the inference script.

Results are written under:

```text
train/
```

including:

```text
train/test_scores.json
train/test_y_data_predicted.csv
```

This output can be useful for comparing training performance with held-out study performance.

---

# LGBM Workflow

For each:

```text
(study, CCL fold i, PDO fold j)
```

the workflow is:

```text
Load pretrained CCL fold model
        ↓
Load pretrained GE scaler
        ↓
Load pretrained Mordred scaler
        ↓
Verify PDO split files
        ↓
Preprocess PDO train/val/test
using pretrained CCL scalers
        ↓
Differential-evolution HPO
        ↓
Select best:
    min_data_in_leaf
    learning_rate
        ↓
Fine-tune pretrained LightGBM model
        ↓
Study test inference
        ↓
Mixed-set preprocessing + inference
        ↓
Training-set preprocessing + inference
        ↓
Write timing and experiment metadata
```

---

# LGBM Parallel Execution

The maximum number of concurrently running LGBM transfer jobs is controlled by:

```python
MAX_PARALLEL_JOBS = 4
```

Jobs are executed using:

```python
ProcessPoolExecutor
```

The full experiment contains combinations of:

```text
study
CCL fold i
PDO fold j
```

When:

```python
PILOT_MODE = False
```

the runner generates:

```text
4 studies × 10 CCL folds × 10 PDO folds
= 400 jobs
```

When:

```python
PILOT_MODE = True
```

only:

```text
i = 0
j = 0
```

is run for each configured study.

This provides a small end-to-end test before launching the full transfer-learning experiment.

---

# LGBM Output

Results are written under:

```text
LGBM_Transfer_Learning_Results/
```

with a structure such as:

```text
LGBM_Transfer_Learning_Results/
└── IMPROVE dataset 1/
    ├── ccl0_j0/
    │   ├── model.txt
    │   ├── best_hyperparameters.json
    │   ├── hpo_history.jsonl
    │   ├── timing.json
    │   ├── study/
    │   ├── mixed/
    │   └── train/
    │
    ├── ccl0_j1/
    ├── ccl0_j2/
    ├── ...
    └── ccl9_j9/
```

For a complete non-pilot run, important required outputs include:

```text
model.txt
best_hyperparameters.json
timing.json

study/test_scores.json
study/test_y_data_predicted.csv

mixed/test_scores.json
mixed/test_y_data_predicted.csv

train/test_scores.json
train/test_y_data_predicted.csv
```

For `Pilot_1`, the mixed outputs are not required.

---

# UNO Transfer Learning

Run:

```bash
python run_experiments/transfer_learning/run_uno.py
```

Requirements:

- `models/UNO/uno_finetune_improve.py`, with a `"canc_drug"` entry (`CANC_LAYERS | DRUG_LAYERS`) in `FREEZE_GROUPS`
- `scipy`

---

# UNO Pretrained Models

UNO starts from the models trained by:

```text
run_experiments/cell_line_only_training/run_uno.py
```

The pretrained model for CCL fold `i` is read from:

```python
PRETRAIN_DIR_TEMPLATE = "cell_line_only_results_uno/{split_type}/split_{i}/processed_ccl"
```

Each of these folders must contain:

```text
model.h5
x_data_gene_expression_scaler.gz
x_data_mordred_scaler.gz
```

For example:

```text
split_type = cancer-blind
i = 4
```

uses:

```text
cell_line_only_results_uno/cancer-blind/split_4/processed_ccl/
```

Before any GPU work starts, the runner checks that all 10 CCL folds have both scalers, and stops with a list of missing files if not.

To use pretrained models saved somewhere else, change `PRETRAIN_DIR_TEMPLATE`.

---

# UNO Preprocessing

PDO data is preprocessed with:

```text
uno_preprocess_improve.py --pretrained_model_dir <CCL fold i folder>
```

so the gene-expression and Mordred scalers fitted on the source CCL fold are loaded and reused rather than refit on the PDO data.

---

# UNO Transfer Strategy

The UNO transfer-learning experiment keeps both the cancer and drug encoders frozen:

```python
FIXED_FREEZE_GROUP = "canc_drug"
```

The hyperparameter search controls how many of the five post-concatenation layers are left trainable.

---

# UNO Hyperparameter Optimization

UNO uses SciPy's `differential_evolution`, minimizing:

```text
val_mse
```

on the study validation split.

The optimized parameters are:

### Batch size

```text
4
8
16
```

### Learning rate

Searched on a log10 scale:

```text
10^-5 to 10^-3
```

### Unfrozen post-concatenation layers

```text
0
1
2
3
```

### Dropout

```text
0, 0.1, 0.2, 0.4, 0.6, 0.9
```

The differential-evolution settings are:

```text
maxiter       = 5
popsize       = 5
tolerance     = 0.01
seed          = 42
strategy      = best1bin
mutation      = 0.5–1.0
recombination = 0.7
```

This gives up to 120 candidates per job. Every HPO trial and the final training use:

```text
finetune_epochs         = 200
early_stopping_patience = 20
```

The best configuration is written to `best_hyperparameters.json`, and every evaluation to `hpo_history.jsonl`. Temporary trial folders are deleted after their scores are recorded.

---

# UNO Workflow

For each:

```text
(split_type, study, CCL fold i, PDO fold j)
```

the runner performs:

```text
Load CCL fold i pretrained model and scalers
        ↓
Preprocess PDO train/val/test
using the CCL fold scalers
        ↓
Differential-evolution HPO
        ↓
Final fine-tuning with the best configuration
        ↓
Study test inference
        ↓
Training-set preprocessing + inference
        ↓
Mixed preprocessing + inference   (skipped for Pilot_1)
        ↓
Write performance summary and timing
```

The split types run in the order given by:

```python
SPLIT_TYPES = ["cancer-blind", "random"]
```

Set `PILOT_MODE = True` to run only `i = 0`, `j = 0` for each study as a quick end-to-end test.

---

# UNO GPU Scheduling

No GPU settings need to be edited. The runner detects the available GPUs when it starts and runs one job per GPU at a time.

- Set `CUDA_VISIBLE_DEVICES` to limit which GPUs are used
- Set `JOBS_PER_GPU` to run more than one job per GPU. The batch sizes are small, so `JOBS_PER_GPU=3` usually fits.
- If no GPU is found, jobs run on the CPU

```bash
JOBS_PER_GPU=3 python run_experiments/transfer_learning/run_uno.py
```

---

# UNO Output

Results are written under:

```text
transfer_learning_results_uno/
```

with one folder per split type:

```text
transfer_learning_results_uno/
└── cancer-blind/
    ├── results_summary.jsonl
    └── IMPROVE dataset 1/
        ├── ccl0_j0/
        │   ├── model.h5
        │   ├── best_hyperparameters.json
        │   ├── hpo_history.jsonl
        │   ├── performance_summary.json
        │   ├── timing.json
        │   ├── study/
        │   ├── train/
        │   └── mixed/
        │
        ├── ccl0_j1/
        └── ...
```

`performance_summary.json` collects the train, validation, test and mixed scores for one run. `results_summary.jsonl` collects one row per run and split for the whole split type.

For `Pilot_1`, the `mixed/` result is not required.

---

# Resume Behavior

All three runners are designed so that large experiments can be restarted.

## GraphDRP

A GraphDRP transfer-learning run is considered complete when the required model, HPO configuration, timing data, and evaluation scores exist.

Completed runs are skipped.

Previously selected HPO parameters can also be reused through:

```text
best_hyperparameters.json
```

The GraphDRP runner validates that a saved HPO result corresponds to the currently configured search space before reusing it.

## LGBM

LGBM checks for the complete set of required outputs before beginning a run.

If all required files exist:

```text
[SKIP DONE]
```

is emitted and the `(study, i, j)` combination is not rerun.

Existing:

```text
best_hyperparameters.json
```

files are also reused so HPO does not have to be repeated unnecessarily.

## UNO

A UNO run is considered complete when its model, HPO configuration, timing, performance summary, and study, training-set and mixed scores exist. Completed runs are skipped.

A saved `best_hyperparameters.json` is reused only if it matches the current search space and training length.

---

# Running in tmux

`run_experiments/launch_tmux.sh` starts a runner in its own detached tmux session, so it keeps running after you disconnect. It activates the given conda environment inside the session and saves all output to `logs/`.

```bash
run_experiments/launch_tmux.sh <conda_env> <runner.py> [work_dir]
```

Example:

```bash
run_experiments/launch_tmux.sh <uno_env> run_experiments/transfer_learning/run_uno.py
```

- The session is named after the folder and model, for example `transfer_learning-uno`
- `work_dir` is the folder holding `pdo_drp_data/` and `models/` (default: the repository root)
- `CUDA_VISIBLE_DEVICES` and `JOBS_PER_GPU` are passed through when set

Useful tmux commands:

```bash
tmux ls                          # list sessions
tmux attach -t <session>         # watch a session; detach with Ctrl-b then d
tmux kill-session -t <session>   # stop a session
```

---

# Timing

All three transfer-learning experiments record runtime information.

Each completed experiment contains:

```text
timing.json
```

The timing metadata identifies the relevant study and fold combination and records experiment runtime.

This can be used to compare the computational requirements of different models or transfer-learning configurations.