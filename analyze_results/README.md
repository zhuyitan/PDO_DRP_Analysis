# Result Analysis

This directory contains scripts for aggregating and summarizing the outputs produced by the joint-training and transfer-learning experiments.

```text
analyze_results/
├── README.md
├── summarize_joint_training.py
└── summarize_transfer_learning.py
```

The scripts scan completed experiment directories, collect evaluation metrics from each discovered fold combination, report missing or incomplete results, and generate tab-separated summary tables for downstream analysis.

The currently summarized metrics are:

```text
R²
PCC
```

where PCC denotes the Pearson correlation coefficient.

---

# Scripts

## `summarize_joint_training.py`

Summarizes joint-training experiment results across:

```text
split strategy
× study
× CCL fold
× PDO fold
× test set
× metric
```

The configured studies are:

```text
IMPROVE dataset 1
IMPROVE dataset 2
IMPROVE dataset 3
Pilot_1
```

The script supports both:

```text
random
cancer-blind
```

split strategies.

### Joint-Training Test Sets

Three evaluation sets are summarized:

| Display name | Result subdirectory | Description |
|---|---|---|
| `CCL Test` | `ccl/` | General held-out CCL test set |
| `Study CCL` | `mixed/` | Study-associated CCL evaluation set |
| `Study PDO` | `study/` | Held-out PDO study test set |

For each discovered experiment run, the script looks for:

```text
ccl/test_scores.json
mixed/test_scores.json
study/test_scores.json
```

and extracts:

```text
r2
pcc
```

Missing files, malformed JSON, missing metric values, and non-finite metric values are treated as unavailable

---

# Joint-Training Run Discovery

Joint-training run directories are expected to use the naming convention:

```text
{study}_i{i}_j{j}
```

For example:

```text
IMPROVE dataset 1_i0_j0
IMPROVE dataset 1_i0_j1
IMPROVE dataset 1_i1_j0
...
```

where:

```text
i = CCL fold
j = PDO fold
```

The script discovers the available fold combinations automatically rather than assuming that every expected run exists.

This is useful for partially completed experiments because the summary can still be generated while reporting how many results are missing.

---

# Joint-Training Aggregation

For every:

```text
split strategy
× study
× test set
× metric
```

the script aggregates all available CCL/PDO fold combinations.

The reported statistics are:

```text
mean
sample standard deviation
number of available values
number of expected/discovered values
number of missing values
number of score files present
```

Summary values are formatted as:

```text
mean ± standard deviation
```

For example:

```text
0.713 ± 0.042
```

If only one valid result is available:

```text
0.713 ± N/A
```

If no valid results are available:

```text
N/A
```

NaN values are excluded from the calculation.

---

# Joint-Training Output

The joint-training script creates five TSV files:

```text
random_r2_summary.tsv
random_pcc_summary.tsv
cancer_blind_r2_summary.tsv
cancer_blind_pcc_summary.tsv
all_phase2_results_averaged.tsv
```

The four metric-specific tables have the structure:

```text
study    CCL Test    Study CCL    Study PDO
```

Each metric cell contains:

```text
mean ± sample standard deviation
```

The combined:

```text
all_phase2_results_averaged.tsv
```

contains the more detailed averaged results, including availability counts and the underlying split/test-set/metric identifiers.

The script also prints result-availability information to the terminal so incomplete experiment sets can be identified quickly.

---

# `summarize_transfer_learning.py`

Summarizes transfer-learning experiment results across:

```text
split strategy
× study
× CCL fold
× PDO fold
× test set
× metric
```

The configured studies are:

```text
IMPROVE dataset 1
IMPROVE dataset 2
IMPROVE dataset 3
Pilot_1
```

The supported split strategies are:

```text
random
cancer-blind
```

---

# Transfer-Learning Test Sets

The transfer-learning analysis uses two final evaluation sets:

| Display name | Result subdirectory | Description |
|---|---|---|
| `Study CCL` | `mixed/` | Study-associated CCL evaluation set |
| `Study PDO` | `study/` | Held-out PDO study test set |

For each run, the script looks for:

```text
mixed/test_scores.json
study/test_scores.json
```

and extracts:

```text
r2
pcc
```

Unlike the joint-training summarizer, the transfer-learning summarizer expects these metric names directly in the score JSON.

Missing files, invalid JSON, missing metric keys, and non-finite values are recorded as unavailable.

---

# Transfer-Learning Run Discovery

Transfer-learning results are expected to be organized first by study:

```text
<split strategy root>/
└── <study>/
```

and then by source and target fold:

```text
ccl{i}_j{j}
```

For example:

```text
IMPROVE dataset 1/
├── ccl0_j0/
├── ccl0_j1/
├── ccl0_j2/
├── ...
├── ccl1_j0/
└── ...
```

where:

```text
i = pretrained CCL fold
j = PDO fine-tuning fold
```

The run directory pattern is therefore:

```text
ccl0_j0
ccl0_j1
...
ccl9_j9
```

Available runs are discovered automatically.

---

# Transfer-Learning Aggregation

For each:

```text
split strategy
× study
× test set
× metric
```

the script calculates the mean and sample standard deviation over all discovered valid `(i, j)` transfer-learning runs.

It also tracks:

```text
number of discovered runs
number of score files present
number of valid metric values
number of missing metric values
```

This allows the summary tables to be interpreted alongside experiment completeness.

---

# Transfer-Learning Output

The transfer-learning script creates:

```text
cancer_blind_r2_summary.tsv
cancer_blind_pcc_summary.tsv
random_r2_summary.tsv
random_pcc_summary.tsv
all_phase3_results.tsv
```

The four metric summary files have the structure:

```text
study    Study CCL    Study PDO
```

with values formatted as:

```text
mean ± sample standard deviation
```

The complete:

```text
all_phase3_results.tsv
```

is a long-form table containing individual discovered results rather than only the study-level averages.

Its fields include information such as:

```text
split_strategy
study
ccl_fold
pdo_fold
run_name
test_set
test_subdirectory
metric
value
available
score_file_present
source_path
```

This table is useful for more detailed downstream analyses or plotting.


# Expected Experiment Results

Conceptually, the analysis scripts expect experiment results with structures similar to the following.

## Joint Training

```text
<joint-training-root>/
├── IMPROVE dataset 1_i0_j0/
│   ├── ccl/
│   │   └── test_scores.json
│   ├── mixed/
│   │   └── test_scores.json
│   └── study/
│       └── test_scores.json
│
├── IMPROVE dataset 1_i0_j1/
│   └── ...
│
└── ...
```

A typical score file contains metrics such as:

```json
{
    "r2": 0.71,
    "pcc": 0.85
}
```

## Transfer Learning

```text
<transfer-learning-root>/
└── IMPROVE dataset 1/
    ├── ccl0_j0/
    │   ├── mixed/
    │   │   └── test_scores.json
    │   └── study/
    │       └── test_scores.json
    │
    ├── ccl0_j1/
    │   └── ...
    │
    └── ...
```

---

# Running the Analysis

Run the scripts from the repository root.

Joint-training analysis:

```bash
python analyze_results/summarize_joint_training.py
```

Transfer-learning analysis:

```bash
python analyze_results/summarize_transfer_learning.py
```

Both scripts use hardcoded configuration near the top of the file to specify:

```text
input result directories
split strategies
studies
metrics
output directory
```

Verify these paths before running the analysis.

---

# Dependencies

Both scripts require:

```text
Python
NumPy
pandas
```

For example:

```bash
pip install numpy pandas
```

No model-training framework is required to run these summary scripts; they operate on the JSON result files generated by completed experiments.

---

# Missing and Incomplete Results

The scripts are designed to tolerate partially completed experiment collections.

A result can be missing because:

```text
the run directory does not exist
the test_scores.json file does not exist
the JSON file cannot be parsed
the requested metric is absent
the metric is not finite
```

These cases are represented internally as missing values rather than terminating the complete summary process.

Availability statistics are printed so that the number of valid results can be compared with the number of discovered runs.

This is particularly important when interpreting:

```text
mean ± standard deviation
```

because different rows may have been calculated from different numbers of completed runs.

---

# Path Configuration

The paths at the top of each script must match the actual locations where experiment results are stored.

The analysis scripts currently contain model-specific result paths, so these configuration values should be updated if the experiment runners write results somewhere else.

For example, the desired organization could be:

```text
models/
├── GraphDRP/
│   ├── Joint_Training_Results/
│   │   ├── random/
│   │   └── cancer-blind/
│   └── joint_training_summary/
│
└── LGBM/
    ├── transfer_learning_results/
    │   ├── random/
    │   └── cancer-blind/
    └── summarized_transfer_learning_results/
```

Alternatively, if experiment results are stored outside `models/`, update the corresponding root constants in the analysis scripts.

The analysis logic itself is independent of where the result directories are located as long as the configured paths point to the correct experiment outputs.

---

# Summary

The `analyze_results/` directory provides the post-processing stage of the experimental workflow.

```text
experiment runs
      ↓
discover completed fold combinations
      ↓
read test_scores.json
      ↓
extract R² and PCC
      ↓
identify missing results
      ↓
aggregate across CCL × PDO folds
      ↓
calculate mean ± sample SD
      ↓
write TSV summary tables
```

`summarize_joint_training.py` analyzes joint-training performance across CCL Test, Study CCL, and Study PDO evaluation sets.

`summarize_transfer_learning.py` analyzes transfer-learning performance across Study CCL and Study PDO evaluation sets.

Together, the scripts provide standardized study-level summaries while retaining information about experiment completeness and individual fold-level results.