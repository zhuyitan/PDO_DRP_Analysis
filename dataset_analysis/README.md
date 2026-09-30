# PDO Drug-Response Dataset Analysis

This repository contains Python scripts for summarizing and comparing drug-response data from three IMPROVE datasets.

The analyses operate on a common TSV file containing standardized IMPROVE sample identifiers, drug identifiers, dataset source labels, and drug-response AUC values.

The scripts provide:

- Dataset-level counts of samples, drugs, and experiments
- Pairwise overlap analysis across datasets
- Pairwise correlation analysis of matched drug-response measurements


## Requirements

The scripts require Python 3 and the following Python packages:

```text
pandas
matplotlib
scipy
```

They can be installed with:

```bash
pip install pandas matplotlib scipy
```

## Input Data

All scripts expect the input file:

```text
pdo_drp_data/y_data/response.tsv
```

The file must be tab-separated.

### Required Columns

The analyses use the following columns:

| Column | Description |
|---|---|
| `source` | Dataset from which the measurement originated |
| `improve_sample_id` | Standardized IMPROVE sample identifier |
| `improve_chem_id` | Standardized IMPROVE drug/chemical identifier |
| `auc` | Drug-response area under the curve; required by `correlation_analysis.py` |

The scripts analyzing overlap and dataset counts require the first three columns. The correlation analysis additionally requires `auc`.

### Dataset Source Names

The cross-dataset analyses expect the following values in the `source` column:

```text
IMPROVE dataset 1
IMPROVE dataset 2
IMPROVE dataset 3
```

## Analyses

### Dataset Counts

```bash
python dataset_counts.py
```

`dataset_counts.py` summarizes the number of unique drugs, samples, cell lines, organoids, and sample-drug experiments in the complete dataset and within each individual source.

The script calculates:

- Number of unique `improve_chem_id` values
- Number of unique `improve_sample_id` values
- Number of unique cell-line samples
- Number of unique organoid samples
- Number of unique source/drug/sample combinations

For this analysis, samples are classified using their IMPROVE identifiers:

- **Cell lines:** `improve_sample_id` begins with `ACH-`
- **Organoids:** `improve_sample_id` does not begin with `ACH-`

Rows with missing identifiers are excluded from calculations that require those identifiers.

The results are printed to the console and written to:

```text
pdo_drp_data/y_data/dataset_counts.tsv
```

The output contains an `OVERALL` row followed by counts for each individual source.

---

### Dataset Overlap Analysis

```bash
python overlap_analysis.py
```

`overlap_analysis.py` measures overlap among the three IMPROVE datasets for three types of entities:

1. Samples
2. Drugs
3. Experiments

An **experiment** is defined here as a unique combination of:

```text
(improve_sample_id, improve_chem_id)
```

Before calculating overlaps, identifier fields are stripped of surrounding whitespace and empty strings are converted to missing values.

For each entity type, the script constructs a pairwise overlap matrix.

For example, a sample overlap matrix has the general form:

```text
                   Dataset 3    Dataset 2    Dataset 1
Dataset 3              N3           N32          N31
Dataset 2              N32          N2           N21
Dataset 1              N31          N21          N1
```

The diagonal represents the total number of unique entities in each dataset.

The off-diagonal entries represent the number of entities shared by a pair of datasets.

The script also calculates:

- **Intersection:** entities present in all three datasets
- **Union:** entities present in at least one of the three datasets

Outputs are written to:

```text
pdo_drp_data/y_data/overlap_analysis/
```

Specifically:

```text
samples_overlap_matrix.tsv
drugs_overlap_matrix.tsv
experiments_overlap_matrix.tsv
overlap_intersection_union.tsv
```

The matrices and intersection/union summary are also printed to the console.

---

### Pairwise AUC Correlation Analysis

```bash
python correlation_analysis.py
```

`correlation_analysis.py` evaluates correlation in drug-response AUC measurements between the three IMPROVE datasets.

The following pairwise comparisons are performed:

```text
Dataset 1 vs Dataset 2
Dataset 1 vs Dataset 3
Dataset 2 vs Dataset 3
```

Measurements are considered matched when both datasets contain the same:

```text
improve_sample_id
improve_chem_id
```

Only matched sample-drug experiments contribute to a pairwise comparison.

For each comparison, the script reports:

- Number of matched experiments (`n`)
- Pearson correlation coefficient (`PCC`)
- Scatterplot of AUC values
- Identity line representing perfect agreement between datasets

The Pearson correlation coefficient is calculated only when:

- At least two matched measurements are available
- Both datasets contain more than one unique AUC value among those measurements

Otherwise, the PCC is reported as `NA`.

#### Duplicate Measurements

If a dataset contains multiple AUC measurements for the same combination of:

```text
source
improve_sample_id
improve_chem_id
```

the script prints a warning and averages those AUC measurements before matching datasets.

#### Plot Scaling

All three scatterplots use shared x- and y-axis limits so that agreement can be visually compared across dataset pairs.

When all observed AUC values fall within the normalized range of 0–1, the plotting limits are constrained to that range where possible.

The resulting figure is written to:

```text
pdo_drp_data/y_data/improve_pairwise_auc_scatterplots.png
```

The figure contains three panels corresponding to the three pairwise dataset comparisons.

## Running All Analyses

From the repository root, run:

```bash
python dataset_analysis/dataset_counts.py
python dataset_analysis/overlap_analysis.py
python dataset_analysis/correlation_analysis.py
```

Because input and output paths are currently hardcoded, the scripts should be run from the repository root with the expected directory structure present.

## Output Summary

| Script | Output |
|---|---|
| `dataset_counts.py` | `pdo_drp_data/y_data/dataset_counts.tsv` |
| `overlap_analysis.py` | `pdo_drp_data/y_data/overlap_analysis/*.tsv` |
| `correlation_analysis.py` | `pdo_drp_data/y_data/improve_pairwise_auc_scatterplots.png` |



