import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ============================================================
# Configuration
# ============================================================

INPUT_TSV = "pdo_drp_data/y_data/response.tsv"
OUTPUT_FIGURE = "pdo_drp_data/y_data/improve_pairwise_auc_scatterplots.png"


SOURCE_MAP = {
    "Dataset 1": "IMPROVE dataset 1",
    "Dataset 2": "IMPROVE dataset 2",
    "Dataset 3": "IMPROVE dataset 3",
}

PAIRWISE_COMPARISONS = [
    ("Dataset 1", "Dataset 2"),
    ("Dataset 1", "Dataset 3"),
    ("Dataset 2", "Dataset 3"),
]


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_TSV, sep="\t")

required_columns = {
    "source",
    "improve_sample_id",
    "improve_chem_id",
    "auc",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# Keep only the three IMPROVE sources of interest
df = df[df["source"].isin(SOURCE_MAP.values())].copy()

# Ensure AUC is numeric
df["auc"] = pd.to_numeric(df["auc"], errors="coerce")

# Remove rows that cannot participate in matching/correlation
df = df.dropna(
    subset=[
        "source",
        "improve_sample_id",
        "improve_chem_id",
        "auc",
    ]
)


# ============================================================
# Check for duplicate sample-drug measurements within a source
# ============================================================

duplicate_counts = (
    df.groupby(
        ["source", "improve_sample_id", "improve_chem_id"]
    )
    .size()
    .reset_index(name="n")
)

duplicates = duplicate_counts[duplicate_counts["n"] > 1]

if len(duplicates) > 0:
    print(
        f"WARNING: Found {len(duplicates)} source/sample/drug combinations "
        "with more than one AUC value."
    )
    print("Averaging duplicate AUC values before matching.")

    df = (
        df.groupby(
            ["source", "improve_sample_id", "improve_chem_id"],
            as_index=False,
        )["auc"]
        .mean()
    )


# ============================================================
# Function for obtaining matched experiments
# ============================================================

def get_matched_data(df, dataset_x, dataset_y):
    source_x = SOURCE_MAP[dataset_x]
    source_y = SOURCE_MAP[dataset_y]

    x = df[df["source"] == source_x][
        ["improve_sample_id", "improve_chem_id", "auc"]
    ].copy()

    y = df[df["source"] == source_y][
        ["improve_sample_id", "improve_chem_id", "auc"]
    ].copy()

    x = x.rename(columns={"auc": "auc_x"})
    y = y.rename(columns={"auc": "auc_y"})

    matched = x.merge(
        y,
        on=["improve_sample_id", "improve_chem_id"],
        how="inner",
    )

    return matched


# ============================================================
# Build all comparisons first so the axes can be shared
# ============================================================

comparison_data = {}

all_auc_values = []

for dataset_x, dataset_y in PAIRWISE_COMPARISONS:
    matched = get_matched_data(df, dataset_x, dataset_y)

    comparison_data[(dataset_x, dataset_y)] = matched

    all_auc_values.extend(matched["auc_x"].tolist())
    all_auc_values.extend(matched["auc_y"].tolist())

    print(
        f"{dataset_x} vs {dataset_y}: "
        f"{len(matched)} matched experiments"
    )


if not all_auc_values:
    raise ValueError(
        "No matched experiments were found between the selected sources."
    )


# ============================================================
# Common plotting limits
# ============================================================

data_min = min(all_auc_values)
data_max = max(all_auc_values)

padding = 0.05 * (data_max - data_min)

if padding == 0:
    padding = 0.05

axis_min = data_min - padding
axis_max = data_max + padding

# Since normalized AUC is typically around 0-1, avoid extending
# unnecessarily beyond that range when the data are inside it.
if data_min >= 0:
    axis_min = max(0, axis_min)

if data_max <= 1:
    axis_max = min(1, axis_max)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    sharex=True,
    sharey=True,
)

for ax, (dataset_x, dataset_y) in zip(
    axes,
    PAIRWISE_COMPARISONS,
):
    matched = comparison_data[(dataset_x, dataset_y)]

    x = matched["auc_x"]
    y = matched["auc_y"]

    n = len(matched)

    if n >= 2 and x.nunique() > 1 and y.nunique() > 1:
        r, p_value = pearsonr(x, y)
        r_text = f"{r:.3f}"
    else:
        r = float("nan")
        r_text = "NA"

    # Scatterplot
    ax.scatter(
        x,
        y,
        alpha=0.65,
        s=30,
    )

    # Identity line: perfect agreement
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"{dataset_x} AUC")
    ax.set_ylabel(f"{dataset_y} AUC")

    ax.set_title(
        f"{dataset_x} vs {dataset_y}\n"
        f"PCC = {r_text}, n = {n}"
    )

    ax.grid(
        True,
        alpha=0.2,
    )


fig.suptitle(
    "Pairwise Comparison of Matched PDO Drug-Response AUC Values",
    fontsize=15,
    y=1.02,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nSaved figure to:\n{OUTPUT_FIGURE}")