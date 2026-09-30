from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Hardcoded configuration
# ---------------------------------------------------------------------

INPUT_TSV = Path(
    "pdo_drp_data/y_data/response.tsv"
)

OUTPUT_DIR = Path(
    "pdo_drp_data/y_data/overlap_analysis"
)

OVERLAP_SOURCES = [
    "IMPROVE dataset 3",
    "IMPROVE dataset 2",
    "IMPROVE dataset 1",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def clean_identifier_column(series):
    """
    Strip whitespace and convert empty strings to missing values.
    """
    return series.astype("string").str.strip().replace("", pd.NA)


def get_source_sets(data, value_columns):
    """
    Build one set of unique entities for each source.

    For one column:
        values are strings.

    For multiple columns:
        values are tuples.
    """
    source_sets = {}

    for source in OVERLAP_SOURCES:
        subset = data.loc[
            data["source"] == source,
            value_columns,
        ].dropna()

        if len(value_columns) == 1:
            values = set(
                subset[value_columns[0]].tolist()
            )
        else:
            values = set(
                subset.itertuples(index=False, name=None)
            )

        source_sets[source] = values

    return source_sets


def make_overlap_matrix(source_sets):
    """
    Create a count matrix.

    Diagonal:
        number of unique entities in that source.

    Off-diagonal:
        pairwise intersection size.
    """
    rows = []

    for row_source in OVERLAP_SOURCES:
        row = {
            "source": row_source
        }

        for column_source in OVERLAP_SOURCES:
            row[column_source] = len(
                source_sets[row_source]
                & source_sets[column_source]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_intersection_and_union(source_sets):
    """
    Calculate the intersection across all sources and
    the union across all sources.
    """
    sets = [
        source_sets[source]
        for source in OVERLAP_SOURCES
    ]

    intersection = set.intersection(*sets)
    union = set.union(*sets)

    return len(intersection), len(union)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(
    INPUT_TSV,
    sep="\t",
    dtype={
        "source": "string",
        "improve_sample_id": "string",
        "improve_chem_id": "string",
    },
)


# ---------------------------------------------------------------------
# Validate columns
# ---------------------------------------------------------------------

required_columns = {
    "source",
    "improve_sample_id",
    "improve_chem_id",
}

missing_columns = required_columns.difference(
    df.columns
)

if missing_columns:
    raise ValueError(
        "Input file is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )


# ---------------------------------------------------------------------
# Clean identifiers
# ---------------------------------------------------------------------

df["source"] = clean_identifier_column(
    df["source"]
)

df["improve_sample_id"] = clean_identifier_column(
    df["improve_sample_id"]
)

df["improve_chem_id"] = clean_identifier_column(
    df["improve_chem_id"]
)


# ---------------------------------------------------------------------
# Validate requested sources
# ---------------------------------------------------------------------

available_sources = set(
    df["source"].dropna().unique()
)

missing_sources = [
    source
    for source in OVERLAP_SOURCES
    if source not in available_sources
]

if missing_sources:
    raise ValueError(
        "The following requested overlap sources were not found: "
        + ", ".join(missing_sources)
    )


# ---------------------------------------------------------------------
# Keep only the requested sources
# ---------------------------------------------------------------------

overlap_df = df.loc[
    df["source"].isin(OVERLAP_SOURCES)
].copy()


# ---------------------------------------------------------------------
# Define the three entity types
# ---------------------------------------------------------------------

sample_sets = get_source_sets(
    overlap_df,
    ["improve_sample_id"],
)

drug_sets = get_source_sets(
    overlap_df,
    ["improve_chem_id"],
)

# Experiment = unique sample-drug combination.
experiment_sets = get_source_sets(
    overlap_df,
    [
        "improve_sample_id",
        "improve_chem_id",
    ],
)


entity_sets = {
    "samples": sample_sets,
    "drugs": drug_sets,
    "experiments": experiment_sets,
}


# ---------------------------------------------------------------------
# Write heatmap count matrices
# ---------------------------------------------------------------------

for entity_name, source_sets in entity_sets.items():

    matrix = make_overlap_matrix(
        source_sets
    )

    output_path = (
        OUTPUT_DIR
        / f"{entity_name}_overlap_matrix.tsv"
    )

    matrix.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"\n{entity_name.capitalize()} overlap matrix"
    )

    print(
        matrix.to_string(index=False)
    )

    print(
        f"Wrote: {output_path}"
    )


# ---------------------------------------------------------------------
# Calculate overall intersection and union
# ---------------------------------------------------------------------

summary_rows = []

for entity_name, source_sets in entity_sets.items():

    intersection, union = (
        calculate_intersection_and_union(
            source_sets
        )
    )

    summary_rows.append(
        {
            "entity": entity_name,
            "intersection": intersection,
            "union": union,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Write intersection / union summary
# ---------------------------------------------------------------------

summary_path = (
    OUTPUT_DIR
    / "overlap_intersection_union.tsv"
)

summary.to_csv(
    summary_path,
    sep="\t",
    index=False,
)


# ---------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------

print(
    "\nIntersection / union summary"
)

print(
    summary.to_string(index=False)
)

print(
    f"\nWrote: {summary_path}"
)