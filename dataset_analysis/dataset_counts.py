import pandas as pd

# Hardcoded input/output paths
input_file = "pdo_drp_data/y_data/response.tsv"
output_file = "pdo_drp_data/y_data/dataset_counts.tsv"

# Read TSV
df = pd.read_csv(input_file, sep="\t", dtype=str)

# Columns required for the analysis
required_columns = [
    "source",
    "improve_sample_id",
    "improve_chem_id",
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Remove rows with missing values in the relevant columns
# for each calculation as appropriate.


def calculate_counts(data):
    # Unique drugs
    unique_chem_ids = data["improve_chem_id"].dropna().nunique()

    # Unique samples, regardless of type
    unique_sample_ids = data["improve_sample_id"].dropna().nunique()

    # Cell lines: sample IDs starting with "ACH-"
    cell_line_mask = data["improve_sample_id"].fillna("").str.startswith("ACH-")
    unique_cell_line_ids = data.loc[
        cell_line_mask, "improve_sample_id"
    ].nunique()

    # Organoids: sample IDs NOT starting with "ACH-"
    # Missing sample IDs are excluded.
    organoid_mask = (
        data["improve_sample_id"].notna()
        & ~data["improve_sample_id"].str.startswith("ACH-", na=False)
    )
    unique_organoid_ids = data.loc[
        organoid_mask, "improve_sample_id"
    ].nunique()

    # Unique source + drug + sample combinations.
    #
    # When this function is run on a single source, this is effectively
    # the number of unique drug + sample combinations within that source.
    unique_source_chem_sample_combinations = (
        data[["source", "improve_chem_id", "improve_sample_id"]]
        .dropna()
        .drop_duplicates()
        .shape[0]
    )

    return {
        "unique_improve_chem_id": unique_chem_ids,
        "unique_improve_sample_id_all": unique_sample_ids,
        "unique_improve_sample_id_cell_lines": unique_cell_line_ids,
        "unique_improve_sample_id_organoids": unique_organoid_ids,
        "unique_source_chem_sample_combinations": unique_source_chem_sample_combinations,
    }


# -------------------------
# Overall counts
# -------------------------

results = []

overall_counts = calculate_counts(df)
results.append({
    "source": "OVERALL",
    **overall_counts,
})


# -------------------------
# Counts by source
# -------------------------

for source, source_df in df.groupby("source", dropna=False):
    counts = calculate_counts(source_df)

    results.append({
        "source": source,
        **counts,
    })


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print to screen
print(results_df.to_string(index=False))

# Save as TSV
results_df.to_csv(output_file, sep="\t", index=False)

print(f"\nSaved results to: {output_file}")