import json
import os
import re

import numpy as np
import pandas as pd


############################################################
# CONFIG
############################################################


STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1"
]



SPLIT_STRATEGY_ROOTS = {
    "random": (
        "models/LGBM/transfer_learning_results/random"
    ),
    "cancer_blind": (
        "models/LGBM/transfer_learning_results/cancer-blind"
    ),
}


TEST_SETS = {
    # "Study CCL" = mixed test set
    "Study CCL": "mixed",

    # "Study PDO" = study test set
    "Study PDO": "study",
}


METRICS = [
    "r2",
    "pcc",
]


OUTPUT_ROOT = (
    "models/LGBM/summarized_transfer_learning_results"
)


RUN_PATTERN = re.compile(
    r"^ccl(\d+)_j(\d+)$"
)


############################################################
# HELPERS
############################################################

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def read_metric(path, metric):
    """
    Read a metric from test_scores.json.

    Missing files, invalid JSON, missing metric keys, and
    non-finite values are returned as NaN.
    """

    if not os.path.isfile(path):
        return np.nan

    try:
        payload = load_json(path)

        if metric not in payload:
            return np.nan

        return finite_float(payload[metric])

    except (
        TypeError,
        json.JSONDecodeError,
        OSError,
    ):
        return np.nan


def mean_std_n(values):
    """
    Calculate mean, sample standard deviation, and N
    from all available finite values.
    """

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    n = len(values)

    if n == 0:
        return np.nan, np.nan, 0

    mean = values.mean()

    if n > 1:
        std = values.std(ddof=1)
    else:
        std = np.nan

    return mean, std, n


def format_mean_std(mean, std, n):
    """
    Format summary TSV value as:

        mean ± std

    If only one result exists, SD cannot be calculated.
    If no results exist, output N/A.
    """

    if n == 0 or pd.isna(mean):
        return "N/A"

    if pd.isna(std):
        return "{:.3f} ± N/A".format(mean)

    return "{:.3f} ± {:.3f}".format(
        mean,
        std,
    )


############################################################
# RUN DISCOVERY
############################################################

def discover_runs(split_root, study):
    """
    Discover Phase 3 run directories of the form:

        ccl{i}_j{j}

    Returns:
        list of dictionaries:
            {
                "ccl_fold": i,
                "pdo_fold": j,
                "run_name": ...,
                "run_root": ...
            }
    """

    study_root = os.path.join(
        split_root,
        study,
    )

    if not os.path.isdir(study_root):
        print(
            "WARNING: study directory not found: {}".format(
                study_root
            )
        )
        return []

    runs = []

    for name in sorted(os.listdir(study_root)):
        match = RUN_PATTERN.match(name)

        if not match:
            continue

        run_root = os.path.join(
            study_root,
            name,
        )

        if not os.path.isdir(run_root):
            continue

        ccl_fold = int(match.group(1))
        pdo_fold = int(match.group(2))

        runs.append(
            {
                "ccl_fold": ccl_fold,
                "pdo_fold": pdo_fold,
                "run_name": name,
                "run_root": run_root,
            }
        )

    runs = sorted(
        runs,
        key=lambda x: (
            x["pdo_fold"],
            x["ccl_fold"],
        ),
    )

    return runs


############################################################
# LOAD ALL RESULTS
############################################################

def load_all_results():
    """
    Build one long-form dataframe containing every discovered
    Phase 3 run.

    One row is generated for each:

        split strategy
        x study
        x CCL fold
        x PDO fold
        x test set
        x metric

    Test sets:
        Study CCL -> mixed/test_scores.json
        Study PDO -> study/test_scores.json

    Metrics:
        r2
        pcc
    """

    rows = []

    for split_strategy, split_root in SPLIT_STRATEGY_ROOTS.items():

        print()
        print("=" * 100)
        print(
            "SPLIT STRATEGY: {}".format(
                split_strategy
            )
        )
        print(
            "ROOT: {}".format(
                split_root
            )
        )
        print("=" * 100)

        if not os.path.isdir(split_root):
            print(
                "WARNING: split strategy root does not exist:"
            )
            print(split_root)

        for study in STUDIES:

            print()
            print(
                "Study: {}".format(
                    study
                )
            )

            runs = discover_runs(
                split_root,
                study,
            )

            print(
                "Discovered run directories: {}".format(
                    len(runs)
                )
            )

            if runs:
                ccl_folds = sorted(
                    {
                        run["ccl_fold"]
                        for run in runs
                    }
                )

                pdo_folds = sorted(
                    {
                        run["pdo_fold"]
                        for run in runs
                    }
                )

                print(
                    "  CCL folds: {}".format(
                        ccl_folds
                    )
                )

                print(
                    "  PDO folds: {}".format(
                        pdo_folds
                    )
                )

            for run in runs:

                ccl_fold = run["ccl_fold"]
                pdo_fold = run["pdo_fold"]
                run_root = run["run_root"]
                run_name = run["run_name"]

                for test_set, subdirectory in TEST_SETS.items():

                    score_path = os.path.join(
                        run_root,
                        subdirectory,
                        "test_scores.json",
                    )

                    file_present = os.path.isfile(
                        score_path
                    )

                    for metric in METRICS:

                        value = read_metric(
                            score_path,
                            metric,
                        )

                        rows.append(
                            {
                                "split_strategy": split_strategy,
                                "study": study,
                                "ccl_fold": ccl_fold,
                                "pdo_fold": pdo_fold,
                                "run_name": run_name,
                                "test_set": test_set,
                                "test_subdirectory": subdirectory,
                                "metric": metric,
                                "value": value,
                                "available": (
                                    not pd.isna(value)
                                ),
                                "score_file_present": file_present,
                                "source_path": score_path,
                            }
                        )

    columns = [
        "split_strategy",
        "study",
        "ccl_fold",
        "pdo_fold",
        "run_name",
        "test_set",
        "test_subdirectory",
        "metric",
        "value",
        "available",
        "score_file_present",
        "source_path",
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


############################################################
# SUMMARY COUNTS
############################################################

def make_count_summary(results_df):
    """
    Count valid metric values for each:

        split strategy
        x study
        x test set
        x metric

    Also reports:
        - number of discovered runs
        - score files present
        - metric values available
        - metric values missing
    """

    rows = []

    for split_strategy in SPLIT_STRATEGY_ROOTS:

        for study in STUDIES:

            for test_set in TEST_SETS:

                for metric in METRICS:

                    subset = results_df[
                        (
                            results_df["split_strategy"]
                            == split_strategy
                        )
                        & (
                            results_df["study"]
                            == study
                        )
                        & (
                            results_df["test_set"]
                            == test_set
                        )
                        & (
                            results_df["metric"]
                            == metric
                        )
                    ]

                    n_runs = len(subset)

                    if n_runs == 0:
                        n_files_present = 0
                        n_available = 0
                    else:
                        n_files_present = int(
                            subset[
                                "score_file_present"
                            ].sum()
                        )

                        n_available = int(
                            subset[
                                "available"
                            ].sum()
                        )

                    n_missing = (
                        n_runs - n_available
                    )

                    rows.append(
                        {
                            "split_strategy": split_strategy,
                            "study": study,
                            "test_set": test_set,
                            "metric": metric,
                            "n_discovered_runs": n_runs,
                            "n_score_files_present": (
                                n_files_present
                            ),
                            "n_metric_available": (
                                n_available
                            ),
                            "n_metric_missing": (
                                n_missing
                            ),
                        }
                    )

    return pd.DataFrame(rows)


############################################################
# STUDY-LEVEL SUMMARY
############################################################

def make_metric_summary(
    results_df,
    split_strategy,
    metric,
):
    """
    Create summary table:

        study    Study CCL    Study PDO

    Each value is:

        mean ± sample_SD
    """

    rows = []

    for study in STUDIES:

        row = {
            "study": study,
        }

        for test_set in TEST_SETS:

            subset = results_df[
                (
                    results_df["split_strategy"]
                    == split_strategy
                )
                & (
                    results_df["study"]
                    == study
                )
                & (
                    results_df["test_set"]
                    == test_set
                )
                & (
                    results_df["metric"]
                    == metric
                )
            ]

            mean, std, n = mean_std_n(
                subset["value"]
            )

            row[test_set] = format_mean_std(
                mean,
                std,
                n,
            )

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df[
        [
            "study",
            "Study CCL",
            "Study PDO",
        ]
    ]

    return summary_df


############################################################
# PRINT COUNTS
############################################################

def print_count_summary(count_df):

    print()
    print()
    print("=" * 100)
    print("RESULT AVAILABILITY COUNTS")
    print("=" * 100)

    for split_strategy in SPLIT_STRATEGY_ROOTS:

        print()
        print("#" * 100)
        print(
            "SPLIT STRATEGY: {}".format(
                split_strategy
            )
        )
        print("#" * 100)

        strategy_df = count_df[
            count_df["split_strategy"]
            == split_strategy
        ]

        for study in STUDIES:

            print()
            print(
                "Study: {}".format(
                    study
                )
            )

            study_df = strategy_df[
                strategy_df["study"]
                == study
            ]

            for test_set in TEST_SETS:

                test_df = study_df[
                    study_df["test_set"]
                    == test_set
                ]

                print(
                    "  {}:".format(
                        test_set
                    )
                )

                for metric in METRICS:

                    metric_df = test_df[
                        test_df["metric"]
                        == metric
                    ]

                    if metric_df.empty:
                        print(
                            "    {}: 0 results".format(
                                metric.upper()
                            )
                        )
                        continue

                    result = metric_df.iloc[0]

                    print(
                        (
                            "    {}: {} valid / {} discovered "
                            "runs, {} score files present"
                        ).format(
                            metric.upper(),
                            int(
                                result[
                                    "n_metric_available"
                                ]
                            ),
                            int(
                                result[
                                    "n_discovered_runs"
                                ]
                            ),
                            int(
                                result[
                                    "n_score_files_present"
                                ]
                            ),
                        )
                    )


############################################################
# SAVE SUMMARY
############################################################

def save_metric_summary(
    results_df,
    split_strategy,
    metric,
    filename,
):

    summary_df = make_metric_summary(
        results_df,
        split_strategy,
        metric,
    )

    path = os.path.join(
        OUTPUT_ROOT,
        filename,
    )

    summary_df.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print()
    print(
        "Saved: {}".format(
            path
        )
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


############################################################
# MAIN
############################################################

def main():

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    ########################################################
    # Load all Phase 3 results
    ########################################################

    results_df = load_all_results()

    ########################################################
    # Save complete long-form results TSV
    ########################################################

    all_results_path = os.path.join(
        OUTPUT_ROOT,
        "all_phase3_results.tsv",
    )

    results_df.to_csv(
        all_results_path,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "Saved all results: {}".format(
            all_results_path
        )
    )

    ########################################################
    # Count result availability
    ########################################################

    count_df = make_count_summary(
        results_df
    )

    print_count_summary(
        count_df
    )

    ########################################################
    # Cancer-blind R2
    ########################################################

    save_metric_summary(
        results_df=results_df,
        split_strategy="cancer_blind",
        metric="r2",
        filename="cancer_blind_r2_summary.tsv",
    )

    ########################################################
    # Cancer-blind PCC
    ########################################################

    save_metric_summary(
        results_df=results_df,
        split_strategy="cancer_blind",
        metric="pcc",
        filename="cancer_blind_pcc_summary.tsv",
    )

    ########################################################
    # Random R2
    ########################################################

    save_metric_summary(
        results_df=results_df,
        split_strategy="random",
        metric="r2",
        filename="random_r2_summary.tsv",
    )

    ########################################################
    # Random PCC
    ########################################################

    save_metric_summary(
        results_df=results_df,
        split_strategy="random",
        metric="pcc",
        filename="random_pcc_summary.tsv",
    )

    ########################################################
    # Finish
    ########################################################

    print()
    print()
    print("=" * 100)
    print("FINISHED")
    print("=" * 100)
    print(
        "Output directory: {}".format(
            OUTPUT_ROOT
        )
    )

    print()
    print("Generated files:")

    print(
        "  cancer_blind_r2_summary.tsv"
    )
    print(
        "  cancer_blind_pcc_summary.tsv"
    )
    print(
        "  random_r2_summary.tsv"
    )
    print(
        "  random_pcc_summary.tsv"
    )
    print(
        "  all_phase3_results.tsv"
    )


if __name__ == "__main__":
    main()