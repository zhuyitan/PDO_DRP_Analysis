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



############################################################
# PHASE 2 ROOTS
############################################################

CANCER_BLIND_ROOT = (
    "models/GraphDRP/Joint_Training_Results/cancer-blind"
)


# Set this to your actual random-split root
RANDOM_ROOT = (
    "models/GraphDRP/Joint_Training_Results/random"
)


SPLIT_STRATEGY_ROOTS = {
    "random": RANDOM_ROOT,
    "cancer_blind": CANCER_BLIND_ROOT,
}


############################################################
# TEST SETS
############################################################

TEST_SETS = {
    # General CCL test set
    "CCL Test": "ccl",

    # Study-specific CCL test set
    "Study CCL": "mixed",

    # Study-specific PDO test set
    "Study PDO": "study",
}


############################################################
# METRICS
############################################################

METRICS = [
    "r2",
    "pcc",
]


############################################################
# OUTPUT
############################################################

OUTPUT_ROOT = (
    "models/GraphDRP/joint_training_summary"
)




def make_run_pattern(study):

    return re.compile(
        r"^{}_i(\d+)_j(\d+)$".format(
            re.escape(study)
        )
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
    Read R2 or PCC from test_scores.json.

    Missing files, malformed JSON, missing metric values,
    and non-finite values are returned as NaN.
    """

    if not os.path.isfile(path):
        return np.nan

    try:
        payload = load_json(path)

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        return np.nan


    ########################################################
    # R2
    ########################################################

    if metric == "r2":

        if "r2" not in payload:
            return np.nan

        return finite_float(
            payload["r2"]
        )


    ########################################################
    # PCC
    ########################################################

    if metric == "pcc":

        possible_keys = [
            "pcc",
            "pearson",
            "pearson_r",
            "pearsonr",
        ]

        for key in possible_keys:

            if key in payload:

                return finite_float(
                    payload[key]
                )

        return np.nan


    raise ValueError(
        "Unknown metric: {}".format(
            metric
        )
    )


def mean_std_n(values):
    """
    Calculate mean, sample standard deviation, and
    number of available values.

    NaN values are ignored.
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

    if n == 0 or pd.isna(mean):
        return "N/A"

    if pd.isna(std):
        return "{:.3f} ± N/A".format(
            mean
        )

    return "{:.3f} ± {:.3f}".format(
        mean,
        std,
    )


############################################################
# DISCOVER PHASE 2 RUNS
############################################################

def discover_runs(
    split_root,
    study,
):
    

    if not os.path.isdir(split_root):

        print(
            "WARNING: root does not exist:"
        )

        print(
            "  {}".format(
                split_root
            )
        )

        return []


    pattern = make_run_pattern(
        study
    )

    runs = []


    for name in os.listdir(split_root):

        match = pattern.match(
            name
        )

        if not match:
            continue


        run_root = os.path.join(
            split_root,
            name,
        )

        if not os.path.isdir(run_root):
            continue


        ccl_fold = int(
            match.group(1)
        )

        pdo_fold = int(
            match.group(2)
        )


        runs.append(
            {
                "run_name": name,
                "run_root": run_root,
                "ccl_fold": ccl_fold,
                "pdo_fold": pdo_fold,
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
# LOAD ALL PHASE 2 RESULTS
############################################################

def load_all_results():
    """
    Load:

        CCL Test:
            ccl/test_scores.json

        Study CCL:
            mixed/test_scores.json

        Study PDO:
            study/test_scores.json

    for every discovered Phase 2 run.
    """

    rows = []


    for split_strategy, split_root in (
        SPLIT_STRATEGY_ROOTS.items()
    ):

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


        for study in STUDIES:

            runs = discover_runs(
                split_root,
                study,
            )


            print()
            print(
                "Study: {}".format(
                    study
                )
            )

            print(
                "Discovered Phase 2 runs: {}".format(
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

                for (
                    test_set,
                    subdirectory,
                ) in TEST_SETS.items():

                    score_path = os.path.join(
                        run["run_root"],
                        subdirectory,
                        "test_scores.json",
                    )


                    file_present = (
                        os.path.isfile(
                            score_path
                        )
                    )


                    for metric in METRICS:

                        value = read_metric(
                            score_path,
                            metric,
                        )


                        rows.append(
                            {
                                "split_strategy": (
                                    split_strategy
                                ),
                                "study": study,
                                "ccl_fold": (
                                    run["ccl_fold"]
                                ),
                                "pdo_fold": (
                                    run["pdo_fold"]
                                ),
                                "run_name": (
                                    run["run_name"]
                                ),
                                "test_set": (
                                    test_set
                                ),
                                "test_subdirectory": (
                                    subdirectory
                                ),
                                "metric": (
                                    metric
                                ),
                                "value": (
                                    value
                                ),
                                "available": (
                                    not pd.isna(
                                        value
                                    )
                                ),
                                "score_file_present": (
                                    file_present
                                ),
                                "source_path": (
                                    score_path
                                ),
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
# CREATE AVERAGED RESULTS
############################################################

def make_averaged_results(
    results_df,
):
    """
    Collapse all CCL x PDO fold results into one row per:

        split strategy
        x study
        x test set
        x metric
    """

    rows = []


    for split_strategy in SPLIT_STRATEGY_ROOTS:

        for study in STUDIES:

            for test_set in TEST_SETS:

                for metric in METRICS:

                    subset = results_df[
                        (
                            results_df[
                                "split_strategy"
                            ]
                            == split_strategy
                        )
                        & (
                            results_df[
                                "study"
                            ]
                            == study
                        )
                        & (
                            results_df[
                                "test_set"
                            ]
                            == test_set
                        )
                        & (
                            results_df[
                                "metric"
                            ]
                            == metric
                        )
                    ]


                    values = subset[
                        "value"
                    ]


                    mean, std, n_available = (
                        mean_std_n(
                            values
                        )
                    )


                    n_total = len(
                        subset
                    )

                    n_missing = (
                        n_total
                        - n_available
                    )


                    if len(subset) > 0:

                        n_score_files_present = int(
                            subset[
                                "score_file_present"
                            ].sum()
                        )

                    else:

                        n_score_files_present = 0


                    rows.append(
                        {
                            "split_strategy": (
                                split_strategy
                            ),
                            "study": (
                                study
                            ),
                            "test_set": (
                                test_set
                            ),
                            "metric": (
                                metric
                            ),
                            "mean": (
                                mean
                            ),
                            "std": (
                                std
                            ),
                            "mean_plus_minus_std": (
                                format_mean_std(
                                    mean,
                                    std,
                                    n_available,
                                )
                            ),
                            "n_available": (
                                n_available
                            ),
                            "n_total": (
                                n_total
                            ),
                            "n_missing": (
                                n_missing
                            ),
                            "n_score_files_present": (
                                n_score_files_present
                            ),
                        }
                    )


    return pd.DataFrame(
        rows
    )


############################################################
# WIDE SUMMARY TABLE
############################################################

def make_metric_summary(
    averaged_df,
    split_strategy,
    metric,
):
    """
    Produce:

        study
        CCL Test
        Study CCL
        Study PDO
    """

    rows = []


    for study in STUDIES:

        row = {
            "study": study,
        }


        for test_set in TEST_SETS:

            subset = averaged_df[
                (
                    averaged_df[
                        "split_strategy"
                    ]
                    == split_strategy
                )
                & (
                    averaged_df[
                        "study"
                    ]
                    == study
                )
                & (
                    averaged_df[
                        "test_set"
                    ]
                    == test_set
                )
                & (
                    averaged_df[
                        "metric"
                    ]
                    == metric
                )
            ]


            if subset.empty:

                row[test_set] = "N/A"

            else:

                row[test_set] = (
                    subset.iloc[0][
                        "mean_plus_minus_std"
                    ]
                )


        rows.append(
            row
        )


    return pd.DataFrame(
        rows,
        columns=[
            "study",
            "CCL Test",
            "Study CCL",
            "Study PDO",
        ],
    )


############################################################
# PRINT RESULT COUNTS
############################################################

def print_result_counts(
    averaged_df,
):

    print()
    print()
    print("=" * 100)
    print("PHASE 2 RESULT COUNTS")
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


        for study in STUDIES:

            print()
            print(
                "Study: {}".format(
                    study
                )
            )


            for test_set in TEST_SETS:

                print(
                    "  {}:".format(
                        test_set
                    )
                )


                for metric in METRICS:

                    subset = averaged_df[
                        (
                            averaged_df[
                                "split_strategy"
                            ]
                            == split_strategy
                        )
                        & (
                            averaged_df[
                                "study"
                            ]
                            == study
                        )
                        & (
                            averaged_df[
                                "test_set"
                            ]
                            == test_set
                        )
                        & (
                            averaged_df[
                                "metric"
                            ]
                            == metric
                        )
                    ]


                    if subset.empty:

                        print(
                            "    {}: 0 results".format(
                                metric.upper()
                            )
                        )

                        continue


                    result = subset.iloc[0]


                    print(
                        (
                            "    {}: "
                            "{} available / "
                            "{} total, "
                            "{} missing, "
                            "{} score files present"
                        ).format(
                            metric.upper(),
                            int(
                                result[
                                    "n_available"
                                ]
                            ),
                            int(
                                result[
                                    "n_total"
                                ]
                            ),
                            int(
                                result[
                                    "n_missing"
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
# SAVE ONE WIDE SUMMARY
############################################################

def save_metric_summary(
    averaged_df,
    split_strategy,
    metric,
    filename,
):

    summary_df = make_metric_summary(
        averaged_df,
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
    # Load individual Phase 2 results
    ########################################################

    results_df = (
        load_all_results()
    )


    ########################################################
    # Aggregate across CCL x PDO folds
    ########################################################

    averaged_df = (
        make_averaged_results(
            results_df
        )
    )


    ########################################################
    # FIFTH TSV
    #
    # Averaged data, not fold-level data
    ########################################################

    combined_path = os.path.join(
        OUTPUT_ROOT,
        "all_phase2_results_averaged.tsv",
    )


    averaged_df.to_csv(
        combined_path,
        sep="\t",
        index=False,
    )


    print()
    print("=" * 100)
    print(
        "Saved combined averaged results:"
    )
    print(
        combined_path
    )


    ########################################################
    # Print availability counts
    ########################################################

    print_result_counts(
        averaged_df
    )


    ########################################################
    # Random R2
    ########################################################

    save_metric_summary(
        averaged_df=averaged_df,
        split_strategy="random",
        metric="r2",
        filename=(
            "random_r2_summary.tsv"
        ),
    )


    ########################################################
    # Random PCC
    ########################################################

    save_metric_summary(
        averaged_df=averaged_df,
        split_strategy="random",
        metric="pcc",
        filename=(
            "random_pcc_summary.tsv"
        ),
    )


    ########################################################
    # Cancer-Blind R2
    ########################################################

    save_metric_summary(
        averaged_df=averaged_df,
        split_strategy="cancer_blind",
        metric="r2",
        filename=(
            "cancer_blind_r2_summary.tsv"
        ),
    )


    ########################################################
    # Cancer-Blind PCC
    ########################################################

    save_metric_summary(
        averaged_df=averaged_df,
        split_strategy="cancer_blind",
        metric="pcc",
        filename=(
            "cancer_blind_pcc_summary.tsv"
        ),
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
        "  random_r2_summary.tsv"
    )

    print(
        "  random_pcc_summary.tsv"
    )

    print(
        "  cancer_blind_r2_summary.tsv"
    )

    print(
        "  cancer_blind_pcc_summary.tsv"
    )

    print(
        "  all_phase2_results_averaged.tsv"
    )


if __name__ == "__main__":
    main()