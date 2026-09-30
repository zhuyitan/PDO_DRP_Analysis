import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import time
import json
import math

from scipy.optimize import differential_evolution


#############################################
# CONFIG
#############################################

PILOT_MODE = False

STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1"
]

PILOT_STUDIES = {
    "Pilot_1"
}

BASE_SPLIT_DIR = (
    "pdo_drp_data/splits/random"
)

RAW_DATA_DIR = (
    "pdo_drp_data"
)

PRETRAIN_DIR = (
    "LGBM/pretrained_models"
)

# Expected filenames inside each:
#
#   ccl_fold_0/
#       model.txt
#       x_data_gene_expression_scaler.gz
#       x_data_mordred_scaler.gz
#
#   ccl_fold_1/
#       ...
#
PRETRAIN_GE_SCALER_FNAME = (
    "x_data_gene_expression_scaler.gz"
)

PRETRAIN_MD_SCALER_FNAME = (
    "x_data_mordred_scaler.gz"
)

OUTPUT_DIR = (
    "LGBM_Transfer_Learning_Results"
)

MAX_PARALLEL_JOBS = 4

LGBM_DIR = "models/LGBM"

PREPROCESS_SCRIPT = os.path.join(
    LGBM_DIR,
    "lgbm_preprocess_improve.py"
)

TRAIN_SCRIPT = os.path.join(
    LGBM_DIR,
    "lgbm_train_improve.py"
)

INFER_SCRIPT = os.path.join(
    LGBM_DIR,
    "lgbm_infer_improve.py"
)


#############################################
# DIFFERENTIAL EVOLUTION CONFIG
#############################################

MIN_DATA_IN_LEAF_BOUNDS = (
    2,
    64,
)

LOG10_LEARNING_RATE_BOUNDS = (
    math.log10(0.001),
    math.log10(0.05),
)

HPO_METRIC = "mse"

DE_MAXITER = 5
DE_POPSIZE = 5
DE_TOL = 0.01
DE_SEED = 42


#############################################
# UTIL
#############################################

def run_cmd(cmd, log_file=None):
    """
    Run a subprocess command.

    If log_file is supplied, stdout/stderr are written to that file.
    """

    if log_file is None:
        subprocess.run(
            cmd,
            check=True,
        )
        return

    with open(log_file, "w") as log_handle:
        subprocess.run(
            cmd,
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )


def read_validation_score(score_file):
    """
    Read the validation metric used for HPO.
    """

    with open(score_file, "r") as f:
        scores = json.load(f)

    if HPO_METRIC not in scores:
        raise KeyError(
            f"Metric '{HPO_METRIC}' not found in "
            f"{score_file}. Available metrics: "
            f"{list(scores.keys())}"
        )

    score = float(
        scores[HPO_METRIC]
    )

    if not math.isfinite(score):
        raise ValueError(
            f"Non-finite {HPO_METRIC} in "
            f"{score_file}: {score}"
        )

    return score


def append_hpo_history(
    history_file,
    record,
):
    """
    Append one HPO trial result to the JSONL history.
    """

    with open(history_file, "a") as f:
        json.dump(
            record,
            f,
        )

        f.write("\n")


def run_preprocessing(
    train_split,
    val_split,
    test_split,
    output_dir,
    pretrained_ge_scaler,
    pretrained_md_scaler,
):
    """
    Run preprocessing using the scalers belonging to the
    corresponding pretrained CCL fold.

    IMPORTANT:
    test_split controls which observations become test_data.parquet.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    run_cmd([
        "python",
        PREPROCESS_SCRIPT,

        "--input_dir",
        RAW_DATA_DIR,

        "--output_dir",
        output_dir,

        "--train_split_file",
        train_split,

        "--val_split_file",
        val_split,

        "--test_split_file",
        test_split,

        "--pretrained_ge_scaler",
        pretrained_ge_scaler,

        "--pretrained_md_scaler",
        pretrained_md_scaler,
    ])


#############################################
# HYPERPARAMETER OPTIMIZATION
#############################################

def optimize_hyperparameters(
    run_dir,
    pretrained_model,
):
    """
    Optimize fine-tuning hyperparameters using validation MSE.

    The preprocessed train/validation data are already in run_dir
    and have already been transformed using the matching pretrained
    CCL scalers.
    """

    best_params_file = os.path.join(
        run_dir,
        "best_hyperparameters.json",
    )

    # -----------------------------------------
    # Resume existing HPO
    # -----------------------------------------

    if os.path.exists(
        best_params_file
    ):

        with open(
            best_params_file,
            "r",
        ) as f:
            saved = json.load(f)

        print(
            f"[HPO RESUME] Using saved parameters: "
            f"min_data_in_leaf="
            f"{saved['min_data_in_leaf']}, "
            f"learning_rate="
            f"{saved['learning_rate']}",
            flush=True,
        )

        return saved

    # -----------------------------------------
    # HPO directories
    # -----------------------------------------

    hpo_dir = os.path.join(
        run_dir,
        "hpo_trials",
    )

    history_file = os.path.join(
        run_dir,
        "hpo_history.jsonl",
    )

    if os.path.exists(
        hpo_dir
    ):
        shutil.rmtree(
            hpo_dir
        )

    if os.path.exists(
        history_file
    ):
        os.remove(
            history_file
        )

    os.makedirs(
        hpo_dir,
        exist_ok=True,
    )

    evaluation_number = 0

    # -----------------------------------------
    # Objective
    # -----------------------------------------

    def objective(x):

        nonlocal evaluation_number

        evaluation_number += 1

        min_data_in_leaf = int(
            round(
                x[0]
            )
        )

        learning_rate = (
            10 ** float(x[1])
        )

        trial_dir = os.path.join(
            hpo_dir,
            f"eval_{evaluation_number:04d}",
        )

        os.makedirs(
            trial_dir,
            exist_ok=True,
        )

        print(
            f"[HPO] "
            f"eval={evaluation_number} "
            f"min_data_in_leaf={min_data_in_leaf} "
            f"learning_rate={learning_rate:.12g} "
            f"log10_learning_rate={float(x[1]):.6f}",
            flush=True,
        )

        try:

            run_cmd(
                [
                    "python",
                    TRAIN_SCRIPT,

                    "--input_dir",
                    run_dir,

                    "--output_dir",
                    trial_dir,

                    "--pretrained_model",
                    pretrained_model,

                    "--transfer_rounds",
                    "10000",

                    "--min_data_in_leaf",
                    str(
                        min_data_in_leaf
                    ),

                    "--learning_rate",
                    f"{learning_rate:.12g}",
                ],
                log_file=os.path.join(
                    trial_dir,
                    "subprocess.log",
                ),
            )

            score_file = os.path.join(
                trial_dir,
                "val_scores.json",
            )

            score = read_validation_score(
                score_file
            )

            append_hpo_history(
                history_file,
                {
                    "evaluation": (
                        evaluation_number
                    ),

                    "min_data_in_leaf": (
                        min_data_in_leaf
                    ),

                    "learning_rate": (
                        learning_rate
                    ),

                    HPO_METRIC: (
                        score
                    ),

                    "status": (
                        "ok"
                    ),
                },
            )

            print(
                f"[HPO RESULT] "
                f"eval={evaluation_number} "
                f"{HPO_METRIC}={score}",
                flush=True,
            )

            return score

        except Exception as e:

            append_hpo_history(
                history_file,
                {
                    "evaluation": (
                        evaluation_number
                    ),

                    "min_data_in_leaf": (
                        min_data_in_leaf
                    ),

                    "learning_rate": (
                        learning_rate
                    ),

                    HPO_METRIC: (
                        None
                    ),

                    "status": (
                        "error"
                    ),

                    "error": (
                        str(e)
                    ),
                },
            )

            print(
                f"[HPO ERROR] "
                f"eval={evaluation_number}: "
                f"{e}",
                flush=True,
            )

            return float(
                "inf"
            )

        finally:

            shutil.rmtree(
                trial_dir,
                ignore_errors=True,
            )

    # -----------------------------------------
    # Differential evolution
    # -----------------------------------------

    try:

        result = differential_evolution(
            objective,
            bounds=[
                MIN_DATA_IN_LEAF_BOUNDS,
                LOG10_LEARNING_RATE_BOUNDS,
            ],
            strategy="best1bin",
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            polish=False,
            init="latinhypercube",
            workers=1,
            seed=DE_SEED,
            disp=False,
        )

    finally:

        shutil.rmtree(
            hpo_dir,
            ignore_errors=True,
        )

    if not math.isfinite(
        result.fun
    ):
        raise RuntimeError(
            "Differential evolution did not "
            "produce a finite validation score."
        )

    # -----------------------------------------
    # Save best parameters
    # -----------------------------------------

    best_params = {
        "min_data_in_leaf": int(
            round(
                result.x[0]
            )
        ),

        "learning_rate": (
            10 ** float(
                result.x[1]
            )
        ),

        "log10_learning_rate": float(
            result.x[1]
        ),

        HPO_METRIC: float(
            result.fun
        ),

        "nfev": int(
            result.nfev
        ),

        "nit": int(
            result.nit
        ),

        "success": bool(
            result.success
        ),

        "message": str(
            result.message
        ),

        "search_space": {
            "min_data_in_leaf": list(
                MIN_DATA_IN_LEAF_BOUNDS
            ),

            "log10_learning_rate": list(
                LOG10_LEARNING_RATE_BOUNDS
            ),

            "learning_rate": [
                10 ** LOG10_LEARNING_RATE_BOUNDS[0],
                10 ** LOG10_LEARNING_RATE_BOUNDS[1],
            ],
        },

        "de_settings": {
            "maxiter": (
                DE_MAXITER
            ),

            "popsize": (
                DE_POPSIZE
            ),

            "tol": (
                DE_TOL
            ),

            "seed": (
                DE_SEED
            ),

            "polish": (
                False
            ),

            "workers": (
                1
            ),
        },
    }

    with open(
        best_params_file,
        "w",
    ) as f:
        json.dump(
            best_params,
            f,
            indent=2,
        )

    print(
        f"[HPO BEST] "
        f"min_data_in_leaf="
        f"{best_params['min_data_in_leaf']} "
        f"learning_rate="
        f"{best_params['learning_rate']:.12g} "
        f"{HPO_METRIC}="
        f"{best_params[HPO_METRIC]}",
        flush=True,
    )

    return best_params


#############################################
# COMPLETION CHECK
#############################################

def job_complete(
    run_dir,
    study,
):
    """
    Determine whether a transfer-learning job has completed all
    required training and evaluation outputs.
    """

    required = [
        "model.txt",
        "best_hyperparameters.json",
        "timing.json",

        # Study test set
        "study/test_scores.json",
        "study/test_y_data_predicted.csv",

        # Fine-tuning training set
        "train/test_scores.json",
        "train/test_y_data_predicted.csv",
    ]

    # Pilot studies do not use the mixed CCL test set.
    if study not in PILOT_STUDIES:

        required.extend([
            "mixed/test_scores.json",
            "mixed/test_y_data_predicted.csv",
        ])

    return all(
        os.path.exists(
            os.path.join(
                run_dir,
                filename,
            )
        )
        for filename in required
    )


#############################################
# SINGLE JOB
#############################################

def run_transfer_job(
    study,
    i,
    j,
):
    """
    Fine-tune CCL pretrained fold i on PDO study fold j.

    i:
        Pretrained CCL fold, 0-9.

    j:
        Fine-tuning/PDO split, 0-9.

    Each CCL fold provides:
        - pretrained LightGBM model
        - pretrained gene-expression scaler
        - pretrained Mordred scaler
    """

    start_time = time.time()

    timing_file = None

    try:

        #####################################
        # RUN DIRECTORY
        #####################################

        run_dir = (
            f"{OUTPUT_DIR}/"
            f"{study}/"
            f"ccl{i}_j{j}"
        )

        os.makedirs(
            run_dir,
            exist_ok=True,
        )

        timing_file = os.path.join(
            run_dir,
            "timing.json",
        )

        # ----------------------------------
        # Skip fully finished jobs
        # ----------------------------------

        if job_complete(
            run_dir,
            study,
        ):

            print(
                f"[SKIP DONE] "
                f"{study} "
                f"i={i} "
                f"j={j}",
                flush=True,
            )

            return

        print(
            f"[START] "
            f"{study} "
            f"i={i} "
            f"j={j}",
            flush=True,
        )


        #####################################
        # PDO SPLITS
        #####################################

        train_split = (
            f"{BASE_SPLIT_DIR}/"
            f"{study}_split_{j}_train_ids.txt"
        )

        val_split = (
            f"{BASE_SPLIT_DIR}/"
            f"{study}_split_{j}_val_ids.txt"
        )

        test_split = (
            f"{BASE_SPLIT_DIR}/"
            f"{study}_split_{j}_test_ids.txt"
        )


        #####################################
        # PRETRAINED CCL FOLD ASSETS
        #####################################

        pretrained_fold_dir = os.path.join(
            PRETRAIN_DIR,
            f"ccl_fold_{i}",
        )

        pretrained_model = os.path.join(
            pretrained_fold_dir,
            "model.txt",
        )

        pretrained_ge_scaler = os.path.join(
            pretrained_fold_dir,
            PRETRAIN_GE_SCALER_FNAME,
        )

        pretrained_md_scaler = os.path.join(
            pretrained_fold_dir,
            PRETRAIN_MD_SCALER_FNAME,
        )

        pretrained_files = [
            pretrained_model,
            pretrained_ge_scaler,
            pretrained_md_scaler,
        ]

        missing_pretrained_files = [
            path
            for path in pretrained_files
            if not os.path.exists(path)
        ]

        if missing_pretrained_files:

            print(
                f"[SKIP] Missing pretrained files "
                f"for CCL fold {i}: "
                f"{missing_pretrained_files}",
                flush=True,
            )

            return

        print(
            f"[PRETRAINED] "
            f"study={study} "
            f"i={i} "
            f"j={j}",
            flush=True,
        )

        print(
            f"  model: "
            f"{pretrained_model}",
            flush=True,
        )

        print(
            f"  GE scaler: "
            f"{pretrained_ge_scaler}",
            flush=True,
        )

        print(
            f"  MD scaler: "
            f"{pretrained_md_scaler}",
            flush=True,
        )


        #####################################
        # VERIFY PDO SPLITS
        #####################################

        missing_split_files = [
            path
            for path in [
                train_split,
                val_split,
                test_split,
            ]
            if not os.path.exists(path)
        ]

        if missing_split_files:

            print(
                f"[SKIP] Missing split files "
                f"for {study} j={j}: "
                f"{missing_split_files}",
                flush=True,
            )

            return


        #####################################
        # PREPROCESS FINE-TUNING DATA
        #####################################
        #
        # This creates the train/val/test data
        # used by:
        #
        #   1. HPO
        #   2. final transfer training
        #   3. study test inference
        #
        # All features are transformed using
        # the matching pretrained CCL fold
        # scalers.
        #
        #####################################

        print(
            f"[PREPROCESS TRAIN/VAL/STUDY] "
            f"{study} "
            f"i={i} "
            f"j={j}",
            flush=True,
        )

        run_preprocessing(
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            output_dir=run_dir,
            pretrained_ge_scaler=(
                pretrained_ge_scaler
            ),
            pretrained_md_scaler=(
                pretrained_md_scaler
            ),
        )


        #####################################
        # HYPERPARAMETER OPTIMIZATION
        #####################################

        best_params = (
            optimize_hyperparameters(
                run_dir=run_dir,
                pretrained_model=(
                    pretrained_model
                ),
            )
        )


        #####################################
        # TRANSFER TRAIN
        #####################################

        print(
            f"[TRANSFER TRAIN] "
            f"{study} "
            f"i={i} "
            f"j={j}",
            flush=True,
        )

        run_cmd([
            "python",
            TRAIN_SCRIPT,

            "--input_dir",
            run_dir,

            "--output_dir",
            run_dir,

            "--pretrained_model",
            pretrained_model,

            "--transfer_rounds",
            "10000",

            "--min_data_in_leaf",
            str(
                best_params[
                    "min_data_in_leaf"
                ]
            ),

            "--learning_rate",
            (
                f"{best_params['learning_rate']:.12g}"
            ),
        ])


        #####################################
        # STUDY TEST SET INFERENCE
        #####################################
        #
        # run_dir already contains:
        #
        #   test_data.parquet
        #
        # corresponding to the real held-out
        # PDO study test split.
        #
        # No second preprocessing pass is
        # necessary.
        #
        #####################################

        study_dir = os.path.join(
            run_dir,
            "study",
        )

        os.makedirs(
            study_dir,
            exist_ok=True,
        )

        print(
            f"[STUDY INFER] "
            f"{study} "
            f"i={i} "
            f"j={j}",
            flush=True,
        )

        run_cmd([
            "python",
            INFER_SCRIPT,

            "--input_data_dir",
            run_dir,

            "--input_model_dir",
            run_dir,

            "--output_dir",
            study_dir,

            "--calc_infer_scores",
            "true",
        ])


        #####################################
        # MIXED TEST SET INFERENCE
        #####################################
        #
        # This gets its own preprocessing
        # directory so test_data.parquet in
        # run_dir is never overwritten.
        #
        #####################################

        if study not in PILOT_STUDIES:

            mixed_split = (
                f"{BASE_SPLIT_DIR}/"
                f"{study}_ccl_ids.txt"
            )

            if not os.path.exists(
                mixed_split
            ):

                raise FileNotFoundError(
                    f"Missing mixed split: "
                    f"{mixed_split}"
                )

            mixed_dir = os.path.join(
                run_dir,
                "mixed",
            )

            mixed_eval_data_dir = (
                os.path.join(
                    run_dir,
                    "mixed_eval_data",
                )
            )

            os.makedirs(
                mixed_dir,
                exist_ok=True,
            )

            os.makedirs(
                mixed_eval_data_dir,
                exist_ok=True,
            )

            print(
                f"[MIXED PREPROCESS] "
                f"{study} "
                f"i={i} "
                f"j={j}",
                flush=True,
            )

            run_preprocessing(
                train_split=train_split,
                val_split=val_split,
                test_split=mixed_split,
                output_dir=(
                    mixed_eval_data_dir
                ),
                pretrained_ge_scaler=(
                    pretrained_ge_scaler
                ),
                pretrained_md_scaler=(
                    pretrained_md_scaler
                ),
            )

            print(
                f"[MIXED INFER] "
                f"{study} "
                f"i={i} "
                f"j={j}",
                flush=True,
            )

            run_cmd([
                "python",
                INFER_SCRIPT,

                "--input_data_dir",
                mixed_eval_data_dir,

                "--input_model_dir",
                run_dir,

                "--output_dir",
                mixed_dir,

                "--calc_infer_scores",
                "true",
            ])


        #####################################
        # TRAINING SET INFERENCE
        #####################################
        #
        # We intentionally reuse:
        #
        #   train_split -> train
        #   val_split   -> val
        #   train_split -> test
        #
        # The existing inference script always
        # predicts "test", so making test equal
        # to train allows us to reuse infer.py
        # unchanged.
        #
        #####################################

        train_eval_dir = os.path.join(
            run_dir,
            "train",
        )

        train_eval_data_dir = os.path.join(
            run_dir,
            "train_eval_data",
        )

        os.makedirs(
            train_eval_dir,
            exist_ok=True,
        )

        os.makedirs(
            train_eval_data_dir,
            exist_ok=True,
        )

        print(
            f"[TRAIN PREPROCESS] "
            f"{study} "
            f"i={i} "
            f"j={j}",
            flush=True,
        )

        run_preprocessing(
            train_split=train_split,
            val_split=val_split,

            # Deliberately reuse training IDs
            # as the test IDs.
            test_split=train_split,

            output_dir=(
                train_eval_data_dir
            ),

            pretrained_ge_scaler=(
                pretrained_ge_scaler
            ),

            pretrained_md_scaler=(
                pretrained_md_scaler
            ),
        )

        print(
            f"[TRAIN INFER] "
            f"{study} "
            f"i={i} "
            f"j={j}",
            flush=True,
        )

        run_cmd([
            "python",
            INFER_SCRIPT,

            "--input_data_dir",
            train_eval_data_dir,

            "--input_model_dir",
            run_dir,

            "--output_dir",
            train_eval_dir,

            "--calc_infer_scores",
            "true",
        ])


        #####################################
        # TIMING
        #####################################

        elapsed = (
            time.time()
            - start_time
        )

        with open(
            timing_file,
            "w",
        ) as f:

            json.dump(
                {
                    "study": (
                        study
                    ),

                    "ccl_fold": (
                        i
                    ),

                    "pdo_fold": (
                        j
                    ),

                    "elapsed_seconds": (
                        elapsed
                    ),

                    "elapsed_minutes": (
                        elapsed / 60
                    ),

                    "pilot_mode": (
                        PILOT_MODE
                    ),

                    "pretrained_model": (
                        pretrained_model
                    ),

                    "pretrained_ge_scaler": (
                        pretrained_ge_scaler
                    ),

                    "pretrained_md_scaler": (
                        pretrained_md_scaler
                    ),

                    "train_split": (
                        train_split
                    ),

                    "val_split": (
                        val_split
                    ),

                    "test_split": (
                        test_split
                    ),
                },
                f,
                indent=2,
            )

        print(
            f"[DONE] "
            f"{study} "
            f"i={i} "
            f"j={j} "
            f"({elapsed / 60:.2f} min)",
            flush=True,
        )


    except Exception as e:

        print(
            f"[ERROR] "
            f"{study} "
            f"i={i} "
            f"j={j}: "
            f"{e}",
            flush=True,
        )

        raise


#############################################
# MAIN
#############################################

def main():

    jobs = []

    #########################################
    # BUILD JOB LIST
    #########################################

    if PILOT_MODE:

        for study in STUDIES:

            jobs.append(
                (
                    study,
                    0,
                    0,
                )
            )

    else:

        for study in STUDIES:

            for i in range(10):

                for j in range(10):

                    jobs.append(
                        (
                            study,
                            i,
                            j,
                        )
                    )

    print(
        f"Total jobs: "
        f"{len(jobs)}",
        flush=True,
    )


    #########################################
    # RUN JOBS
    #########################################

    with ProcessPoolExecutor(
        max_workers=MAX_PARALLEL_JOBS
    ) as executor:

        futures = [
            executor.submit(
                run_transfer_job,
                study,
                i,
                j,
            )
            for study, i, j in jobs
        ]

        for future in as_completed(
            futures
        ):
            future.result()


if __name__ == "__main__":
    main()