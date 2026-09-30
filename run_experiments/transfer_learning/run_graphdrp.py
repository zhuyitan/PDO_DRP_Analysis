import os
import sys
import json
import math
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Manager

from scipy.optimize import differential_evolution

############################################################
# CONFIG
############################################################

# Run all CCL folds i = 0..9.
CCL_FOLDS = list(range(10))

STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1"
]


PILOT_STUDIES = {
    "Pilot_1"
}

RAW_DATA_DIR = (
    "pdo_drp_data"
)

BASE_SPLIT_DIR = (
    "pdo_drp_data/splits/random"
)

# models pretrained on cell line data
PRETRAIN_DIR = (
    "models/GraphDRP/pretrained_models"
)

PRETRAINED_SCALER_ROOT = (
    "models/GraphDRP/pretrained_scalers"
)

# IMPORTANT:
# Keep this separate from earlier pilot/single-study output directories so
# completed files from an older scaler configuration cannot be reused.
BASE_OUTPUT_DIR = (
    "GraphDRP_transfer_learning_results"
)

GRAPHDRP_DIR = "models/GraphDRP"

TRAIN_SCRIPT = os.path.join(
    GRAPHDRP_DIR,
    "graphdrp_train_improve.py"
)

PREPROCESS_SCRIPT = os.path.join(
    GRAPHDRP_DIR,
    "graphdrp_preprocess_improve.py"
)

INFER_SCRIPT = os.path.join(
    GRAPHDRP_DIR,
    "graphdrp_infer_improve.py"
)

CCL_PREPROCESS_SCRIPT = PREPROCESS_SCRIPT

# generate pretraining preprocessing data if not already present
CCL_PREPROCESS_SCRIPT = "graphdrp_preprocess_improve.py"
CCL_SPLIT_DIR = os.path.join(
    RAW_DATA_DIR,
    "splits",
    "random",
)
FORCE_CCL_PREPROCESS = False

MODEL_FILE_NAME = "model.pt"
TRAIN_EPOCHS = 200

############################################################
# GPU CONFIG
############################################################

# GPUs 0-5: two concurrent transfer jobs/models per GPU.
# GPUs 6-7: four concurrent transfer jobs/models per GPU.
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

MAX_ACTIVE_JOBS = sum(GPU_SLOTS.values())

############################################################
# TRANSFER-LEARNING CONFIG
############################################################

# This experiment isolates the effect of prediction-head adaptation.
# Both pretrained encoders remain fixed for every HPO candidate.
FIXED_FREEZE_LAYERS = "Freeze Both"

############################################################
# DIFFERENTIAL EVOLUTION CONFIG
############################################################

BATCH_SIZE_CHOICES = [
    4,
    8,
    16
]

# Five equal-width regions: [0,1), [1,2), ..., [4,5].
BATCH_INDEX_BOUNDS = (
    0.0,
    float(len(BATCH_SIZE_CHOICES)),
)

LOG10_LEARNING_RATE_BOUNDS = (-5.0, -3.0)

HEAD_FREEZE_STRATEGIES = [
    "None",
    "Freeze FC1",
    "Freeze FC1 FC2",
    "Freeze Head",
]

# A continuous interval [0, 4] is divided into four equal-width
# categorical regions by floor() in decode_params().
HEAD_FREEZE_INDEX_BOUNDS = (
    0.0,
    float(len(HEAD_FREEZE_STRATEGIES)),
)

# Prediction-head dropout is optimized as a continuous variable.
DROPOUT_BOUNDS = (0.0, 0.9)

HPO_METRIC = "mse"

DE_MAXITER = 5
DE_POPSIZE = 5
DE_TOL = 0.01
DE_SEED = 42
DE_STRATEGY = "best1bin"
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7

DELETE_HPO_TRIALS = True

############################################################
# HELPERS
############################################################


def run_cmd(cmd, log_file=None, env=None):
    """Run a subprocess, stream output to the console, and optionally log it."""

    start = time.time()

    if log_file is None:
        subprocess.run(
            cmd,
            check=True,
            env=env,
        )
        return time.time() - start

    log_parent = os.path.dirname(log_file)
    if log_parent:
        os.makedirs(log_parent, exist_ok=True)

    with open(log_file, "a") as log_handle:
        header = (
            "\n"
            + "=" * 80
            + "\n"
            + "TIME: {}\n".format(datetime.now())
            + "COMMAND: {}\n".format(" ".join(cmd))
            + "=" * 80
            + "\n"
        )

        print(header, end="", flush=True)
        log_handle.write(header)
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )

        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
        finally:
            if process.stdout is not None:
                process.stdout.close()

        return_code = process.wait()
        elapsed = time.time() - start

        footer = (
            "\nELAPSED: {:.2f} sec\n".format(elapsed)
            + "RETURN CODE: {}\n".format(return_code)
            + "=" * 80
            + "\n"
        )

        print(footer, end="", flush=True)
        log_handle.write(footer)
        log_handle.flush()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            cmd,
        )

    return elapsed


def read_validation_score(score_file):
    with open(score_file, "r") as f:
        scores = json.load(f)

    if HPO_METRIC not in scores:
        raise KeyError(
            "Metric {!r} not found in {}. Available metrics: {}".format(
                HPO_METRIC,
                score_file,
                list(scores.keys()),
            )
        )

    score = float(scores[HPO_METRIC])

    if not math.isfinite(score):
        raise ValueError(
            "Non-finite {} in {}: {}".format(
                HPO_METRIC,
                score_file,
                score,
            )
        )

    return score


def append_hpo_history(history_file, record):
    with open(history_file, "a") as f:
        json.dump(record, f)
        f.write("\n")


def decode_params(x):
    """Decode the continuous DE vector into GraphDRP parameters."""

    # Batch size: categorical powers of two.
    batch_index = int(
        math.floor(float(x[0]))
    )
    batch_index = max(
        0,
        min(
            len(BATCH_SIZE_CHOICES) - 1,
            batch_index,
        ),
    )
    batch_size = BATCH_SIZE_CHOICES[
        batch_index
    ]

    # Learning rate: continuous log10 scale.
    log10_learning_rate = float(x[1])
    learning_rate = (
        10.0 ** log10_learning_rate
    )

    # Prediction-head freezing strategy: categorical.
    head_freeze_index = int(
        math.floor(float(x[2]))
    )
    head_freeze_index = max(
        0,
        min(
            len(HEAD_FREEZE_STRATEGIES) - 1,
            head_freeze_index,
        ),
    )
    freeze_head_layers = HEAD_FREEZE_STRATEGIES[
        head_freeze_index
    ]

    # Prediction-head dropout: continuous in [0.0, 0.9].
    dropout = float(x[3])
    dropout = max(
        DROPOUT_BOUNDS[0],
        min(DROPOUT_BOUNDS[1], dropout),
    )

    return {
        "batch_index": batch_index,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "log10_learning_rate": (
            log10_learning_rate
        ),
        "head_freeze_index": head_freeze_index,
        "freeze_head_layers": freeze_head_layers,
        "dropout": dropout,
        "freeze_layers": FIXED_FREEZE_LAYERS,
    }


############################################################
# RECOVER CCL PREPROCESSING / SCALERS
############################################################


def get_ccl_split_file(ccl_fold, stage):
    """Return the cancer-blind CCL split file for one fold/stage."""
    return os.path.join(
        CCL_SPLIT_DIR,
        "ccl_split_{}_{}_ids.txt".format(ccl_fold, stage),
    )


def get_ccl_preprocessed_dir(ccl_fold):
    """Return the preprocessing directory for one CCL fold."""
    return os.path.join(
        PRETRAINED_SCALER_ROOT,
        "ccl{}".format(ccl_fold),
    )


def required_ccl_preprocessed_files(ccl_fold):
    """Files created by gdrp_preprocess_replay.py for one CCL fold."""
    output_dir = get_ccl_preprocessed_dir(ccl_fold)

    return [
        os.path.join(output_dir, "processed", "train_data.pt"),
        os.path.join(output_dir, "processed", "val_data.pt"),
        os.path.join(output_dir, "processed", "test_data.pt"),
        os.path.join(output_dir, "train_y_data.csv"),
        os.path.join(output_dir, "val_y_data.csv"),
        os.path.join(output_dir, "test_y_data.csv"),
        os.path.join(output_dir, "x_data_gene_expression_scaler.gz"),
    ]


def ccl_preprocessing_complete(ccl_fold):
    """Return True only when the CCL preprocessing is complete."""
    return all(
        os.path.exists(path)
        for path in required_ccl_preprocessed_files(ccl_fold)
    )


def validate_ccl_preprocessing_inputs():
    """Fail before launching if the CCL recovery inputs are unavailable."""
    missing = []

    if not os.path.exists(CCL_PREPROCESS_SCRIPT):
        missing.append(CCL_PREPROCESS_SCRIPT)

    if not os.path.isdir(RAW_DATA_DIR):
        missing.append(RAW_DATA_DIR)

    if not os.path.isdir(CCL_SPLIT_DIR):
        missing.append(CCL_SPLIT_DIR)

    for ccl_fold in CCL_FOLDS:
        for stage in ("train", "val", "test"):
            split_file = get_ccl_split_file(ccl_fold, stage)
            if not os.path.exists(split_file):
                missing.append(split_file)

    if missing:
        raise FileNotFoundError(
            "Missing required CCL preprocessing inputs: {}".format(missing)
        )


def preprocess_ccl_fold(ccl_fold):
    """
    Generate one CCL fold's preprocessing and fitted gene-expression scaler.

    This deliberately does NOT pass --pretrained_ge_scaler. The CCL training
    split is the source/pretraining split, so preprocessing must fit its own
    scaler. That scaler is then reused by the transfer-learning preprocessing.
    """
    output_dir = get_ccl_preprocessed_dir(ccl_fold)

    if (
        not FORCE_CCL_PREPROCESS
        and ccl_preprocessing_complete(ccl_fold)
    ):
        print(
            "[CCL PREPROCESS REUSE] ccl{}: {}".format(
                ccl_fold,
                output_dir,
            ),
            flush=True,
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    train_split = get_ccl_split_file(ccl_fold, "train")
    val_split = get_ccl_split_file(ccl_fold, "val")
    test_split = get_ccl_split_file(ccl_fold, "test")

    print(
        "[CCL PREPROCESS START] ccl{} -> {}".format(
            ccl_fold,
            output_dir,
        ),
        flush=True,
    )

    run_cmd(
        [
            sys.executable,
            "-u",
            CCL_PREPROCESS_SCRIPT,
            "--input_dir",
            RAW_DATA_DIR,
            "--train_split_file",
            train_split,
            "--val_split_file",
            val_split,
            "--test_split_file",
            test_split,
            "--output_dir",
            output_dir,
        ],
        log_file=os.path.join(
            output_dir,
            "preprocess.log",
        ),
    )

    missing = [
        path
        for path in required_ccl_preprocessed_files(ccl_fold)
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "CCL{} preprocessing finished but expected files are missing: "
            "{}".format(ccl_fold, missing)
        )

    print(
        "[CCL PREPROCESS DONE] ccl{} scaler: {}".format(
            ccl_fold,
            get_pretrained_scaler(ccl_fold),
        ),
        flush=True,
    )


def preprocess_all_ccl_folds():
    """Create preprocess directories before transfer jobs start."""
    validate_ccl_preprocessing_inputs()

    os.makedirs(
        PRETRAINED_SCALER_ROOT,
        exist_ok=True,
    )

    print(
        "Preparing CCL preprocessing/scalers in: {}".format(
            PRETRAINED_SCALER_ROOT
        ),
        flush=True,
    )

    for ccl_fold in CCL_FOLDS:
        preprocess_ccl_fold(ccl_fold)

    print(
        "All required CCL preprocessing/scalers are ready.",
        flush=True,
    )


def get_pretrained_scaler(ccl_fold):
    """Return the pretraining gene-expression scaler for CCL fold i."""
    return os.path.join(
        PRETRAINED_SCALER_ROOT,
        "ccl{}".format(ccl_fold),
        "x_data_gene_expression_scaler.gz",
    )


def validate_pretrained_inputs():
    """Fail fast if any required i=0..9 checkpoint or scaler is missing."""
    missing = []

    for ccl_fold in CCL_FOLDS:
        pretrained_model = os.path.join(
            PRETRAIN_DIR,
            "split_{}_model.pt".format(ccl_fold),
        )
        pretrained_scaler = get_pretrained_scaler(ccl_fold)

        if not os.path.exists(pretrained_model):
            missing.append(pretrained_model)

        if not os.path.exists(pretrained_scaler):
            missing.append(pretrained_scaler)

    if missing:
        raise FileNotFoundError(
            "Missing required pretrained model/scaler files: {}".format(
                missing
            )
        )


def job_complete(run_dir, study):
    required = [
        MODEL_FILE_NAME,
        "best_hyperparameters.json",
        "timing.json",
        "study/test_scores.json",
    ]

    if study not in PILOT_STUDIES:
        required.append("mixed/test_scores.json")

    return all(
        os.path.exists(os.path.join(run_dir, path))
        for path in required
    )


############################################################
# HYPERPARAMETER OPTIMIZATION
############################################################


def optimize_hyperparameters(
    run_dir,
    processed_study_dir,
    pretrained_model,
    gpu_id,
    gpu_env,
):
    best_params_file = os.path.join(
        run_dir,
        "best_hyperparameters.json",
    )

    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            saved = json.load(f)

        # Prevent accidental reuse of a result generated with an older
        # HPO search space, including the previous categorical-dropout search.
        saved_search_space = saved.get("search_space", {})
        if (
            "freeze_head_layers" not in saved
            or "dropout" not in saved
            or saved.get("freeze_layers") != FIXED_FREEZE_LAYERS
            or saved_search_space.get("dropout_type") != "continuous"
            or saved_search_space.get("dropout_bounds") != list(DROPOUT_BOUNDS)
        ):
            raise RuntimeError(
                "Existing best_hyperparameters.json is incompatible with "
                "the continuous-dropout HPO experiment: {}".format(
                    best_params_file
                )
            )

        print(
            "[HPO RESUME] batch_size={} learning_rate={} "
            "freeze_layers={!r} freeze_head_layers={!r} dropout={}".format(
                saved["batch_size"],
                saved["learning_rate"],
                saved["freeze_layers"],
                saved["freeze_head_layers"],
                saved["dropout"],
            ),
            flush=True,
        )

        return saved

    hpo_dir = os.path.join(run_dir, "hpo_trials")
    history_file = os.path.join(run_dir, "hpo_history.jsonl")

    if os.path.exists(hpo_dir):
        shutil.rmtree(hpo_dir)

    if os.path.exists(history_file):
        os.remove(history_file)

    os.makedirs(hpo_dir, exist_ok=True)

    evaluation_number = 0

    def objective(x):
        nonlocal evaluation_number
        evaluation_number += 1

        decoded = decode_params(x)

        batch_index = decoded["batch_index"]
        batch_size = decoded["batch_size"]
        learning_rate = decoded["learning_rate"]
        log10_learning_rate = decoded["log10_learning_rate"]
        head_freeze_index = decoded["head_freeze_index"]
        freeze_head_layers = decoded["freeze_head_layers"]
        dropout = decoded["dropout"]
        freeze_layers = decoded["freeze_layers"]

        trial_dir = os.path.join(
            hpo_dir,
            "eval_{:04d}".format(evaluation_number),
        )
        os.makedirs(trial_dir, exist_ok=True)

        print(
            "[HPO] eval={} batch_size={} learning_rate={:.12g} "
            "log10_learning_rate={:.6f} freeze_layers={!r} "
            "freeze_head_layers={!r} dropout={} GPU={}".format(
                evaluation_number,
                batch_size,
                learning_rate,
                log10_learning_rate,
                freeze_layers,
                freeze_head_layers,
                dropout,
                gpu_id,
            ),
            flush=True,
        )

        trial_elapsed = None

        try:
            trial_elapsed = run_cmd(
                [
                    sys.executable,
                    "-u",
                    TRAIN_SCRIPT,
                    "--input_dir",
                    processed_study_dir,
                    "--output_dir",
                    trial_dir,
                    "--cuda_name",
                    "cuda:0",
                    "--learning_rate",
                    "{:.12g}".format(learning_rate),
                    "--batch_size",
                    str(batch_size),
                    "--freeze_layers",
                    freeze_layers,
                    "--freeze_head_layers",
                    freeze_head_layers,
                    "--dropout",
                    str(dropout),
                    "--epochs",
                    str(TRAIN_EPOCHS),
                    "--pretrained_model",
                    pretrained_model,
                ],
                log_file=os.path.join(
                    trial_dir,
                    "subprocess.log",
                ),
                env=gpu_env,
            )

            score_file = os.path.join(
                trial_dir,
                "val_scores.json",
            )
            score = read_validation_score(score_file)

            append_hpo_history(
                history_file,
                {
                    "evaluation": evaluation_number,
                    "batch_size": batch_size,
                    "batch_index": batch_index,
                    "learning_rate": learning_rate,
                    "log10_learning_rate": log10_learning_rate,
                    "freeze_layers": freeze_layers,
                    "head_freeze_index": head_freeze_index,
                    "freeze_head_layers": freeze_head_layers,
                    "dropout": dropout,
                    HPO_METRIC: score,
                    "elapsed_seconds": trial_elapsed,
                    "status": "ok",
                },
            )

            print(
                "[HPO RESULT] eval={} {}={}".format(
                    evaluation_number,
                    HPO_METRIC,
                    score,
                ),
                flush=True,
            )

            return score

        except Exception as e:
            append_hpo_history(
                history_file,
                {
                    "evaluation": evaluation_number,
                    "batch_size": batch_size,
                    "batch_index": batch_index,
                    "learning_rate": learning_rate,
                    "log10_learning_rate": log10_learning_rate,
                    "freeze_layers": freeze_layers,
                    "head_freeze_index": head_freeze_index,
                    "freeze_head_layers": freeze_head_layers,
                    "dropout": dropout,
                    HPO_METRIC: None,
                    "elapsed_seconds": trial_elapsed,
                    "status": "error",
                    "error": str(e),
                },
            )

            print(
                "[HPO ERROR] eval={}: {}".format(
                    evaluation_number,
                    e,
                ),
                flush=True,
            )

            return float("inf")

        finally:
            if DELETE_HPO_TRIALS:
                shutil.rmtree(
                    trial_dir,
                    ignore_errors=True,
                )

    try:
        result = differential_evolution(
            objective,
            bounds=[
                BATCH_INDEX_BOUNDS,
                LOG10_LEARNING_RATE_BOUNDS,
                HEAD_FREEZE_INDEX_BOUNDS,
                DROPOUT_BOUNDS,
            ],
            strategy=DE_STRATEGY,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            mutation=DE_MUTATION,
            recombination=DE_RECOMBINATION,
            polish=False,
            init="latinhypercube",
            seed=DE_SEED,
            disp=False,
        )
    finally:
        if DELETE_HPO_TRIALS:
            shutil.rmtree(
                hpo_dir,
                ignore_errors=True,
            )

    if not math.isfinite(result.fun):
        raise RuntimeError(
            "Differential evolution did not produce a finite "
            "validation score."
        )

    decoded_best = decode_params(result.x)

    best_params = {
        "batch_size": decoded_best["batch_size"],
        "batch_index": decoded_best["batch_index"],
        "learning_rate": decoded_best["learning_rate"],
        "log10_learning_rate": decoded_best[
            "log10_learning_rate"
        ],
        "freeze_layers": decoded_best["freeze_layers"],
        "head_freeze_index": decoded_best[
            "head_freeze_index"
        ],
        "freeze_head_layers": decoded_best[
            "freeze_head_layers"
        ],
        "dropout": decoded_best[
            "dropout"
        ],
        HPO_METRIC: float(result.fun),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "success": bool(result.success),
        "message": str(result.message),
        "search_space": {
            "batch_size_choices": BATCH_SIZE_CHOICES,
            "batch_encoded_bounds": list(
                BATCH_INDEX_BOUNDS
            ),
            "log10_learning_rate": list(
                LOG10_LEARNING_RATE_BOUNDS
            ),
            "learning_rate": [
                10.0 ** LOG10_LEARNING_RATE_BOUNDS[0],
                10.0 ** LOG10_LEARNING_RATE_BOUNDS[1],
            ],
            "fixed_freeze_layers": FIXED_FREEZE_LAYERS,
            "head_freeze_strategies": HEAD_FREEZE_STRATEGIES,
            "head_freeze_encoded_bounds": list(
                HEAD_FREEZE_INDEX_BOUNDS
            ),
            "dropout_bounds": list(DROPOUT_BOUNDS),
            "dropout_type": "continuous",
            "epochs": TRAIN_EPOCHS,
        },
        "de_settings": {
            "strategy": DE_STRATEGY,
            "maxiter": DE_MAXITER,
            "popsize": DE_POPSIZE,
            "tol": DE_TOL,
            "mutation": list(DE_MUTATION),
            "recombination": DE_RECOMBINATION,
            "seed": DE_SEED,
            "polish": False,
            "workers": 1,
        },
    }

    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=2)

    print(
        "[HPO BEST] batch_size={} learning_rate={:.12g} "
        "freeze_layers={!r} freeze_head_layers={!r} dropout={} {}={}".format(
            best_params["batch_size"],
            best_params["learning_rate"],
            best_params["freeze_layers"],
            best_params["freeze_head_layers"],
            best_params["dropout"],
            HPO_METRIC,
            best_params[HPO_METRIC],
        ),
        flush=True,
    )

    return best_params


############################################################
# SINGLE FOLD JOB
############################################################


def run_transfer_job(
    study,
    ccl_fold,
    pdo_fold,
    gpu_queue,
):
    run_dir = os.path.join(
        BASE_OUTPUT_DIR,
        study,
        "ccl{}_j{}".format(ccl_fold, pdo_fold),
    )
    os.makedirs(run_dir, exist_ok=True)

    if job_complete(run_dir, study):
        print(
            "[SKIP DONE] {} ccl={} pdo={}".format(
                study,
                ccl_fold,
                pdo_fold,
            ),
            flush=True,
        )
        return

    gpu_id = gpu_queue.get()
    start_time = time.time()

    preprocess_study_time = None
    hpo_time = None
    final_train_time = None
    study_infer_time = None
    preprocess_mixed_time = None
    mixed_infer_time = None

    timing_file = os.path.join(run_dir, "timing.json")

    try:
        print(
            "[START] {} ccl={} pdo={} GPU={}".format(
                study,
                ccl_fold,
                pdo_fold,
                gpu_id,
            ),
            flush=True,
        )

        gpu_env = os.environ.copy()
        gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_env["PYTHONUNBUFFERED"] = "1"

        pretrained_model = os.path.join(
            PRETRAIN_DIR,
            "split_{}_model.pt".format(ccl_fold),
        )

        pretrained_scaler = get_pretrained_scaler(ccl_fold)

        train_split = os.path.join(
            BASE_SPLIT_DIR,
            "{}_split_{}_train_ids.txt".format(
                study,
                pdo_fold,
            ),
        )
        val_split = os.path.join(
            BASE_SPLIT_DIR,
            "{}_split_{}_val_ids.txt".format(
                study,
                pdo_fold,
            ),
        )
        test_split = os.path.join(
            BASE_SPLIT_DIR,
            "{}_split_{}_test_ids.txt".format(
                study,
                pdo_fold,
            ),
        )

        required_inputs = [
            pretrained_model,
            pretrained_scaler,
            train_split,
            val_split,
            test_split,
        ]

        mixed_split = None
        if study not in PILOT_STUDIES:
            mixed_split = os.path.join(
                BASE_SPLIT_DIR,
                "{}_ccl_ids.txt".format(study),
            )
            required_inputs.append(mixed_split)

        missing_inputs = [
            path
            for path in required_inputs
            if not os.path.exists(path)
        ]

        if missing_inputs:
            raise FileNotFoundError(
                "Missing required input files: {}".format(
                    missing_inputs
                )
            )

        processed_root = os.path.join(
            run_dir,
            "processed",
        )
        processed_study_dir = os.path.join(
            processed_root,
            "study",
        )
        processed_mixed_dir = os.path.join(
            processed_root,
            "mixed",
        )

        os.makedirs(
            processed_study_dir,
            exist_ok=True,
        )

        preprocess_study_time = run_cmd(
            [
                sys.executable,
                "-u",
                PREPROCESS_SCRIPT,
                "--input_dir",
                RAW_DATA_DIR,
                "--output_dir",
                processed_study_dir,
                "--train_split_file",
                train_split,
                "--val_split_file",
                val_split,
                "--test_split_file",
                test_split,
                "--pretrained_ge_scaler",
                pretrained_scaler,
            ],
            log_file=os.path.join(
                run_dir,
                "preprocess_study_subprocess.log",
            ),
            env=gpu_env,
        )

        ####################################################
        # HYPERPARAMETER OPTIMIZATION
        ####################################################

        hpo_start = time.time()

        best_params = optimize_hyperparameters(
            run_dir=run_dir,
            processed_study_dir=processed_study_dir,
            pretrained_model=pretrained_model,
            gpu_id=gpu_id,
            gpu_env=gpu_env,
        )

        hpo_time = time.time() - hpo_start

        ####################################################
        # FINAL TRAINING WITH THE BEST CONFIGURATION
        ####################################################

        final_train_time = run_cmd(
            [
                sys.executable,
                "-u",
                TRAIN_SCRIPT,
                "--input_dir",
                processed_study_dir,
                "--output_dir",
                run_dir,
                "--cuda_name",
                "cuda:0",
                "--learning_rate",
                "{:.12g}".format(
                    best_params["learning_rate"]
                ),
                "--batch_size",
                str(best_params["batch_size"]),
                "--freeze_layers",
                best_params["freeze_layers"],
                "--freeze_head_layers",
                best_params["freeze_head_layers"],
                "--dropout",
                str(best_params["dropout"]),
                "--epochs",
                str(TRAIN_EPOCHS),
                "--pretrained_model",
                pretrained_model,
            ],
            log_file=os.path.join(
                run_dir,
                "transfer_train_subprocess.log",
            ),
            env=gpu_env,
        )

        ####################################################
        # STUDY TEST INFERENCE
        ####################################################

        study_dir = os.path.join(run_dir, "study")
        os.makedirs(study_dir, exist_ok=True)

        study_infer_time = run_cmd(
            [
                sys.executable,
                "-u",
                INFER_SCRIPT,
                "--input_data_dir",
                processed_study_dir,
                "--input_model_dir",
                run_dir,
                "--output_dir",
                study_dir,
                "--cuda_name",
                "cuda:0",
                "--calc_infer_scores",
                "true",
            ],
            log_file=os.path.join(
                study_dir,
                "infer_subprocess.log",
            ),
            env=gpu_env,
        )

        ####################################################
        # MIXED TEST INFERENCE
        ####################################################

        if study not in PILOT_STUDIES:
            os.makedirs(
                processed_mixed_dir,
                exist_ok=True,
            )

            preprocess_mixed_time = run_cmd(
                [
                    sys.executable,
                    "-u",
                    PREPROCESS_SCRIPT,
                    "--input_dir",
                    RAW_DATA_DIR,
                    "--output_dir",
                    processed_mixed_dir,
                    "--train_split_file",
                    train_split,
                    "--val_split_file",
                    val_split,
                    "--test_split_file",
                    mixed_split,
                    "--pretrained_ge_scaler",
                    pretrained_scaler,
                ],
                log_file=os.path.join(
                    run_dir,
                    "preprocess_mixed_subprocess.log",
                ),
                env=gpu_env,
            )

            mixed_dir = os.path.join(run_dir, "mixed")
            os.makedirs(mixed_dir, exist_ok=True)

            mixed_infer_time = run_cmd(
                [
                    sys.executable,
                    "-u",
                    INFER_SCRIPT,
                    "--input_data_dir",
                    processed_mixed_dir,
                    "--input_model_dir",
                    run_dir,
                    "--output_dir",
                    mixed_dir,
                    "--cuda_name",
                    "cuda:0",
                    "--calc_infer_scores",
                    "true",
                ],
                log_file=os.path.join(
                    mixed_dir,
                    "infer_subprocess.log",
                ),
                env=gpu_env,
            )

        total_elapsed = time.time() - start_time

        timing = {
            "status": "complete",
            "study": study,
            "ccl_fold": ccl_fold,
            "pdo_fold": pdo_fold,
            "gpu_id": gpu_id,
            "pretrained_scaler": pretrained_scaler,
            "preprocess_study_seconds": preprocess_study_time,
            "hpo_elapsed_seconds": hpo_time,
            "final_train_elapsed_seconds": final_train_time,
            "study_infer_elapsed_seconds": study_infer_time,
            "preprocess_mixed_seconds": preprocess_mixed_time,
            "mixed_infer_elapsed_seconds": mixed_infer_time,
            "total_elapsed_seconds": total_elapsed,
            "total_elapsed_minutes": total_elapsed / 60.0,
            "hpo_evaluations": best_params["nfev"],
            "best_batch_size": best_params["batch_size"],
            "best_learning_rate": best_params[
                "learning_rate"
            ],
            "fixed_freeze_layers": best_params[
                "freeze_layers"
            ],
            "best_freeze_head_layers": best_params[
                "freeze_head_layers"
            ],
            "best_dropout": best_params["dropout"],
            "train_epochs": TRAIN_EPOCHS,
        }

        with open(timing_file, "w") as f:
            json.dump(timing, f, indent=2)

        print(
            "[DONE] {} ccl={} pdo={} GPU={} "
            "({:.2f} min)".format(
                study,
                ccl_fold,
                pdo_fold,
                gpu_id,
                total_elapsed / 60.0,
            ),
            flush=True,
        )

    except Exception as e:
        total_elapsed = time.time() - start_time

        error_record = {
            "status": "error",
            "study": study,
            "ccl_fold": ccl_fold,
            "pdo_fold": pdo_fold,
            "gpu_id": gpu_id,
            "pretrained_scaler": pretrained_scaler,
            "total_elapsed_seconds": total_elapsed,
            "total_elapsed_minutes": total_elapsed / 60.0,
            "error": str(e),
        }

        with open(timing_file, "w") as f:
            json.dump(error_record, f, indent=2)

        with open(
            os.path.join(run_dir, "job_error.log"),
            "w",
        ) as f:
            f.write(traceback.format_exc())

        print(
            "[ERROR] {} ccl={} pdo={} GPU={}: {}".format(
                study,
                ccl_fold,
                pdo_fold,
                gpu_id,
                e,
            ),
            flush=True,
        )

        raise

    finally:
        gpu_queue.put(gpu_id)


############################################################
# FOLD DISCOVERY
############################################################


def discover_pdo_folds(study):
    """Return PDO folds with train, validation, and test split files."""
    folds = []

    prefix = "{}_split_".format(study)
    suffix = "_train_ids.txt"

    for name in os.listdir(BASE_SPLIT_DIR):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue

        fold_text = name[
            len(prefix):
            len(name) - len(suffix)
        ]

        if not fold_text.isdigit():
            continue

        fold = int(fold_text)

        # PDO fold j is study-specific and is allowed only in 0..9.
        if fold < 0 or fold > 9:
            continue

        val_file = os.path.join(
            BASE_SPLIT_DIR,
            "{}_split_{}_val_ids.txt".format(study, fold),
        )
        test_file = os.path.join(
            BASE_SPLIT_DIR,
            "{}_split_{}_test_ids.txt".format(study, fold),
        )

        if os.path.exists(val_file) and os.path.exists(test_file):
            folds.append(fold)

    folds = sorted(set(folds))

    if not folds:
        raise RuntimeError(
            "No complete PDO folds found for study {!r} in {}".format(
                study,
                BASE_SPLIT_DIR,
            )
        )

    return folds


############################################################
# MAIN
############################################################


def main():
    jobs = []

    # The transfer preprocessing below requires one scaler per CCL fold.
    # Generate those CCL preprocessing directories/scalers first if needed.
    preprocess_all_ccl_folds()

    validate_pretrained_inputs()

    print(
        "CCL folds: {}".format(CCL_FOLDS),
        flush=True,
    )

    for study in STUDIES:
        pdo_folds = discover_pdo_folds(study)

        print(
            "Discovered PDO folds for {}: {}".format(
                study,
                pdo_folds,
            ),
            flush=True,
        )

        for ccl_fold in CCL_FOLDS:
            for pdo_fold in pdo_folds:
                jobs.append(
                    (
                        study,
                        ccl_fold,
                        pdo_fold,
                    )
                )

    print(
        "Total jobs: {}".format(len(jobs)),
        flush=True,
    )
    print(
        "Maximum concurrent jobs: {}".format(
            MAX_ACTIVE_JOBS,
        ),
        flush=True,
    )
    print(
        "GPU slots: {}".format(GPU_SLOTS),
        flush=True,
    )
    print(
        "Fixed encoder freezing: {!r}".format(
            FIXED_FREEZE_LAYERS
        ),
        flush=True,
    )
    print(
        "Head-freezing HPO choices: {}".format(
            HEAD_FREEZE_STRATEGIES
        ),
        flush=True,
    )
    print(
        "Continuous dropout HPO bounds: {}".format(
            DROPOUT_BOUNDS
        ),
        flush=True,
    )
    print(
        "Training epochs per HPO/final run: {}".format(
            TRAIN_EPOCHS
        ),
        flush=True,
    )

    with Manager() as manager:
        gpu_queue = manager.Queue()

        for gpu_id, slots in GPU_SLOTS.items():
            for _ in range(slots):
                gpu_queue.put(gpu_id)

        with ProcessPoolExecutor(
            max_workers=MAX_ACTIVE_JOBS
        ) as executor:
            futures = [
                executor.submit(
                    run_transfer_job,
                    study,
                    ccl_fold,
                    pdo_fold,
                    gpu_queue,
                )
                for study, ccl_fold, pdo_fold in jobs
            ]

            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
