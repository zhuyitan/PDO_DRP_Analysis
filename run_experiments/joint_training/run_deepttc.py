import os
import sys
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

#############################################
# CONFIG
#############################################

STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1"
]


# Every split type is run. Results for each go in their
# own folder under BASE_OUTPUT_DIR.
SPLIT_TYPES = [
    "random",
    "cancer-blind"
]

SPLITS_ROOT = "pdo_drp_data/splits"
RAW_DATA_DIR = "pdo_drp_data"
BASE_OUTPUT_DIR = "joint_training_results_deepttc"

# Number of CCL splits (i). The number of PDO splits (j) for
# each study is detected from the split files, because
# cancer-blind studies do not all have the same number.
NUM_CCL_SPLITS = 10

DEEPTTC_DIR = "models/DeepTTC"

# All paths are converted to absolute paths so they stay
# valid wherever a command runs from.
SPLITS_ROOT = os.path.abspath(SPLITS_ROOT)
RAW_DATA_DIR = os.path.abspath(RAW_DATA_DIR)
BASE_OUTPUT_DIR = os.path.abspath(BASE_OUTPUT_DIR)
DEEPTTC_DIR = os.path.abspath(DEEPTTC_DIR)

PREPROCESS_SCRIPT = os.path.join(
    DEEPTTC_DIR,
    "deepttc_preprocess_improve.py"
)

TRAIN_SCRIPT = os.path.join(
    DEEPTTC_DIR,
    "deepttc_train_improve.py"
)

INFER_SCRIPT = os.path.join(
    DEEPTTC_DIR,
    "deepttc_infer_improve.py"
)

# Saved by the training script. If it already exists, a
# restarted run skips training and only redoes any missing
# inference.
MODEL_FILE_NAME = "model.pt"

# Studies that do NOT have a {study}_ccl_ids.txt file.
# The mixed (study CCL) inference is skipped for these.
SEPERATE_STUDIES = {
    "Pilot_1"
}

#############################################
# FILES ALLOWED TO BE DELETED
#############################################

# After every inference for a job has finished, intermediate
# preprocessed data files with these extensions are deleted
# to save disk space. The model, logs, predictions and scores
# are kept.
INTERMEDIATE_EXTENSIONS = (
    ".h5",
    ".parquet"
)

#############################################
# GPU DETECTION
#############################################

def detect_gpus():
    """
    Return the GPUs available on this machine.

    If CUDA_VISIBLE_DEVICES is set, only those GPUs are used.
    Otherwise every GPU reported by nvidia-smi is used.
    If no GPU is found, a single CPU worker ("cpu") is used.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if visible is not None:
        gpus = [
            gpu.strip()
            for gpu in visible.split(",")
            if gpu.strip()
        ]
        return gpus or ["cpu"]

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True
        )
        count = len([
            line
            for line in result.stdout.splitlines()
            if line.startswith("GPU")
        ])
    except OSError:
        count = 0

    if count == 0:
        return ["cpu"]

    return [str(gpu) for gpu in range(count)]


GPUS = detect_gpus()

#############################################
# SHARED STATE
#############################################

# Number of split pipelines that share each GPU. Defaults to 1.
# Set the JOBS_PER_GPU environment variable to run more.
JOBS_PER_GPU = int(os.environ.get("JOBS_PER_GPU", "1"))

manager = Manager()

gpu_queue = manager.Queue()

for gpu_id in GPUS:
    for _ in range(JOBS_PER_GPU):
        gpu_queue.put(gpu_id)

#############################################
# UTILITIES
#############################################

def run_cmd(cmd, log_file, env=None, cwd=None):
    """
    Run a subprocess with its output saved to log_file, and
    raise an exception if it fails.
    """

    os.makedirs(
        os.path.dirname(log_file),
        exist_ok=True
    )

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd
        )

    if result.returncode != 0:

        with open(log_file, errors="replace") as log:
            tail = log.readlines()[-30:]

        print(
            "\n[COMMAND FAILED]\n"
            + " ".join(cmd),
            flush=True
        )

        print(
            f"\n[LOG TAIL] {log_file}\n"
            + "".join(tail),
            flush=True
        )

        raise RuntimeError(
            f"Command failed with return code "
            f"{result.returncode}: {' '.join(cmd)}"
        )


def job_env(gpu_id, i, j):
    """
    Environment for one command.

    Pins the command to its GPU and sets the hash seed
    from the split indices. DeepTTC commands run from
    inside DEEPTTC_DIR.
    """

    env = os.environ.copy()

    if gpu_id != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    env["PYTHONHASHSEED"] = f"{i}{j}"

    return env


def detect_pdo_split_count(split_dir, study):
    """
    Count the PDO splits (j) for a study by looking for
    {study}_split_{j}_val_ids.txt files.
    """

    j = 0

    while os.path.isfile(
        os.path.join(split_dir, f"{study}_split_{j}_val_ids.txt")
    ):
        j += 1

    if j == 0:
        raise FileNotFoundError(
            f"No PDO splits found for study '{study}' in {split_dir}"
        )

    return j


def job_complete(run_dir, study):
    """
    True if every inference output for this job already exists.
    """

    required = [
        os.path.join(run_dir, "ccl", "test_scores.json"),
        os.path.join(run_dir, "study", "test_scores.json")
    ]

    if study not in SEPERATE_STUDIES:
        required.append(
            os.path.join(run_dir, "mixed", "test_scores.json")
        )

    return all(
        os.path.isfile(path)
        for path in required
    )


def remove_intermediate_files(run_dir):
    """
    Delete ONLY the preprocessed data files listed in
    INTERMEDIATE_EXTENSIONS. No directories are removed.
    """

    for root, _, files in os.walk(run_dir):

        for filename in files:

            if filename.endswith(INTERMEDIATE_EXTENSIONS):

                os.remove(
                    os.path.join(root, filename)
                )


#############################################
# TEST SET INFERENCE
#############################################

def run_test_set(label, train_split, val_split, test_split,
                 processed_dir, out_dir, model_dir, gpu_id, i, j):
    """
    Preprocess one test set and run inference on it with the
    trained model. Skipped if its scores already exist.
    """

    if os.path.isfile(os.path.join(out_dir, "test_scores.json")):

        print(
            f"[SKIP] {label} already done",
            flush=True
        )

        return

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    #################################################
    # PREPROCESS
    #################################################

    print(
        f"[{label} PREPROCESS START]",
        flush=True
    )

    run_cmd([
        "python",
        PREPROCESS_SCRIPT,

        "--input_dir",
        RAW_DATA_DIR,

        "--output_dir",
        processed_dir,

        "--train_split_file",
        train_split,

        "--val_split_file",
        val_split,

        "--test_split_file",
        test_split
    ], log_file=os.path.join(processed_dir, "preprocess.log"),
       env=job_env(gpu_id, i, j), cwd=DEEPTTC_DIR)

    print(
        f"[{label} PREPROCESS DONE]",
        flush=True
    )

    #################################################
    # INFERENCE
    #################################################

    print(
        f"[{label} INFER START]",
        flush=True
    )

    run_cmd([
        "python",
        INFER_SCRIPT,

        "--input_data_dir",
        processed_dir,

        # Reuse the model trained during the CCL stage.
        "--input_model_dir",
        model_dir,

        "--output_dir",
        out_dir,

        "--calc_infer_scores",
        "true"
    ], log_file=os.path.join(out_dir, "infer.log"),
       env=job_env(gpu_id, i, j), cwd=DEEPTTC_DIR)

    print(
        f"[{label} INFER DONE]",
        flush=True
    )


#############################################
# CORE PIPELINE
#############################################

def run_pipeline(split_type, study, i, j):
    """
    Train and evaluate one (split_type, study, i, j) job.
    Returns True if every step finished without errors.
    """

    ok = False

    gpu_id = None

    name = f"{split_type} {study} i={i} j={j}"

    try:

        #####################################################
        # GET GPU SLOT
        #####################################################

        gpu_id = gpu_queue.get()

        #####################################################
        # RUN DIRECTORY
        #####################################################

        split_dir = os.path.join(
            SPLITS_ROOT,
            split_type
        )

        run_dir = os.path.join(
            BASE_OUTPUT_DIR,
            split_type,
            f"{study}_i{i}_j{j}"
        )

        if job_complete(run_dir, study):

            print(
                f"[SKIP DONE] {name}",
                flush=True
            )

            ok = True

            return ok

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        print(
            f"[START] {name} "
            f"GPU={gpu_id}",
            flush=True
        )

        #####################################################
        # COMMON SPLITS
        #####################################################

        train_split = os.path.join(
            split_dir,
            f"{study}_split_{j}_ccl_split_{i}_train_ids.txt"
        )

        val_split = os.path.join(
            split_dir,
            f"{study}_split_{j}_val_ids.txt"
        )

        ccl_test_split = os.path.join(
            split_dir,
            f"ccl_split_{i}_test_ids.txt"
        )

        #####################################################
        #####################################################
        # 1. CCL
        #####################################################
        #####################################################

        ccl_dir = os.path.join(
            run_dir,
            "processed_ccl"
        )

        os.makedirs(
            ccl_dir,
            exist_ok=True
        )

        if os.path.isfile(os.path.join(ccl_dir, MODEL_FILE_NAME)):

            print(
                f"[TRAIN SKIP] {name}: "
                f"{MODEL_FILE_NAME} exists, only redoing missing inference",
                flush=True
            )

        else:

            #################################################
            # CCL PREPROCESS
            #################################################

            print(
                f"[CCL PREPROCESS START] {name}",
                flush=True
            )

            run_cmd([
                "python",
                PREPROCESS_SCRIPT,

                "--input_dir",
                RAW_DATA_DIR,

                "--output_dir",
                ccl_dir,

                "--train_split_file",
                train_split,

                "--val_split_file",
                val_split,

                "--test_split_file",
                ccl_test_split
            ], log_file=os.path.join(ccl_dir, "preprocess.log"),
               env=job_env(gpu_id, i, j), cwd=DEEPTTC_DIR)

            print(
                f"[CCL PREPROCESS DONE] {name}",
                flush=True
            )

            #################################################
            # TRAIN MODEL
            #################################################

            print(
                f"[TRAIN START] {name} "
                f"GPU={gpu_id}",
                flush=True
            )

            run_cmd([
                "python",
                TRAIN_SCRIPT,

                "--input_dir",
                ccl_dir,

                "--output_dir",
                ccl_dir
            ], log_file=os.path.join(ccl_dir, "train.log"),
               env=job_env(gpu_id, i, j), cwd=DEEPTTC_DIR)

            print(
                f"[TRAIN DONE] {name}",
                flush=True
            )

        #####################################################
        # TEST SETS
        #
        # The same trained model is reused for every test set.
        # A failure in one test set is reported and the
        # remaining test sets still run.
        #####################################################

        any_failed = False

        #####################################################
        # CCL INFERENCE
        #
        # The CCL test split was already preprocessed with
        # the training data, so it only needs inference.
        #####################################################

        ccl_out = os.path.join(
            run_dir,
            "ccl"
        )

        if os.path.isfile(os.path.join(ccl_out, "test_scores.json")):

            print(
                f"[SKIP] {name} CCL already done",
                flush=True
            )

        else:

            try:

                os.makedirs(
                    ccl_out,
                    exist_ok=True
                )

                print(
                    f"[CCL INFER START] {name}",
                    flush=True
                )

                run_cmd([
                    "python",
                    INFER_SCRIPT,

                    "--input_data_dir",
                    ccl_dir,

                    "--input_model_dir",
                    ccl_dir,

                    "--output_dir",
                    ccl_out,

                    "--calc_infer_scores",
                    "true"
                ], log_file=os.path.join(ccl_out, "infer.log"),
                   env=job_env(gpu_id, i, j), cwd=DEEPTTC_DIR)

                print(
                    f"[CCL INFER DONE] {name}",
                    flush=True
                )

            except Exception as e:
                any_failed = True
                print(
                    f"[ERROR] {name} CCL: {e}",
                    flush=True
                )

        #####################################################
        # STUDY INFERENCE  (held-out PDO split j)
        #####################################################

        study_split = os.path.join(
            split_dir,
            f"{study}_split_{j}_test_ids.txt"
        )

        try:
            run_test_set(
                f"{name} STUDY",
                train_split,
                val_split,
                study_split,
                os.path.join(run_dir, "processed_study"),
                os.path.join(run_dir, "study"),
                ccl_dir,
                gpu_id,
                i,
                j
            )
        except Exception as e:
            any_failed = True
            print(
                f"[ERROR] {name} STUDY: {e}",
                flush=True
            )

        #####################################################
        # MIXED INFERENCE  (study CCL IDs)
        #####################################################

        if study in SEPERATE_STUDIES:

            print(
                f"[SKIP] {name} MIXED "
                f"(no CCL IDs file for this study)",
                flush=True
            )

        else:

            mixed_split = os.path.join(
                split_dir,
                f"{study}_ccl_ids.txt"
            )

            try:
                run_test_set(
                    f"{name} MIXED",
                    train_split,
                    val_split,
                    mixed_split,
                    os.path.join(run_dir, "processed_mixed"),
                    os.path.join(run_dir, "mixed"),
                    ccl_dir,
                    gpu_id,
                    i,
                    j
                )
            except Exception as e:
                any_failed = True
                print(
                    f"[ERROR] {name} MIXED: {e}",
                    flush=True
                )

        #####################################################
        # SUCCESS
        #####################################################

        ok = not any_failed

        if any_failed:
            print(
                f"[PIPELINE COMPLETE WITH ERRORS] {name}",
                flush=True
            )
        else:
            remove_intermediate_files(
                run_dir
            )

            print(
                f"[PIPELINE COMPLETE] {name}",
                flush=True
            )

    except Exception as e:

        #####################################################
        # ERROR
        #####################################################

        print(
            f"[ERROR] {name}: {e}",
            flush=True
        )

    finally:

        #####################################################
        # RETURN GPU SLOT
        #####################################################

        if gpu_id is not None:
            gpu_queue.put(gpu_id)

    return ok


#############################################
# MAIN
#############################################

def build_jobs():
    """
    Every (split_type, study, i, j) combination, with the
    number of PDO splits detected for each study.
    """

    jobs = []

    for split_type in SPLIT_TYPES:

        split_dir = os.path.join(
            SPLITS_ROOT,
            split_type
        )

        for study in STUDIES:

            num_pdo_splits = detect_pdo_split_count(
                split_dir,
                study
            )

            print(
                f"{split_type} {study}: "
                f"{num_pdo_splits} PDO splits",
                flush=True
            )

            jobs.extend(
                (split_type, study, i, j)
                for i, j in itertools.product(
                    range(NUM_CCL_SPLITS),
                    range(num_pdo_splits)
                )
            )

    return jobs


def main():

    jobs = build_jobs()

    print(
        f"Total jobs: {len(jobs)}",
        flush=True
    )

    if GPUS == ["cpu"]:
        print(
            "GPUs: none found, running on CPU",
            flush=True
        )
    else:
        print(
            f"GPUs: {', '.join(GPUS)}",
            flush=True
        )

    print(
        f"Jobs per GPU: {JOBS_PER_GPU}",
        flush=True
    )

    with ProcessPoolExecutor(
        max_workers=len(GPUS) * JOBS_PER_GPU
    ) as executor:

        futures = [
            executor.submit(
                run_pipeline,
                *job
            )
            for job in jobs
        ]

        failed = 0

        for future in as_completed(futures):
            if not future.result():
                failed += 1

    print(
        f"Finished: {len(jobs) - failed} of {len(jobs)} "
        f"jobs completed without errors",
        flush=True
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()