import os
import subprocess
import itertools
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
BASE_OUTPUT_DIR = "cell_line_only_results_graphdrp"

NUM_SPLITS = 10

GRAPHDRP_DIR = "models/GraphDRP"

PREPROCESS_SCRIPT = os.path.join(
    GRAPHDRP_DIR,
    "graphdrp_preprocess_improve.py"
)

TRAIN_SCRIPT = os.path.join(
    GRAPHDRP_DIR,
    "graphdrp_train_improve.py"
)

INFER_SCRIPT = os.path.join(
    GRAPHDRP_DIR,
    "graphdrp_infer_improve.py"
)

# Studies that do NOT have a {study}_ccl_ids.txt file.
# The study CCL inference is skipped for these.
SEPERATE_STUDIES = {
    "Pilot_1"
}

#############################################
# FILES ALLOWED TO BE DELETED
#############################################

# These are the ONLY files this script will delete.
#
# GraphDRP TestbedDataset creates:
#
#   <output_dir>/processed/train_data.pt
#   <output_dir>/processed/val_data.pt
#   <output_dir>/processed/test_data.pt
#
# Everything else is retained.
PREPROCESSING_PT_FILES = (
    "train_data.pt",
    "val_data.pt",
    "test_data.pt"
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

manager = Manager()

# One split pipeline runs on each GPU at a time.
gpu_queue = manager.Queue()

for gpu_id in GPUS:
    gpu_queue.put(gpu_id)

#############################################
# UTILITIES
#############################################

def run_cmd(cmd, env=None, cwd=None):
    """
    Run a subprocess and raise an exception if it fails.
    """

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd
    )

    if result.returncode != 0:

        print(
            "\n[COMMAND FAILED]\n"
            + " ".join(cmd),
            flush=True
        )

        print(
            "\n[STDOUT]\n"
            + result.stdout,
            flush=True
        )

        print(
            "\n[STDERR]\n"
            + result.stderr,
            flush=True
        )

        raise RuntimeError(
            f"Command failed with return code "
            f"{result.returncode}: {' '.join(cmd)}"
        )


def job_env(gpu_id, i):
    """
    Environment for one command.

    Pins the command to its GPU and sets the hash seed
    to the split index.
    """

    env = os.environ.copy()

    if gpu_id != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    env["PYTHONHASHSEED"] = str(i)

    return env


def cuda_name(gpu_id):
    """
    Device for --cuda_name. Each command only sees its own GPU
    (see job_env), so that GPU is always cuda:0.
    """

    if gpu_id == "cpu":
        return "cpu"

    return "cuda:0"



def remove_preprocessing_pt_files(preprocess_dir):
    """
    Delete ONLY the three GraphDRP preprocessing tensor files:

        processed/train_data.pt
        processed/val_data.pt
        processed/test_data.pt

    No directories are removed.

    No other files are removed.

    In particular, this preserves:

        model.pt
        train_y_data.csv
        val_y_data.csv
        test_y_data.csv
        val_y_data_predicted.csv
        test_y_data_predicted.csv
        val_scores.json
        test_scores.json
        history.csv
        runtime_preprocess.json
        runtime_train.json
        runtime_infer.json
        transformation files
        all output directories
    """

    if preprocess_dir is None:
        return

    processed_dir = os.path.join(
        preprocess_dir,
        "processed"
    )

    for filename in PREPROCESSING_PT_FILES:

        file_path = os.path.join(
            processed_dir,
            filename
        )

        if os.path.isfile(file_path):

            os.remove(file_path)

            print(
                f"[DELETE] {file_path}",
                flush=True
            )


#############################################
# TEST SET INFERENCE
#############################################

def run_test_set(label, split_dir, i, gpu_id, test_split, processed_dir, out_dir, model_dir):
    """
    Preprocess one test set and run inference on it with the
    model trained for CCL split i.
    """

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    train_split = os.path.join(
        split_dir,
        f"ccl_split_{i}_train_ids.txt"
    )

    val_split = os.path.join(
        split_dir,
        f"ccl_split_{i}_val_ids.txt"
    )

    #################################################
    # PREPROCESS
    #################################################

    print(
        f"[{label} PREPROCESS START] "
        f"i={i}",
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
    ], env=job_env(gpu_id, i))

    print(
        f"[{label} PREPROCESS DONE] "
        f"i={i}",
        flush=True
    )

    #################################################
    # INFERENCE
    #################################################

    print(
        f"[{label} INFER START] "
        f"i={i}",
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

        "--cuda_name",
        cuda_name(gpu_id),

        "--calc_infer_scores",
        "true"
    ], env=job_env(gpu_id, i))

    print(
        f"[{label} INFER DONE] "
        f"i={i}",
        flush=True
    )

    #################################################
    # CLEANUP
    #
    # These tensors are no longer needed.
    #################################################

    remove_preprocessing_pt_files(
        processed_dir
    )


#############################################
# CORE PIPELINE
#############################################

def run_pipeline(split_type, i):

    gpu_id = None

    # Every preprocessing directory is tracked so the
    # finally block can clean up after a failure at any
    # pipeline stage.
    processed_dirs = []

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
            f"split_{i}"
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        print(
            f"[START] "
            f"{split_type} i={i} "
            f"GPU={gpu_id}",
            flush=True
        )

        #####################################################
        # COMMON SPLITS
        #####################################################

        train_split = os.path.join(
            split_dir,
            f"ccl_split_{i}_train_ids.txt"
        )

        val_split = os.path.join(
            split_dir,
            f"ccl_split_{i}_val_ids.txt"
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

        processed_dirs.append(ccl_dir)

        #####################################################
        # CCL PREPROCESS
        #####################################################

        print(
            f"[CCL PREPROCESS START] "
            f"{split_type} i={i}",
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
        ], env=job_env(gpu_id, i))

        print(
            f"[CCL PREPROCESS DONE] "
            f"{split_type} i={i}",
            flush=True
        )

        #####################################################
        # TRAIN MODEL
        #####################################################

        print(
            f"[TRAIN START] "
            f"{split_type} i={i} "
            f"GPU={gpu_id}",
            flush=True
        )

        run_cmd([
            "python",
            TRAIN_SCRIPT,

            "--input_dir",
            ccl_dir,

            "--output_dir",
            ccl_dir,

            "--cuda_name",
            cuda_name(gpu_id)
        ], env=job_env(gpu_id, i))

        print(
            f"[TRAIN DONE] "
            f"{split_type} i={i}",
            flush=True
        )

        #####################################################
        # CCL INFERENCE
        #####################################################

        ccl_out = os.path.join(
            run_dir,
            "ccl"
        )

        os.makedirs(
            ccl_out,
            exist_ok=True
        )

        print(
            f"[CCL INFER START] "
            f"{split_type} i={i}",
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

            "--cuda_name",
            cuda_name(gpu_id),

            "--calc_infer_scores",
            "true"
        ], env=job_env(gpu_id, i))

        print(
            f"[CCL INFER DONE] "
            f"{split_type} i={i}",
            flush=True
        )

        #####################################################
        # CCL CLEANUP
        #
        # CCL train/val/test tensors are no longer needed.
        #
        # IMPORTANT:
        # model.pt is NOT inside processed/, so it remains.
        #####################################################

        remove_preprocessing_pt_files(
            ccl_dir
        )

        #####################################################
        #####################################################
        # 2. STUDIES
        #####################################################
        #####################################################

        # The same trained model is reused for every study.
        # A failure in one test set is reported and the
        # remaining test sets still run.

        any_failed = False

        for study in STUDIES:

            study_root = os.path.join(
                run_dir,
                study
            )

            #################################################
            # STUDY PDO
            #################################################

            pdo_split = os.path.join(
                split_dir,
                f"{study}_pdo_ids.txt"
            )

            try:
                processed_dirs.append(
                    os.path.join(study_root, "processed_pdo")
                )

                run_test_set(
                    f"{split_type} {study} PDO",
                    split_dir,
                    i,
                    gpu_id,
                    pdo_split,
                    os.path.join(study_root, "processed_pdo"),
                    os.path.join(study_root, "pdo"),
                    ccl_dir
                )
            except Exception as e:
                any_failed = True
                print(
                    f"[ERROR] "
                    f"{split_type} {study} PDO i={i}: {e}",
                    flush=True
                )

            #################################################
            # STUDY CCL
            #################################################

            if study in SEPERATE_STUDIES:

                print(
                    f"[SKIP] "
                    f"{split_type} {study} CCL i={i} "
                    f"(no CCL IDs file for this study)",
                    flush=True
                )

                continue

            study_ccl_split = os.path.join(
                split_dir,
                f"{study}_ccl_ids.txt"
            )

            try:
                processed_dirs.append(
                    os.path.join(study_root, "processed_ccl")
                )

                run_test_set(
                    f"{split_type} {study} CCL",
                    split_dir,
                    i,
                    gpu_id,
                    study_ccl_split,
                    os.path.join(study_root, "processed_ccl"),
                    os.path.join(study_root, "ccl"),
                    ccl_dir
                )
            except Exception as e:
                any_failed = True
                print(
                    f"[ERROR] "
                    f"{split_type} {study} CCL i={i}: {e}",
                    flush=True
                )

        #####################################################
        # SUCCESS
        #####################################################

        if any_failed:
            print(
                f"[PIPELINE COMPLETE WITH ERRORS] "
                f"{split_type} i={i}",
                flush=True
            )
        else:
            print(
                f"[PIPELINE COMPLETE] "
                f"{split_type} i={i}",
                flush=True
            )

    except Exception as e:

        #####################################################
        # ERROR
        #####################################################

        print(
            f"[ERROR] "
            f"{split_type} i={i}: {e}",
            flush=True
        )

    finally:

        #####################################################
        # SAFETY CLEANUP
        #####################################################
        #
        # Normally each set was already deleted immediately
        # after its inference stage.
        #
        # This final cleanup is important if something fails
        # halfway through.
        #
        # Calling the cleanup again is safe because it only
        # removes files that currently exist.
        #####################################################

        print(
            f"[FINAL SAFETY CLEANUP START] "
            f"i={i}",
            flush=True
        )

        for processed_dir in processed_dirs:
            try:
                remove_preprocessing_pt_files(
                    processed_dir
                )
            except Exception as cleanup_error:
                print(
                    f"[CLEANUP ERROR] "
                    f"cleanup failed for "
                    f"{processed_dir}: "
                    f"{cleanup_error}",
                    flush=True
                )

        print(
            f"[FINAL SAFETY CLEANUP DONE] "
            f"i={i}",
            flush=True
        )

        #####################################################
        # RETURN GPU SLOT
        #####################################################

        if gpu_id is not None:
            gpu_queue.put(gpu_id)

        print(
            f"[DONE] "
            f"{split_type} i={i}",
            flush=True
        )


#############################################
# MAIN
#############################################

def main():

    jobs = list(
        itertools.product(
            SPLIT_TYPES,
            range(NUM_SPLITS)
        )
    )

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

    with ProcessPoolExecutor(
        max_workers=len(GPUS)
    ) as executor:

        futures = [
            executor.submit(
                run_pipeline,
                *job
            )
            for job in jobs
        ]

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()