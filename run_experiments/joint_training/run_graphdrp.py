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


BASE_SPLIT_DIR = "pdo_drp_data/splits/random"
RAW_DATA_DIR = "pdo_drp_data"
BASE_OUTPUT_DIR = "joint_training_results_graphdrp"

NUM_GPUS = 8
SLOTS_PER_GPU = 4

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

#############################################
# ACTIVE RUN CONTROL
#############################################

# Maximum number of active (study, i, j) pipelines.
#
# A slot is acquired before preprocessing starts and is
# released only after the entire run has completed and
# cleanup has finished.
MAX_ACTIVE_FOLDS = 32


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
# SHARED STATE
#############################################

manager = Manager()

gpu_queue = manager.Queue()

for gpu_id in range(NUM_GPUS):
    for _ in range(SLOTS_PER_GPU):
        gpu_queue.put(gpu_id)

memory_slots = manager.Semaphore(MAX_ACTIVE_FOLDS)

#############################################
# UTILITIES
#############################################

def run_cmd(cmd):
    """
    Run a subprocess and raise an exception if it fails.
    """

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
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
# CORE PIPELINE
#############################################

def run_pipeline(study, i, j):

    #########################################################
    # Acquire one of the 32 active-run slots.
    #
    # No preprocessing for this job starts before this
    # succeeds.
    #
    # The slot is released only after final cleanup.
    #########################################################

    memory_slots.acquire()

    gpu_id = None

    # These are initialized here so the finally block can
    # safely clean up after a failure at any pipeline stage.
    ccl_dir = None
    study_dir = None
    mixed_dir = None

    try:

        #####################################################
        # GET GPU SLOT
        #####################################################

        gpu_id = gpu_queue.get()

        #####################################################
        # RUN DIRECTORY
        #####################################################

        run_dir = os.path.join(
            BASE_OUTPUT_DIR,
            f"{study}_i{i}_j{j}"
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        print(
            f"[START] "
            f"{study} i={i} j={j} "
            f"GPU={gpu_id}",
            flush=True
        )

        #####################################################
        # COMMON SPLITS
        #####################################################

        train_split = os.path.join(
            BASE_SPLIT_DIR,
            f"{study}_split_{j}_ccl_split_{i}_train_ids.txt"
        )

        val_split = os.path.join(
            BASE_SPLIT_DIR,
            f"{study}_split_{j}_val_ids.txt"
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

        ccl_test_split = os.path.join(
            BASE_SPLIT_DIR,
            f"ccl_split_{i}_test_ids.txt"
        )

        #####################################################
        # CCL PREPROCESS
        #####################################################

        print(
            f"[CCL PREPROCESS START] "
            f"{study} i={i} j={j}",
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
        ])

        print(
            f"[CCL PREPROCESS DONE] "
            f"{study} i={i} j={j}",
            flush=True
        )

        #####################################################
        # TRAIN MODEL
        #####################################################

        print(
            f"[TRAIN START] "
            f"{study} i={i} j={j} "
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
            f"cuda:{gpu_id}"
        ])

        print(
            f"[TRAIN DONE] "
            f"{study} i={i} j={j}",
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
            f"{study} i={i} j={j}",
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
            f"cuda:{gpu_id}",

            "--calc_infer_scores",
            "true"
        ])

        print(
            f"[CCL INFER DONE] "
            f"{study} i={i} j={j}",
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

        print(
            f"[CCL CLEANUP START] "
            f"{study} i={i} j={j}",
            flush=True
        )

        remove_preprocessing_pt_files(
            ccl_dir
        )

        print(
            f"[CCL CLEANUP DONE] "
            f"{study} i={i} j={j}",
            flush=True
        )

        #####################################################
        #####################################################
        # 2. STUDY
        #####################################################
        #####################################################

        # Study preprocessing does not begin until the CCL
        # .pt files have been deleted.

        study_dir = os.path.join(
            run_dir,
            "processed_study"
        )

        os.makedirs(
            study_dir,
            exist_ok=True
        )

        study_split = os.path.join(
            BASE_SPLIT_DIR,
            f"{study}_split_{j}_test_ids.txt"
        )

        #####################################################
        # STUDY PREPROCESS
        #####################################################

        print(
            f"[STUDY PREPROCESS START] "
            f"{study} i={i} j={j}",
            flush=True
        )

        run_cmd([
            "python",
            PREPROCESS_SCRIPT,

            "--input_dir",
            RAW_DATA_DIR,

            "--output_dir",
            study_dir,

            "--train_split_file",
            train_split,

            "--val_split_file",
            val_split,

            "--test_split_file",
            study_split
        ])

        print(
            f"[STUDY PREPROCESS DONE] "
            f"{study} i={i} j={j}",
            flush=True
        )

        #####################################################
        # STUDY INFERENCE
        #####################################################

        study_out = os.path.join(
            run_dir,
            "study"
        )

        os.makedirs(
            study_out,
            exist_ok=True
        )

        print(
            f"[STUDY INFER START] "
            f"{study} i={i} j={j}",
            flush=True
        )

        run_cmd([
            "python",
            INFER_SCRIPT,

            "--input_data_dir",
            study_dir,

            # Reuse the model trained during the CCL stage.
            "--input_model_dir",
            ccl_dir,

            "--output_dir",
            study_out,

            "--cuda_name",
            f"cuda:{gpu_id}",

            "--calc_infer_scores",
            "true"
        ])

        print(
            f"[STUDY INFER DONE] "
            f"{study} i={i} j={j}",
            flush=True
        )

        #####################################################
        # STUDY CLEANUP
        #
        # Delete its three tensors BEFORE mixed
        # preprocessing starts.
        #####################################################

        print(
            f"[STUDY CLEANUP START] "
            f"{study} i={i} j={j}",
            flush=True
        )

        remove_preprocessing_pt_files(
            study_dir
        )

        print(
            f"[STUDY CLEANUP DONE] "
            f"{study} i={i} j={j}",
            flush=True
        )

        #####################################################
        #####################################################
        # 3. MIXED
        #####################################################
        #####################################################

        if study not in SEPERATE_STUDIES:

            # Mixed preprocessing does not begin until the
            # Study .pt files have been deleted.

            mixed_dir = os.path.join(
                run_dir,
                "processed_mixed"
            )

            os.makedirs(
                mixed_dir,
                exist_ok=True
            )

            mixed_split = os.path.join(
                BASE_SPLIT_DIR,
                f"{study}_ccl_ids.txt"
            )

            #################################################
            # MIXED PREPROCESS
            #################################################

            print(
                f"[MIXED PREPROCESS START] "
                f"{study} i={i} j={j}",
                flush=True
            )

            run_cmd([
                "python",
                PREPROCESS_SCRIPT,

                "--input_dir",
                RAW_DATA_DIR,

                "--output_dir",
                mixed_dir,

                "--train_split_file",
                train_split,

                "--val_split_file",
                val_split,

                "--test_split_file",
                mixed_split
            ])

            print(
                f"[MIXED PREPROCESS DONE] "
                f"{study} i={i} j={j}",
                flush=True
            )

            #################################################
            # MIXED INFERENCE
            #################################################

            mixed_out = os.path.join(
                run_dir,
                "mixed"
            )

            os.makedirs(
                mixed_out,
                exist_ok=True
            )

            print(
                f"[MIXED INFER START] "
                f"{study} i={i} j={j}",
                flush=True
            )

            run_cmd([
                "python",
                INFER_SCRIPT,

                "--input_data_dir",
                mixed_dir,

                # Reuse the trained CCL model.
                "--input_model_dir",
                ccl_dir,

                "--output_dir",
                mixed_out,

                "--cuda_name",
                f"cuda:{gpu_id}",

                "--calc_infer_scores",
                "true"
            ])

            print(
                f"[MIXED INFER DONE] "
                f"{study} i={i} j={j}",
                flush=True
            )

            #################################################
            # MIXED CLEANUP
            #################################################

            print(
                f"[MIXED CLEANUP START] "
                f"{study} i={i} j={j}",
                flush=True
            )

            remove_preprocessing_pt_files(
                mixed_dir
            )

            print(
                f"[MIXED CLEANUP DONE] "
                f"{study} i={i} j={j}",
                flush=True
            )

        #####################################################
        # SUCCESS
        #####################################################

        print(
            f"[PIPELINE COMPLETE] "
            f"{study} i={i} j={j}",
            flush=True
        )

    except Exception as e:

        #####################################################
        # ERROR
        #####################################################

        print(
            f"[ERROR] "
            f"{study} i={i} j={j}: {e}",
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
        # Example:
        #
        #   Study preprocess succeeds
        #   Study inference crashes
        #
        # Without this block, those three .pt files would
        # remain on disk.
        #
        # Calling the cleanup again is safe because it only
        # removes files that currently exist.
        #####################################################

        print(
            f"[FINAL SAFETY CLEANUP START] "
            f"{study} i={i} j={j}",
            flush=True
        )

        try:
            remove_preprocessing_pt_files(
                ccl_dir
            )
        except Exception as cleanup_error:
            print(
                f"[CLEANUP ERROR] "
                f"CCL cleanup failed for "
                f"{study} i={i} j={j}: "
                f"{cleanup_error}",
                flush=True
            )

        try:
            remove_preprocessing_pt_files(
                study_dir
            )
        except Exception as cleanup_error:
            print(
                f"[CLEANUP ERROR] "
                f"Study cleanup failed for "
                f"{study} i={i} j={j}: "
                f"{cleanup_error}",
                flush=True
            )

        try:
            remove_preprocessing_pt_files(
                mixed_dir
            )
        except Exception as cleanup_error:
            print(
                f"[CLEANUP ERROR] "
                f"Mixed cleanup failed for "
                f"{study} i={i} j={j}: "
                f"{cleanup_error}",
                flush=True
            )

        print(
            f"[FINAL SAFETY CLEANUP DONE] "
            f"{study} i={i} j={j}",
            flush=True
        )

        #####################################################
        # RETURN GPU SLOT
        #####################################################

        if gpu_id is not None:
            gpu_queue.put(gpu_id)

        #####################################################
        # RELEASE ACTIVE-RUN SLOT
        #
        # This intentionally happens AFTER all cleanup.
        #####################################################

        memory_slots.release()

        print(
            f"[DONE] "
            f"{study} i={i} j={j}",
            flush=True
        )


#############################################
# MAIN
#############################################

def main():

    jobs = list(
        itertools.product(
            STUDIES,
            range(10),
            range(10)
        )
    )

    total_gpu_slots = (
        NUM_GPUS * SLOTS_PER_GPU
    )

    print(
        f"Total jobs: {len(jobs)}",
        flush=True
    )

    print(
        f"GPUs: {NUM_GPUS}",
        flush=True
    )

    print(
        f"Slots per GPU: {SLOTS_PER_GPU}",
        flush=True
    )

    print(
        f"Total GPU slots: {total_gpu_slots}",
        flush=True
    )

    print(
        f"Maximum active folds: "
        f"{MAX_ACTIVE_FOLDS}",
        flush=True
    )

    with ProcessPoolExecutor(
        max_workers=total_gpu_slots
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

