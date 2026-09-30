import os
import subprocess
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import re

#############################################
# CONFIG
#############################################

STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1"
]


BASE_SPLIT_DIR = "pdo_drp_data/splits/cancer-blind"
RAW_DATA_DIR = "pdo_drp_data"
BASE_OUTPUT_DIR = "./LGBM_Joint_Training_Results"

MAX_PARALLEL_JOBS = 6
START_FROM = None  # (study, i, j)

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
# SPECIAL CASES
#############################################

PILOT1_STUDIES = {
    "Pilot_1"
}

#############################################
# UTILITY
#############################################

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def job_fully_complete(run_dir, study):
    required_dirs = ["ccl", "study"]

    if study not in PILOT1_STUDIES:
        required_dirs.append("mixed")

    for d in required_dirs:
        if not os.path.exists(os.path.join(run_dir, d, "test_scores.json")):
            return False

    return True

def cleanup_run_dir(run_dir):
    keep_files = {
        "model.txt",
        "val_scores.json",
        "param_log_file.txt",
        "val_y_data_predicted.csv"
    }

    keep_dirs = {"ccl", "study", "mixed"}

    for item in os.listdir(run_dir):
        path = os.path.join(run_dir, item)

        if item in keep_files:
            continue

        if item in keep_dirs:
            continue

        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

#############################################
# SPLIT DETECTION (FIXED)
#############################################

def get_indices_for_study(study):
    files = os.listdir(BASE_SPLIT_DIR)

    i_vals = set()
    j_vals = set()

    pattern = re.compile(
        rf"^{re.escape(study)}_split_(\d+)_ccl_split_(\d+)_train_ids\.txt$"
    )

    for f in files:
        m = pattern.match(f)
        if m:
            j_vals.add(int(m.group(1)))
            i_vals.add(int(m.group(2)))

    i_vals = sorted(i_vals)
    j_vals = sorted(j_vals)

    if not i_vals or not j_vals:
        print(f"[WARNING] No splits found for {study}", flush=True)

    return i_vals, j_vals

#############################################
# CORE PIPELINE
#############################################

def run_pipeline(study, i, j):

    try:
        run_dir = f"{BASE_OUTPUT_DIR}/{study}_i{i}_j{j}"
        os.makedirs(run_dir, exist_ok=True)

        # Skip if already complete
        if job_fully_complete(run_dir, study):
            print(f"[SKIP DONE] {study} i={i} j={j}", flush=True)
            return

        print(f"[START] {study} i={i} j={j}", flush=True)

        #########################################
        # SPLITS
        #########################################

        train_split = f"{BASE_SPLIT_DIR}/{study}_split_{j}_ccl_split_{i}_train_ids.txt"
        val_split   = f"{BASE_SPLIT_DIR}/{study}_split_{j}_val_ids.txt"
        base_test_split = f"{BASE_SPLIT_DIR}/{study}_split_{j}_test_ids.txt"

        #########################################
        # STEP 1 — PREPROCESS
        #########################################

        run_cmd([
            "python", PREPROCESS_SCRIPT,
            "--input_dir", RAW_DATA_DIR,
            "--output_dir", run_dir,
            "--train_split_file", train_split,
            "--val_split_file", val_split,
            "--test_split_file", base_test_split
        ])

        #########################################
        # STEP 2 — TRAIN
        #########################################

        run_cmd([
            "python", TRAIN_SCRIPT,
            "--input_dir", run_dir,
            "--output_dir", run_dir
        ])

        #########################################
        # STEP 3 — INFERENCE
        #########################################

        test_variants = {
            "ccl":   f"{BASE_SPLIT_DIR}/ccl_split_{i}_test_ids.txt",
            "study": f"{BASE_SPLIT_DIR}/{study}_split_{j}_test_ids.txt",
        }

        if study not in PILOT1_STUDIES:
            test_variants["mixed"] = f"{BASE_SPLIT_DIR}/{study}_ccl_ids.txt"

        for name, test_split in test_variants.items():

            out_dir = os.path.join(run_dir, name)
            os.makedirs(out_dir, exist_ok=True)

            if os.path.exists(os.path.join(out_dir, "test_scores.json")):
                print(f"[SKIP INFER-{name}] {study} i={i} j={j}", flush=True)
                continue

            print(f"[INFER-{name}] {study} i={i} j={j}", flush=True)

            run_cmd([
                "python", PREPROCESS_SCRIPT,
                "--input_dir", RAW_DATA_DIR,
                "--output_dir", run_dir,
                "--train_split_file", train_split,
                "--val_split_file", val_split,
                "--test_split_file", test_split
            ])

            run_cmd([
                "python", INFER_SCRIPT,
                "--input_data_dir", run_dir,
                "--input_model_dir", run_dir,
                "--output_dir", out_dir,
                "--calc_infer_scores", "true"
            ])

        print(f"[DONE] {study} i={i} j={j}", flush=True)

        #########################################
        # CLEANUP
        #########################################

        if job_fully_complete(run_dir, study):
            print(f"[CLEANUP] {study} i={i} j={j}", flush=True)
            cleanup_run_dir(run_dir)
        else:
            print(f"[SKIP CLEANUP — INCOMPLETE] {study} i={i} j={j}", flush=True)

    except Exception as e:
        print(f"[ERROR] {study} i={i} j={j}: {e}", flush=True)

#############################################
# MAIN
#############################################

def main():

    jobs = []

    for study in STUDIES:
        i_vals, j_vals = get_indices_for_study(study)

        jobs.extend(
            (study, i, j)
            for i in i_vals
            for j in j_vals
        )

    if START_FROM is not None:
        if START_FROM in jobs:
            start_idx = jobs.index(START_FROM)
            jobs = jobs[start_idx:]
        else:
            raise ValueError("START_FROM tuple not found in job list")

    print(f"Total jobs: {len(jobs)}")
    print(f"Running at most {MAX_PARALLEL_JOBS} jobs at a time")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        futures = [executor.submit(run_pipeline, *job) for job in jobs]

        for f in as_completed(futures):
            f.result()

if __name__ == "__main__":
    main()

