import os
import re
import sys
import time
import traceback
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# Hardcoded experiment configuration
# =============================================================================

HIDRA_REPO_DIR = Path("models/HiDRA")
RAW_INPUT_DIR = Path("pdo_drp_data")
SPLIT_DIR = RAW_INPUT_DIR / "splits" / "random"

EXPERIMENT_ROOT = Path("HiDRA_joint_training_results")

PREPROCESS_SCRIPT = HIDRA_REPO_DIR / "HiDRA_preprocess_improve.py"
TRAIN_SCRIPT = HIDRA_REPO_DIR / "HiDRA_train_improve.py"
INFER_SCRIPT = HIDRA_REPO_DIR / "HiDRA_infer_improve.py"

PYTHON_EXE = sys.executable

GPUS = list(range(7))
CELL_LINE_SPLITS = list(range(10))


STUDIES = [
    "IMPROVE dataset 1",
    "IMPROVE dataset 2",
    "IMPROVE dataset 3",
    "Pilot_1"
]



# Resume behavior:
# False = skip complete fold-runs that already have RUN_COMPLETE.marker
# True  = rerun everything, even completed runs
FORCE_RERUN = False

# Training params
EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.001
PATIENCE = 20
LOSS = "mse"

# HiDRA input config constants
X_DATA_CANC_FILES = '[["cancer_gene_expression.tsv", ["Gene_Symbol"]]]'
X_DATA_DRUG_FILES = '[["drug_ecfp4_nbits512.tsv"]]'
Y_DATA_FILES = '[["response.tsv"]]'
INPUT_SUPP_DATA_DIR = "./author_data"
DATA_FORMAT = ".csv"
MODEL_FILE_NAME = "model"
MODEL_FILE_FORMAT = ".h5"

MASTER_LOG = EXPERIMENT_ROOT / "experiment.log"


# =============================================================================
# Thread-safe logging helpers
# =============================================================================

print_lock = threading.Lock()
master_log_lock = threading.Lock()


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def write_master_log(message):
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp()}] {message}\n"

    with master_log_lock:
        with MASTER_LOG.open("a", encoding="utf-8") as f:
            f.write(line)

    with print_lock:
        print(line, end="", flush=True)


def append_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


# =============================================================================
# Config generation
# =============================================================================

def write_hidra_config(
    config_path,
    train_split_file,
    val_split_file,
    test_split_file,
    preprocess_output_dir,
    train_input_dir,
    train_output_dir,
    infer_input_data_dir,
    infer_input_model_dir,
    infer_output_dir,
):
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_text = f"""[Global_Params]
model_name = hidra

[Preprocess]
train_split_file = {train_split_file}
val_split_file = {val_split_file}
test_split_file = {test_split_file}
x_data_canc_files = {X_DATA_CANC_FILES}
x_data_drug_files = {X_DATA_DRUG_FILES}
y_data_files = {Y_DATA_FILES}
input_supp_data_dir = {INPUT_SUPP_DATA_DIR}
data_format = {DATA_FORMAT}
input_dir = {RAW_INPUT_DIR}
output_dir = {preprocess_output_dir}

[Train]
model_file_name = {MODEL_FILE_NAME}
model_file_format = {MODEL_FILE_FORMAT}
epochs = {EPOCHS}
batch_size = {BATCH_SIZE}
learning_rate = {LEARNING_RATE}
patience = {PATIENCE}
loss = {LOSS}
data_format = {DATA_FORMAT}
input_dir = {train_input_dir}
output_dir = {train_output_dir}

[Infer]
model_file_name = {MODEL_FILE_NAME}
model_file_format = {MODEL_FILE_FORMAT}
data_format = {DATA_FORMAT}
input_data_dir = {infer_input_data_dir}
input_model_dir = {infer_input_model_dir}
output_dir = {infer_output_dir}
calc_infer_scores = True
"""

    config_path.write_text(config_text, encoding="utf-8")


# =============================================================================
# Fold discovery
# =============================================================================

def discover_pdo_splits_for_study(study):
    """
    Finds PDO fold j values using files like:
      {study}_split_{j}_val_ids.txt
      {study}_split_{j}_test_ids.txt

    Returns sorted list of available j values where both val and test exist.
    """
    found = set()

    pattern = re.compile(
        rf"^{re.escape(study)}_split_(\d+)_val_ids\.txt$"
    )

    for path in SPLIT_DIR.glob(f"{study}_split_*_val_ids.txt"):
        match = pattern.match(path.name)
        if not match:
            continue

        j = int(match.group(1))
        val_file = SPLIT_DIR / f"{study}_split_{j}_val_ids.txt"
        test_file = SPLIT_DIR / f"{study}_split_{j}_test_ids.txt"

        if val_file.exists() and test_file.exists():
            found.add(j)

    return sorted(found)


def study_requires_ccl_file(study):
    """
    pilot_1 study do not have {study}_ccl_ids.txt in this split directory.
    For pilot_1, skip only the study-level CCL inference stage.
    For non-pilot_1 studies, keep {study}_ccl_ids.txt as required.
    """
    return not study.startswith("Pilot_1")


def build_jobs():
    """
    Each job is one full trained model workflow:
      - base preprocess/train/infer using PDO test set
      - optionally preprocess/infer on {study}_ccl_ids.txt using same trained model
      - preprocess/infer on ccl_split_{i}_test_ids.txt using same trained model

    The ThreadPoolExecutor schedules these jobs across 8 GPUs.
    """
    jobs = []

    for study in STUDIES:
        pdo_splits = discover_pdo_splits_for_study(study)

        if not pdo_splits:
            write_master_log(f"WARNING: no PDO folds discovered for study={study}")
            continue

        require_study_ccl_file = study_requires_ccl_file(study)
        study_ccl_file = SPLIT_DIR / f"{study}_ccl_ids.txt"

        if not study_ccl_file.exists():
            if require_study_ccl_file:
                write_master_log(f"WARNING: missing required study CCL test file: {study_ccl_file}")
            else:
                write_master_log(
                    "INFO: optional study CCL test file missing; "
                    f"will skip study-level CCL inference for study={study}: {study_ccl_file}"
                )
            study_ccl_file = None

        for j in pdo_splits:
            val_file = SPLIT_DIR / f"{study}_split_{j}_val_ids.txt"
            pdo_test_file = SPLIT_DIR / f"{study}_split_{j}_test_ids.txt"

            for i in CELL_LINE_SPLITS:
                train_file = SPLIT_DIR / f"{study}_split_{j}_ccl_split_{i}_train_ids.txt"
                ccl_test_file = SPLIT_DIR / f"ccl_split_{i}_test_ids.txt"

                required_files = [
                    train_file,
                    val_file,
                    pdo_test_file,
                    ccl_test_file,
                ]

                if require_study_ccl_file:
                    required_files.append(study_ccl_file)

                missing = []
                for required_file in required_files:
                    if required_file is None or not required_file.exists():
                        missing.append(str(required_file))

                if missing:
                    write_master_log(
                        "WARNING: skipping missing split files for "
                        f"study={study}, pdo_split={j}, ccl_split={i}: "
                        + "; ".join(missing)
                    )
                    continue

                run_id = f"{safe_name(study)}__pdo_split_{j}__ccl_split_{i}"
                run_dir = EXPERIMENT_ROOT / safe_name(study) / f"pdo_split_{j}" / f"ccl_split_{i}"

                jobs.append(
                    {
                        "study": study,
                        "pdo_split": j,
                        "ccl_split": i,
                        "run_id": run_id,
                        "run_dir": run_dir,
                        "train_file": train_file,
                        "val_file": val_file,
                        "pdo_test_file": pdo_test_file,
                        "study_ccl_file": study_ccl_file,
                        "ccl_test_file": ccl_test_file,
                    }
                )

    return jobs


# =============================================================================
# Process execution
# =============================================================================

def run_command(command, gpu_id, log_file, run_id, stage_name):
    """
    Runs a command with CUDA_VISIBLE_DEVICES pinned to one GPU.
    Streams stdout/stderr to both terminal and a per-run logfile.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    header = (
        "\n"
        + "=" * 100
        + "\n"
        + f"[{timestamp()}] START stage={stage_name} run={run_id} gpu={gpu_id}\n"
        + f"Command: {' '.join(map(str, command))}\n"
        + f"Working directory: {HIDRA_REPO_DIR}\n"
        + f"CUDA_VISIBLE_DEVICES={gpu_id}\n"
        + "=" * 100
        + "\n"
    )
    append_text(log_file, header)

    with print_lock:
        print(header, end="", flush=True)

    process = subprocess.Popen(
        [str(x) for x in command],
        cwd=str(HIDRA_REPO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        prefixed_line = f"[{run_id} | GPU {gpu_id} | {stage_name}] {line}"

        append_text(log_file, prefixed_line)

        with print_lock:
            print(prefixed_line, end="", flush=True)

    return_code = process.wait()

    footer = (
        "\n"
        + "=" * 100
        + "\n"
        + f"[{timestamp()}] END stage={stage_name} run={run_id} gpu={gpu_id} return_code={return_code}\n"
        + "=" * 100
        + "\n"
    )
    append_text(log_file, footer)

    with print_lock:
        print(footer, end="", flush=True)

    if return_code != 0:
        raise RuntimeError(
            f"Stage failed: run={run_id}, stage={stage_name}, gpu={gpu_id}, return_code={return_code}"
        )


def run_preprocess(config_file, gpu_id, log_file, run_id, stage_name):
    run_command(
        [PYTHON_EXE, PREPROCESS_SCRIPT, "--config_file", config_file],
        gpu_id=gpu_id,
        log_file=log_file,
        run_id=run_id,
        stage_name=stage_name,
    )


def run_train(config_file, gpu_id, log_file, run_id):
    run_command(
        [PYTHON_EXE, TRAIN_SCRIPT, "--config_file", config_file],
        gpu_id=gpu_id,
        log_file=log_file,
        run_id=run_id,
        stage_name="train",
    )


def run_infer(config_file, gpu_id, log_file, run_id, stage_name):
    run_command(
        [PYTHON_EXE, INFER_SCRIPT, "--config_file", config_file],
        gpu_id=gpu_id,
        log_file=log_file,
        run_id=run_id,
        stage_name=stage_name,
    )


# =============================================================================
# One full fold workflow
# =============================================================================

def run_one_job(job, gpu_id):
    study = job["study"]
    j = job["pdo_split"]
    i = job["ccl_split"]
    run_id = job["run_id"]
    run_dir = job["run_dir"]

    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "terminal_output.log"
    complete_marker = run_dir / "RUN_COMPLETE.marker"
    failed_marker = run_dir / "RUN_FAILED.marker"

    if complete_marker.exists() and not FORCE_RERUN:
        write_master_log(f"SKIP already complete: run={run_id}")
        return {
            "run_id": run_id,
            "gpu_id": gpu_id,
            "status": "skipped_complete",
        }

    if failed_marker.exists():
        failed_marker.unlink()

    write_master_log(
        f"START run={run_id} gpu={gpu_id} study={study} pdo_split={j} ccl_split={i}"
    )

    append_text(
        log_file,
        "\n"
        + "#" * 100
        + "\n"
        + f"[{timestamp()}] RUN START\n"
        + f"study={study}\n"
        + f"pdo_split={j}\n"
        + f"ccl_split={i}\n"
        + f"gpu={gpu_id}\n"
        + f"run_dir={run_dir}\n"
        + "#" * 100
        + "\n",
    )

    try:
        # ---------------------------------------------------------------------
        # Base PDO test run:
        # preprocess -> train -> infer
        # ---------------------------------------------------------------------
        base_config = run_dir / "config_base_pdo_test.txt"

        write_hidra_config(
            config_path=base_config,
            train_split_file=job["train_file"],
            val_split_file=job["val_file"],
            test_split_file=job["pdo_test_file"],
            preprocess_output_dir=run_dir,
            train_input_dir=run_dir,
            train_output_dir=run_dir,
            infer_input_data_dir=run_dir,
            infer_input_model_dir=run_dir,
            infer_output_dir=run_dir,
        )

        run_preprocess(
            config_file=base_config,
            gpu_id=gpu_id,
            log_file=log_file,
            run_id=run_id,
            stage_name="preprocess_pdo_test",
        )

        run_train(
            config_file=base_config,
            gpu_id=gpu_id,
            log_file=log_file,
            run_id=run_id,
        )

        run_infer(
            config_file=base_config,
            gpu_id=gpu_id,
            log_file=log_file,
            run_id=run_id,
            stage_name="infer_pdo_test",
        )

        # ---------------------------------------------------------------------
        # Additional inference test set 1:
        # {study}_ccl_ids.txt
        #
        # Some studies, including pilot_1, do not have this file.
        # If it is missing and marked optional, skip only this inference stage.
        # ---------------------------------------------------------------------
        if job["study_ccl_file"] is not None:
            study_ccl_dir = run_dir / f"{safe_name(study)}_ccl_ids"
            study_ccl_config = run_dir / "config_infer_study_ccl_ids.txt"

            write_hidra_config(
                config_path=study_ccl_config,
                train_split_file=job["train_file"],
                val_split_file=job["val_file"],
                test_split_file=job["study_ccl_file"],
                preprocess_output_dir=study_ccl_dir,
                train_input_dir=study_ccl_dir,
                train_output_dir=study_ccl_dir,
                infer_input_data_dir=study_ccl_dir,
                infer_input_model_dir=run_dir,
                infer_output_dir=study_ccl_dir,
            )

            run_preprocess(
                config_file=study_ccl_config,
                gpu_id=gpu_id,
                log_file=log_file,
                run_id=run_id,
                stage_name="preprocess_study_ccl_ids",
            )

            run_infer(
                config_file=study_ccl_config,
                gpu_id=gpu_id,
                log_file=log_file,
                run_id=run_id,
                stage_name="infer_study_ccl_ids",
            )
        else:
            write_master_log(
                f"SKIP optional study_ccl_ids inference: run={run_id} study={study}"
            )

        # ---------------------------------------------------------------------
        # Additional inference test set 2:
        # ccl_split_{i}_test_ids.txt
        #
        # preprocess into subdir, infer using original trained model
        # ---------------------------------------------------------------------
        ccl_test_dir = run_dir / f"ccl_split_{i}_test_ids"
        ccl_test_config = run_dir / "config_infer_ccl_split_test_ids.txt"

        write_hidra_config(
            config_path=ccl_test_config,
            train_split_file=job["train_file"],
            val_split_file=job["val_file"],
            test_split_file=job["ccl_test_file"],
            preprocess_output_dir=ccl_test_dir,
            train_input_dir=ccl_test_dir,
            train_output_dir=ccl_test_dir,
            infer_input_data_dir=ccl_test_dir,
            infer_input_model_dir=run_dir,
            infer_output_dir=ccl_test_dir,
        )

        run_preprocess(
            config_file=ccl_test_config,
            gpu_id=gpu_id,
            log_file=log_file,
            run_id=run_id,
            stage_name="preprocess_ccl_split_test_ids",
        )

        run_infer(
            config_file=ccl_test_config,
            gpu_id=gpu_id,
            log_file=log_file,
            run_id=run_id,
            stage_name="infer_ccl_split_test_ids",
        )

        complete_marker.write_text(
            f"completed_at={timestamp()}\n"
            f"study={study}\n"
            f"pdo_split={j}\n"
            f"ccl_split={i}\n"
            f"gpu={gpu_id}\n",
            encoding="utf-8",
        )

        write_master_log(f"COMPLETE run={run_id} gpu={gpu_id}")

        return {
            "run_id": run_id,
            "gpu_id": gpu_id,
            "status": "complete",
        }

    except Exception as exc:
        error_text = (
            "\n"
            + "!" * 100
            + "\n"
            + f"[{timestamp()}] RUN FAILED\n"
            + f"run={run_id}\n"
            + f"gpu={gpu_id}\n"
            + f"error={repr(exc)}\n"
            + traceback.format_exc()
            + "\n"
            + "!" * 100
            + "\n"
        )

        append_text(log_file, error_text)

        failed_marker.write_text(
            f"failed_at={timestamp()}\n"
            f"study={study}\n"
            f"pdo_split={j}\n"
            f"ccl_split={i}\n"
            f"gpu={gpu_id}\n"
            f"error={repr(exc)}\n",
            encoding="utf-8",
        )

        write_master_log(f"FAILED run={run_id} gpu={gpu_id} error={repr(exc)}")

        return {
            "run_id": run_id,
            "gpu_id": gpu_id,
            "status": "failed",
            "error": repr(exc),
        }


# =============================================================================
# GPU scheduler
# =============================================================================

def worker_loop(gpu_id, jobs_for_worker, results):
    for job in jobs_for_worker:
        result = run_one_job(job, gpu_id)
        results.append(result)


def distribute_jobs_round_robin(jobs, gpu_ids):
    buckets = {gpu_id: [] for gpu_id in gpu_ids}

    for idx, job in enumerate(jobs):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        buckets[gpu_id].append(job)

    return buckets


def main():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    write_master_log("=" * 100)
    write_master_log("PDO Phase 2 experiment starting")
    write_master_log(f"HIDRA_REPO_DIR={HIDRA_REPO_DIR}")
    write_master_log(f"RAW_INPUT_DIR={RAW_INPUT_DIR}")
    write_master_log(f"SPLIT_DIR={SPLIT_DIR}")
    write_master_log(f"EXPERIMENT_ROOT={EXPERIMENT_ROOT}")
    write_master_log(f"GPUS={GPUS}")
    write_master_log(f"FORCE_RERUN={FORCE_RERUN}")

    for required_path in [HIDRA_REPO_DIR, RAW_INPUT_DIR, SPLIT_DIR, PREPROCESS_SCRIPT, TRAIN_SCRIPT, INFER_SCRIPT]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path does not exist: {required_path}")

    jobs = build_jobs()

    write_master_log(f"Discovered runnable jobs: {len(jobs)}")

    if not jobs:
        write_master_log("No jobs found. Exiting.")
        return

    job_buckets = distribute_jobs_round_robin(jobs, GPUS)

    for gpu_id in GPUS:
        write_master_log(f"GPU {gpu_id}: assigned {len(job_buckets[gpu_id])} jobs")

    results = []

    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = []
        for gpu_id in GPUS:
            futures.append(
                executor.submit(worker_loop, gpu_id, job_buckets[gpu_id], results)
            )

        for future in as_completed(futures):
            future.result()

    complete = sum(1 for r in results if r["status"] == "complete")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped_complete")

    write_master_log("=" * 100)
    write_master_log("PDO Phase 2 experiment finished")
    write_master_log(f"complete={complete}")
    write_master_log(f"failed={failed}")
    write_master_log(f"skipped_complete={skipped}")
    write_master_log(f"total_results={len(results)}")
    write_master_log("=" * 100)


if __name__ == "__main__":
    main()