import os
import re
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
BASE_OUTPUT_DIR = "cell_line_only_results_igtd"

NUM_SPLITS = 10

IGTD_DIR = "models/IGTD"

# IGTD commands run from inside IGTD_DIR, so all paths are
# converted to absolute paths to stay valid there.
SPLITS_ROOT = os.path.abspath(SPLITS_ROOT)
RAW_DATA_DIR = os.path.abspath(RAW_DATA_DIR)
BASE_OUTPUT_DIR = os.path.abspath(BASE_OUTPUT_DIR)
IGTD_DIR = os.path.abspath(IGTD_DIR)

PREPROCESS_SCRIPT = os.path.join(
    IGTD_DIR,
    "IGTD_preprocess_improve.py"
)

TRAIN_SCRIPT = os.path.join(
    IGTD_DIR,
    "IGTD_train_improve.py"
)

INFER_SCRIPT = os.path.join(
    IGTD_DIR,
    "IGTD_infer_improve.py"
)

# Separate preprocessing used for each inference test set.
PREPROCESS_FOR_INFERENCE_SCRIPT = os.path.join(
    IGTD_DIR,
    "IGTD_preprocess_for_inference_improve.py"
)

MODEL_CONFIG_FILE = os.path.join(
    IGTD_DIR,
    "IGTD_params.txt"
)

# Studies that do NOT have a {study}_ccl_ids.txt file.
# The study CCL inference is skipped for these.
SEPERATE_STUDIES = {
    "Pilot_1"
}

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



#############################################
# TEST SET INFERENCE
#############################################

def run_test_set(label, sub_label, i, gpu_id, test_split, infer_dir, model_dir):
    """
    Preprocess one test set with IGTD's inference preprocessing
    and run inference on it with the model trained for CCL split i.

    Each test set gets a local copy of IGTD_params.txt where
    inference_split_file points at its test IDs file.

    Preprocessing writes into a subdirectory named after
    inference_data_name (== sub_label), so test_data.pkl lives at
    infer_dir/sub_label/test_data.pkl.
    """

    os.makedirs(
        infer_dir,
        exist_ok=True
    )

    #################################################
    # PATCHED PARAMS FILE
    #################################################

    local_params = os.path.join(
        infer_dir,
        "IGTD_params.txt"
    )

    with open(MODEL_CONFIG_FILE) as f:
        params_text = f.read()

    params_text = re.sub(
        r"^(inference_split_file\s*=\s*).*$",
        lambda m: m.group(1) + test_split,
        params_text,
        flags=re.M
    )

    params_text = re.sub(
        r"^(inference_data_name\s*=\s*).*$",
        lambda m: m.group(1) + sub_label,
        params_text,
        flags=re.M
    )

    with open(local_params, "w") as f:
        f.write(params_text)

    #################################################
    # PREPROCESSOR PARAMS FROM TRAINING
    #################################################

    pkl_src = os.path.join(
        model_dir,
        "preprocessor_params.pkl"
    )

    pkl_dst = os.path.join(
        infer_dir,
        "preprocessor_params.pkl"
    )

    if not os.path.isfile(pkl_src):
        raise RuntimeError(
            f"preprocessor_params.pkl not found at {pkl_src} "
            f"-- was training successful?"
        )

    if os.path.lexists(pkl_dst):
        os.remove(pkl_dst)

    os.symlink(
        pkl_src,
        pkl_dst
    )

    #################################################
    # PREPROCESS
    #################################################

    print(
        f"[{label} PREPROCESS START] "
        f"i={i}",
        flush=True
    )

    # Pass --inference_data_name as a CLI flag so it takes
    # effect even if improvelib treats it as a locked param
    # that ignores the config file value.
    run_cmd([
        "python",
        PREPROCESS_FOR_INFERENCE_SCRIPT,

        "--config_file",
        local_params,

        "--input_dir",
        RAW_DATA_DIR,

        "--output_dir",
        infer_dir,

        "--inference_data_name",
        sub_label
    ], env=job_env(gpu_id, i), cwd=IGTD_DIR)

    infer_data_dir = os.path.join(
        infer_dir,
        sub_label
    )

    if not os.path.isfile(
        os.path.join(infer_data_dir, "test_data.pkl")
    ):
        raise RuntimeError(
            f"Preprocessing succeeded but test_data.pkl "
            f"not found in {infer_data_dir}"
        )

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

        "--config_file",
        local_params,

        "--input_data_dir",
        infer_data_dir,

        # Reuse the model trained during the CCL stage.
        "--input_model_dir",
        model_dir,

        "--output_dir",
        infer_dir,

        "--calc_infer_scores",
        "true"
    ], env=job_env(gpu_id, i), cwd=IGTD_DIR)

    print(
        f"[{label} INFER DONE] "
        f"i={i}",
        flush=True
    )


#############################################
# CORE PIPELINE
#############################################

def run_pipeline(split_type, i):

    gpu_id = None

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
        ], env=job_env(gpu_id, i), cwd=IGTD_DIR)

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
            ccl_dir
        ], env=job_env(gpu_id, i), cwd=IGTD_DIR)

        print(
            f"[TRAIN DONE] "
            f"{split_type} i={i}",
            flush=True
        )

        if not os.path.isfile(
            os.path.join(ccl_dir, "preprocessor_params.pkl")
        ):
            raise RuntimeError(
                f"Training succeeded but preprocessor_params.pkl "
                f"is missing in {ccl_dir} -- cannot run inference"
            )

        #####################################################
        # CCL INFERENCE
        #####################################################

        run_test_set(
            f"{split_type} CCL",
            "ccl_test",
            i,
            gpu_id,
            ccl_test_split,
            os.path.join(run_dir, "ccl"),
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
                run_test_set(
                    f"{split_type} {study} PDO",
                    "pdo",
                    i,
                    gpu_id,
                    pdo_split,
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
                run_test_set(
                    f"{split_type} {study} CCL",
                    "ccl",
                    i,
                    gpu_id,
                    study_ccl_split,
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