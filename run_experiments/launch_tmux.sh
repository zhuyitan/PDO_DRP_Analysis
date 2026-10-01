#!/usr/bin/env bash
#
# Start an experiment runner in its own detached tmux session.
#
# The runner keeps going after you disconnect. Its output is shown in the
# tmux session and also saved to a log file.
#
# Usage (from anywhere):
#
#   run_experiments/launch_tmux.sh <conda_env> <runner.py> [work_dir]
#
#   conda_env  conda environment the model needs (activated inside tmux)
#   runner.py  e.g. run_experiments/cell_line_only_training/run_graphdrp.py
#   work_dir   directory holding pdo_drp_data/ and models/
#              (default: the repository root)
#
# Example:
#
#   run_experiments/launch_tmux.sh graphdrp \
#       run_experiments/cell_line_only_training/run_graphdrp.py
#
# CUDA_VISIBLE_DEVICES and JOBS_PER_GPU are passed through if set:
#
#   CUDA_VISIBLE_DEVICES=0,1 run_experiments/launch_tmux.sh graphdrp ...
#
# Useful tmux commands:
#
#   tmux ls                     list sessions
#   tmux attach -t <session>    watch a session (detach again: Ctrl-b then d)
#   tmux kill-session -t <name> stop a session

set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

#############################################
# INSIDE TMUX
#############################################

if [[ "${1:-}" == "--inner" ]]; then

    CONDA_BASE=$2
    CONDA_ENV=$3
    RUNNER=$4
    LOG_FILE=$5

    set +eu

    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if ! conda activate "$CONDA_ENV"; then
        echo "Could not activate conda environment '$CONDA_ENV'." | tee -a "$LOG_FILE"
        exec bash
    fi

    # Each progress message is then written as one whole line,
    # even when several jobs print at the same moment.
    unset PYTHONUNBUFFERED

    {
        echo "Started:     $(date)"
        echo "Host:        $(hostname)"
        echo "Conda env:   $CONDA_ENV"
        echo "Python:      $(command -v python)"
        echo "Runner:      $RUNNER"
        echo "Work dir:    $(pwd)"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
        echo "JOBS_PER_GPU:         ${JOBS_PER_GPU:-1}"
        echo ""
    } | tee -a "$LOG_FILE"

    python "$RUNNER" 2>&1 | tee -a "$LOG_FILE"
    status=${PIPESTATUS[0]}

    {
        echo ""
        echo "Finished:    $(date)"
        echo "Exit code:   $status"
    } | tee -a "$LOG_FILE"

    # Keep the window open so the output can still be read.
    exec bash
fi

#############################################
# ARGUMENTS
#############################################

if [[ $# -lt 2 || $# -gt 3 ]]; then
    sed -n '2,32p' "$SELF" | sed 's/^# \{0,1\}//'
    exit 1
fi

CONDA_ENV=$1
RUNNER=$2

REPO_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
WORK_DIR="${3:-$REPO_ROOT}"

if [[ ! -f "$RUNNER" ]]; then
    echo "Runner not found: $RUNNER" >&2
    exit 1
fi

if [[ ! -d "$WORK_DIR" ]]; then
    echo "Work dir not found: $WORK_DIR" >&2
    exit 1
fi

RUNNER="$(cd "$(dirname "$RUNNER")" && pwd)/$(basename "$RUNNER")"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

if ! command -v tmux > /dev/null; then
    echo "tmux is not installed." >&2
    exit 1
fi

if ! command -v conda > /dev/null; then
    echo "conda was not found. Run 'conda init bash' or load conda first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"

#############################################
# SESSION NAME AND LOG FILE
#############################################

# e.g. cell_line_only_training-graphdrp
PHASE="$(basename "$(dirname "$RUNNER")")"
MODEL="$(basename "$RUNNER" .py)"
MODEL="${MODEL#run_}"
SESSION="${PHASE}-${MODEL}"
SESSION="${SESSION//[.:]/_}"

if tmux has-session -t "=$SESSION" 2> /dev/null; then
    echo "A tmux session named '$SESSION' is already running." >&2
    echo "Watch it with:  tmux attach -t $SESSION" >&2
    echo "Or stop it with: tmux kill-session -t $SESSION" >&2
    exit 1
fi

mkdir -p "$WORK_DIR/logs"
LOG_FILE="$WORK_DIR/logs/${SESSION}_$(date +%Y%m%d-%H%M%S).log"

#############################################
# LAUNCH
#############################################

# Pass GPU settings explicitly: a tmux server that is already running
# does not see environment variables from this shell.
PASS_ENV=()
[[ -n "${CUDA_VISIBLE_DEVICES+x}" ]] && PASS_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
[[ -n "${JOBS_PER_GPU+x}" ]] && PASS_ENV+=("JOBS_PER_GPU=$JOBS_PER_GPU")

COMMAND="$(printf '%q ' env ${PASS_ENV[@]+"${PASS_ENV[@]}"} bash "$SELF" --inner "$CONDA_BASE" "$CONDA_ENV" "$RUNNER" "$LOG_FILE")"

tmux new-session -d -s "$SESSION" -c "$WORK_DIR" "$COMMAND"

echo "Started tmux session: $SESSION"
echo "Log file:             $LOG_FILE"
echo ""
echo "Watch it:   tmux attach -t $SESSION   (detach again: Ctrl-b then d)"
echo "Follow log: tail -f \"$LOG_FILE\""