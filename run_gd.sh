#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: run_gd.sh --work-dir DIR --input-depth FILE --gd-table FILE [options]

Runs preprocess, infer, call, optional eval, and plot with the gatk-sv-gd CLI.

Required arguments:
--work-dir DIR
--input-depth FILE
--gd-table FILE

Optional data inputs:
--high-res-counts FILE             Optional bgzipped tabix-indexed high-res counts
--high-res-depth FILE              Alias for --high-res-counts
--baf-table FILE                   Optional BAF table

Optional annotation and mask inputs:
--segdup-bed FILE
--centromere-bed FILE
--acrocentric-arm-bed FILE
--gaps-bed FILE
--gtf FILE
--par-bed FILE                     Recommended when chrX bins are present
--custom-mask-bed FILE
--hard-inclusion-bed FILE
--hard-inclusion-intervals FILE [FILE ...]
--flank-exclusion-interval FILE
--flank-exclusion-intervals FILE [FILE ...]

Optional arguments:
--truth-table FILE
--preprocess-args STRING
--infer-args STRING
--call-args STRING
--eval-args STRING
--plot-args STRING
--gd-cmd CMD

For backwards compatibility, a single positional argument is accepted as --work-dir.
EOF
    exit 1
}

require_arg() {
    local value="$1"
    local flag="$2"
    if [[ -z "${value}" ]]; then
        echo "Missing required argument: ${flag}" >&2
        usage
    fi
}

log() {
    printf '%s\n' "$*" >&2
}

WORK_DIR=""
INPUT_DEPTH=""
HIGH_RESOLUTION_DEPTH=""
BAF_TABLE=""
GD_TABLE=""
SEG_DUP_BED=""
CENTROMERE_BED=""
ACROCENTRIC_ARM_BED=""
GAPS_BED=""
GTF=""
PAR_BED=""
CUSTOM_MASK_BED=""
TRUTH_TABLE=""

PREPROCESS_ARGS=""
INFER_ARGS=""
CALL_ARGS=""
EVAL_ARGS=""
PLOT_ARGS=""

HARD_INCLUSION_INTERVALS=()
FLANK_EXCLUSION_INTERVALS=()
POSITIONAL_ARGS=()

GD_CMD=("gatk-sv-gd")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --input-depth)
            INPUT_DEPTH="$2"
            shift 2
            ;;
        --high-res-depth|--high-res-counts)
            HIGH_RESOLUTION_DEPTH="$2"
            shift 2
            ;;
        --baf-table)
            BAF_TABLE="$2"
            shift 2
            ;;
        --gd-table)
            GD_TABLE="$2"
            shift 2
            ;;
        --segdup-bed)
            SEG_DUP_BED="$2"
            shift 2
            ;;
        --centromere-bed)
            CENTROMERE_BED="$2"
            shift 2
            ;;
        --acrocentric-arm-bed)
            ACROCENTRIC_ARM_BED="$2"
            shift 2
            ;;
        --custom-mask-bed)
            CUSTOM_MASK_BED="$2"
            shift 2
            ;;
        --hard-inclusion-bed)
            HARD_INCLUSION_INTERVALS+=("$2")
            shift 2
            ;;
        --hard-inclusion-intervals)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                HARD_INCLUSION_INTERVALS+=("$1")
                shift
            done
            ;;
        --flank-exclusion-interval)
            FLANK_EXCLUSION_INTERVALS+=("$2")
            shift 2
            ;;
        --flank-exclusion-intervals)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                FLANK_EXCLUSION_INTERVALS+=("$1")
                shift
            done
            ;;
        --gaps-bed)
            GAPS_BED="$2"
            shift 2
            ;;
        --gtf)
            GTF="$2"
            shift 2
            ;;
        --par-bed)
            PAR_BED="$2"
            shift 2
            ;;
        --truth-table)
            TRUTH_TABLE="$2"
            shift 2
            ;;
        --preprocess-args)
            PREPROCESS_ARGS="$2"
            shift 2
            ;;
        --infer-args)
            INFER_ARGS="$2"
            shift 2
            ;;
        --call-args)
            CALL_ARGS="$2"
            shift 2
            ;;
        --eval-args)
            EVAL_ARGS="$2"
            shift 2
            ;;
        --plot-args)
            PLOT_ARGS="$2"
            shift 2
            ;;
        --gd-cmd)
            GD_CMD=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            POSITIONAL_ARGS+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${WORK_DIR}" && ${#POSITIONAL_ARGS[@]} -eq 1 ]]; then
    WORK_DIR="${POSITIONAL_ARGS[0]}"
elif [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    echo "Unexpected positional arguments: ${POSITIONAL_ARGS[*]}" >&2
    usage
fi

require_arg "${WORK_DIR}" "--work-dir"
require_arg "${INPUT_DEPTH}" "--input-depth"
require_arg "${GD_TABLE}" "--gd-table"

# ── Directories ────────────────────────────────────────────────────────────
PREPROCESS_DIR="${WORK_DIR}/preprocess"
INFER_DIR="${WORK_DIR}/infer"
CALL_DIR="${WORK_DIR}/call"
PLOT_DIR="${WORK_DIR}/plot"
EVAL_DIR="${WORK_DIR}/eval"

# ── Derived paths ──────────────────────────────────────────────────────────
BIN_MAPPINGS="${PREPROCESS_DIR}/bin_mappings.tsv.gz"
FILTERED_GD_TABLE="${PREPROCESS_DIR}/gd_table_filtered.tsv"
CN_POSTERIORS="${INFER_DIR}/cn_posteriors.tsv.gz"
SAMPLE_POSTERIORS="${INFER_DIR}/sample_posteriors.tsv.gz"
GD_CALLS="${CALL_DIR}/gd_cnv_calls.tsv.gz"
EVENT_MARGINALS="${CALL_DIR}/event_marginals.tsv.gz"
PLOIDY_TABLE="${PREPROCESS_DIR}/ploidy_estimates.tsv"
EVAL_REPORT="${EVAL_DIR}/truth_evaluation_report.tsv"

mkdir -p "${WORK_DIR}"

# ── Step 1: preprocess ─────────────────────────────────────────────────────
log "[1/5] preprocess"
rm -rf "${PREPROCESS_DIR}"
PREPROCESS_CMD=(
    "${GD_CMD[@]}"
    preprocess
    -i "${INPUT_DEPTH}" \
    -g "${GD_TABLE}" \
    -o "${PREPROCESS_DIR}" \
    --verbose \
)

if [[ -n "${HIGH_RESOLUTION_DEPTH}" ]]; then
    PREPROCESS_CMD+=(--high-res-counts "${HIGH_RESOLUTION_DEPTH}")
fi

if [[ -n "${BAF_TABLE}" ]]; then
    PREPROCESS_CMD+=(--baf-table "${BAF_TABLE}")
fi

if [[ -n "${SEG_DUP_BED}" ]]; then
    PREPROCESS_CMD+=(-e "${SEG_DUP_BED}")
fi

if [[ -n "${CENTROMERE_BED}" ]]; then
    PREPROCESS_CMD+=(-e "${CENTROMERE_BED}")
fi

if [[ -n "${ACROCENTRIC_ARM_BED}" ]]; then
    PREPROCESS_CMD+=(-e "${ACROCENTRIC_ARM_BED}")
fi

if [[ -n "${CUSTOM_MASK_BED}" ]]; then
    PREPROCESS_CMD+=(-e "${CUSTOM_MASK_BED}")
fi

if [[ ${#HARD_INCLUSION_INTERVALS[@]} -gt 0 ]]; then
    PREPROCESS_CMD+=(--hard-inclusion-intervals "${HARD_INCLUSION_INTERVALS[@]}")
fi

if [[ -n "${PAR_BED}" ]]; then
    PREPROCESS_CMD+=(--par-intervals "${PAR_BED}")
fi

if [[ ${#FLANK_EXCLUSION_INTERVALS[@]} -gt 0 ]]; then
    PREPROCESS_CMD+=(--flank-exclusion-intervals "${FLANK_EXCLUSION_INTERVALS[@]}")
fi

if [[ -n "${PREPROCESS_ARGS}" ]]; then
    # shellcheck disable=SC2206
    PREPROCESS_EXTRA=( ${PREPROCESS_ARGS} )
    PREPROCESS_CMD+=("${PREPROCESS_EXTRA[@]}")
fi

"${PREPROCESS_CMD[@]}"

# ── Step 2: infer ──────────────────────────────────────────────────────────
log "[2/5] infer"
rm -rf "${INFER_DIR}"
INFER_CMD=(
    "${GD_CMD[@]}"
    infer
    --preprocessed-dir "${PREPROCESS_DIR}" \
    -o "${INFER_DIR}" \
    --verbose \
)

if [[ -n "${INFER_ARGS}" ]]; then
    # shellcheck disable=SC2206
    INFER_EXTRA=( ${INFER_ARGS} )
    INFER_CMD+=("${INFER_EXTRA[@]}")
fi

"${INFER_CMD[@]}"

# ── Step 3: call ───────────────────────────────────────────────────────────
log "[3/5] call"
rm -rf "${CALL_DIR}"
CALL_CMD=(
    "${GD_CMD[@]}"
    call
    --cn-posteriors "${CN_POSTERIORS}" \
    --bin-mappings "${BIN_MAPPINGS}" \
    -g "${FILTERED_GD_TABLE}" \
    -o "${CALL_DIR}" \
    --ploidy-table "${PLOIDY_TABLE}" \
    --verbose \
)

if [[ -n "${CALL_ARGS}" ]]; then
    # shellcheck disable=SC2206
    CALL_EXTRA=( ${CALL_ARGS} )
    CALL_CMD+=("${CALL_EXTRA[@]}")
fi

"${CALL_CMD[@]}"

# ── Step 4: eval (optional) ────────────────────────────────────────────────
if [[ -n "${TRUTH_TABLE}" ]]; then
    log "[4/5] eval"
    rm -rf "${EVAL_DIR}"
    EVAL_CMD=(
        "${GD_CMD[@]}"
        eval
        --calls "${GD_CALLS}" \
        --truth-table "${TRUTH_TABLE}" \
        --gd-table "${FILTERED_GD_TABLE}" \
        --ploidy-table "${PLOIDY_TABLE}" \
        -o "${EVAL_DIR}" \
    )

    if [[ -n "${EVAL_ARGS}" ]]; then
        # shellcheck disable=SC2206
        EVAL_EXTRA=( ${EVAL_ARGS} )
        EVAL_CMD+=("${EVAL_EXTRA[@]}")
    fi

    "${EVAL_CMD[@]}"
else
    log "[4/5] eval  (skipped — TRUTH_TABLE not set)"
fi

# ── Step 5: plot ───────────────────────────────────────────────────────────
log "[5/5] plot"
rm -rf "${PLOT_DIR}"

PLOT_CMD=(
    "${GD_CMD[@]}"
    plot
    --calls "${GD_CALLS}"
    --cn-posteriors "${CN_POSTERIORS}"
    --sample-posteriors "${SAMPLE_POSTERIORS}"
    --raw-counts "${INPUT_DEPTH}"
    -g "${FILTERED_GD_TABLE}"
    -o "${PLOT_DIR}"
    --ploidy-table "${PLOIDY_TABLE}"
    --event-marginals "${EVENT_MARGINALS}"
)

if [[ -n "${HIGH_RESOLUTION_DEPTH}" ]]; then
    PLOT_CMD+=(--high-res-counts "${HIGH_RESOLUTION_DEPTH}")
fi

if [[ -n "${GAPS_BED}" ]]; then
    PLOT_CMD+=(--gaps-bed "${GAPS_BED}")
fi

if [[ -n "${GTF}" ]]; then
    PLOT_CMD+=(--gtf "${GTF}")
fi

if [[ -n "${SEG_DUP_BED}" ]]; then
    PLOT_CMD+=(--segdup-bed "${SEG_DUP_BED}")
fi

if [[ -n "${TRUTH_TABLE}" ]]; then
    PLOT_CMD+=(--eval-report "${EVAL_REPORT}")
fi

if [[ -n "${PLOT_ARGS}" ]]; then
    # shellcheck disable=SC2206
    PLOT_EXTRA=( ${PLOT_ARGS} )
    PLOT_CMD+=("${PLOT_EXTRA[@]}")
fi

"${PLOT_CMD[@]}"

log ""
if [[ -n "${TRUTH_TABLE}" ]]; then
    log "Eval report: ${EVAL_REPORT}"
else
    log "Eval report: skipped (TRUTH_TABLE not set)"
fi
log "Plot directory: ${PLOT_DIR}"
