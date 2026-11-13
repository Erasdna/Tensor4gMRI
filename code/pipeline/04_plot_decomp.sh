#!/bin/bash -l
# Figures of each group's fitted rank (gmri plot decomposition) in
# plots/decomposition/cp/rank_<r>_<group>/figures/decomposition/: the mode grid, subject mode
# (by the group variable and against NUMERICAL), time mode, spatial maps (REFERENCE_SUBJECT's
# segmentation over REFERENCE_BACKGROUND at SLICES, settings.sh) and outliers (leverage against
# residual), plus the subject and time coefficients as csv. The group variable is the first variable that varies within the
# group (Group for all, M and F; Sex for REF and iNPH), the others are box-plotted too.
# 02_decompose.sh starts it for each fit (PLOT=yes); run it on its own to redraw.
# Requires: 02_decompose.sh (decompose/cp/rank_<r>_<group>/cp_rank_<r>/model.h5),
#           00_find_files.sh (the reference segmentation and T1w).
# Usage: bash code/pipeline/04_plot_decomp.sh "<groups>" ["<ranks>"]

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- gmri plot decomposition options (gMRI docs/cli.md, "gmri plot decomposition") --
RANKS=${2:-$RANKS}                         # ranks to plot, or the optional 2nd argument
NUMERICAL=${NUMERICAL-"Age"}  # --numerical: subject-info columns to scatter the
                                           # subject mode against; empty: none
ALPHA=${ALPHA:-0.05}                       # --alpha: group tests
OUTLIER_ALPHA=${OUTLIER_ALPHA:-0.01}       # --outlier-alpha: Bonferroni-corrected cut-offs
PAGE_WIDTH=${PAGE_WIDTH:-double}           # --page-width: single (3.5"), onehalf (5.5"), double
                                           # (7"), or inches; the height follows the content
MIN_PANEL_WIDTH=${MIN_PANEL_WIDTH:-}       # --min-panel-width: inches; widens the subject-mode
                                           # figure beyond PAGE_WIDTH; empty: fit PAGE_WIDTH
FORMATS=${FORMATS:-"pdf png"}              # --formats
DPI=${DPI:-400}                            # --dpi

# --- SLURM options (sbatch) -----------------------------------------------------
PLOT_JOB=(--partition=normal --cpus-per-task=1 --mem-per-cpu=8G --time=00:30:00)

require_each group "${1:-}" "${SUBJECT_GROUPS[@]}"
require_ranks "$RANKS"
require_data
require_reference

for group in $1; do
    read -r group_variable categorical <<< "$(group_columns "$group" covariates)"
    for rank in $RANKS; do
        # shellcheck disable=SC2206  # one word per format
        cmd=(gmri plot decomposition --model "$(model_dir "$group" "$rank")/cp_rank_$rank"
            --subject-info "$SUBJECT_INFO" --group-variable "$group_variable"
            --output-dir "$RESULTS_DIR/plots/decomposition/cp/rank_${rank}_$group"
            --segmentation "$REFERENCE_SEGMENTATION" --background "$REFERENCE_BACKGROUND"
            --alpha "$ALPHA" --outlier-alpha "$OUTLIER_ALPHA" --page-width "$PAGE_WIDTH"
            --formats $FORMATS --dpi "$DPI")
        add_optional cmd --min-panel-width "$MIN_PANEL_WIDTH"
        add_optional cmd --categorical "$categorical"
        add_optional cmd --numerical "$NUMERICAL"
        add_optional cmd --slices "$SLICES"  # settings.sh
        run_or_submit "$(job_name plot_decomposition "$group" "$rank")" --sbatch PLOT_JOB "${cmd[@]}"
    done
done
