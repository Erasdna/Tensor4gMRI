#!/bin/bash -l
# Covariate-adjusted tests of each group's fitted rank's subject coefficients (gmri stats): does a
# component differ between groups, or go along with a clinical measurement, once the primary
# variables are accounted for? Each primary variable is tested adjusted for the other primary
# variables, each clinical variable in its own model adjusted for all primary ones, with
# Freedman-Lane permutation p-values and FDR correction per table. Writes
# statistics/cp_rank_<r>__subject_stats_primary.csv (and _clinical.csv) into the fit's plot
# folder, plots/decomposition/cp/rank_<r>_<group>/, next to 04's figures. A second job then draws
# their figures there (code/plot_stats/plot_subject_stats.py): the coefficients by primary
# variable (__subject_stats_groups), the standardised effects per component (Cohen's d or partial
# correlation with 95% CI; __subject_stats_primary, __subject_stats_clinical, with ..._p_fdr.csv),
# the clinical correlations adjusted for the primary variables (__subject_stats_clinical_partial_r)
# and per category the added-variable plots of its numerical variables, and a LaTeX table of the
# primary and clinical variables by category, mean +- SD or counts per level of each categorical
# primary variable (__subject_stats_clinical_table.tex).
# The categorical primary variables are those that vary within the group (Group and Sex for all,
# Sex for REF and iNPH, Group for M and F) unless PRIMARY_CATEGORICAL is set; the first is the
# group variable of the figures. See gMRI's docs/subject-statistics.md for the columns.
# Requires: 02_decompose.sh (decompose/cp/rank_<r>_<group>/cp_rank_<r>/model.h5).
# Usage: bash code/pipeline/05_subject_stats.sh "<groups>" ["<ranks>"]
#   groups: all REF iNPH M F
#   e.g. bash code/pipeline/05_subject_stats.sh "all REF iNPH" "3 4"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- gmri stats options (gMRI docs/cli.md, "gmri stats") ------------------------------
RANKS=${2:-$RANKS}    # ranks to test (one model each), or the 2nd argument
# Primary variables: each is tested adjusted for the others. At least one is needed. Each is a
# subject_info.csv column name, or name[unit] or name[unit,label] to show it in the figures with
# that unit and label, e.g. 'Age[years,Age at scan]'.
# PRIMARY_CATEGORICAL                    # --primary-categorical; unset: the group's (above)
PRIMARY_NUMERICAL=${PRIMARY_NUMERICAL-"Age[years,Age]"}   # --primary-numerical
# Clinical variables (column names only): each is tested in its own model, adjusted for all
# primary ones, in one clinical table. Empty: the primary table only.
CLINICAL_CATEGORICAL=${CLINICAL_CATEGORICAL:-}            # --clinical-categorical
CLINICAL_NUMERICAL=${CLINICAL_NUMERICAL-"Height Weight"}  # --clinical-numerical
# Interactions, 'clinical:primary' pairs of a CLINICAL_NUMERICAL and a PRIMARY_CATEGORICAL
# variable: the clinical variable's slope within each group of the primary one, and the slopes'
# differences (gMRI docs/subject-statistics.md, section 5). Empty: none.
CLINICAL_INTERACTIONS=${CLINICAL_INTERACTIONS-}           # --interaction
N_PERMUTATIONS=${N_PERMUTATIONS:-10000}  # --n-permutations: Freedman-Lane permutations per test
SEED=${SEED:-0}                          # --seed: seed of the permutations
ALPHA=${ALPHA:-0.05}                     # --alpha: FDR level of the significant column
# --- figures (code/plot_stats/plot_subject_stats.py) -----------------------------
# The clinical variables to plot, by category, "Title=variable,variable;Title=...": each one of
# CLINICAL_CATEGORICAL/NUMERICAL, in at most one category (the tables keep every clinical
# variable; those in no category are not plotted), each as for the primary variables. A category
# 'Title|Group=...' is drawn per level of the primary variable Group (each of its variables
# numerical and paired with Group in CLINICAL_INTERACTIONS). Each element of the array is a set of
# these figures and their table; a set named 'name:Title=...' goes in statistics/<name>/, one may
# be unnamed (statistics/). Set as an environment variable, CLINICAL_CATEGORIES is one set.
if [ -z "${CLINICAL_CATEGORIES+set}" ]; then
CLINICAL_CATEGORIES=(
    'Anthropometrics=Height[cm],Weight[kg]'
)
fi
MARKER_VARIABLE=${MARKER_VARIABLE-Sex}  # subject_info.csv column of the dot shapes ("": one)
PLOTS_ONLY=${PLOTS_ONLY:-no}            # yes: only redraw the figures from existing tables

# --- SLURM options (sbatch) of the jobs this stage submits ----------------------
SMALL_JOB=(--partition=normal --cpus-per-task=1 --mem-per-cpu=4G --time=02:00:00)
PLOT_JOB=(--partition=normal --cpus-per-task=1 --mem-per-cpu=4G --time=00:30:00)

require_each group "${1:-}" "${SUBJECT_GROUPS[@]}"
require_ranks "$RANKS"
require_each PLOTS_ONLY "$PLOTS_ONLY" yes no
require_data

# Print the parts of <text> between any of the characters of <separators> outside
# square brackets, one per line, without surrounding spaces or empty parts; line breaks are
# spaces.
split_outside() {
    local separators=$1 text=${2//[$'\n\t']/ } part="" depth=0 char i
    for (( i = 0; i <= ${#text}; i++ )); do
        char=${text:i:1}
        if [ "$char" = "[" ]; then depth=$((depth + 1)); fi
        if [ "$char" = "]" ]; then depth=$((depth - 1)); fi
        if [ "$i" -eq "${#text}" ] || { [ "$depth" -eq 0 ] && [[ "$separators" == *"$char"* ]]; }
        then
            part="${part#"${part%%[![:space:]]*}"}"
            part="${part%"${part##*[![:space:]]}"}"
            if [ -n "$part" ]; then printf '%s\n' "$part"; fi
            part=""
        else
            part+=$char
        fi
    done
}

# Print the column names of the variables in <list> (name, name[unit] or name[unit,label]),
# one per line; exit on a variable that is none of these.
variable_names() {
    local spec opens closes pattern='^[^][]+\[.*\]$'
    while IFS= read -r spec; do
        opens=${spec//[^\[]/}
        closes=${spec//[^\]]/}
        if [[ "$spec" == *[][]* ]] && { ! [[ "$spec" =~ $pattern ]] \
            || [ ${#opens} -ne ${#closes} ]; }; then
            echo "Expected a variable as 'name[unit,label]', got '$spec'" >&2
            exit 1
        fi
        printf '%s\n' "${spec%%\[*}"
    done < <(split_outside " " "$1")
}

# Exit unless <categories> (a set of CLINICAL_CATEGORIES) names only the clinical variables
# <names...>, each in at most one category, and only numerical ones paired with Group in
# CLINICAL_INTERACTIONS in a 'Title|Group' category.
check_categories() {
    local categories=$1 entry head title group spec name repeated named=() extra=() grouped=()
    local categorical=() untested=() numerical=" ${CLINICAL_NUMERICAL//[$'\n\t']/ } "
    shift
    while IFS= read -r entry; do
        head=${entry%%=*}
        read -r title <<< "${head%%|*}"
        read -r group <<< "${head#*|}"
        if [[ "$entry" != *=* ]] || [ -z "$title" ] || [ -z "${entry#*=}" ] \
            || { [[ "$head" == *"|"* ]] && [ -z "$group" ]; }; then
            echo "CLINICAL_CATEGORIES: expected 'Title=variable,variable' or" \
                "'Title|Group=variable,variable', got '$entry'" >&2
            exit 1
        fi
        while IFS= read -r spec; do
            name=$(variable_names "$spec") || exit 1
            named+=("$name")
            if [[ "$head" == *"|"* ]]; then grouped+=("$name:$group"); fi
        done < <(split_outside "," "${entry#*=}")
    done < <(split_outside ";" "$categories")
    for name in "${named[@]}"; do
        if [[ " $* " != *" $name "* ]]; then extra+=("$name"); fi
    done
    for name in "${grouped[@]}"; do
        if [[ "$numerical" != *" ${name%:*} "* ]]; then
            categorical+=("${name%:*}")
        elif [[ " ${CLINICAL_INTERACTIONS//[$'\n\t']/ } " != *" $name "* ]]; then
            untested+=("$name")
        fi
    done
    repeated=$(printf '%s\n' "${named[@]}" | sort | uniq -d | paste -sd' ')
    if [ -n "$repeated" ]; then
        echo "Variables in more than one category of CLINICAL_CATEGORIES: $repeated" >&2
        exit 1
    fi
    if [ ${#extra[@]} -gt 0 ]; then
        echo "Not clinical variables, in CLINICAL_CATEGORIES: ${extra[*]}" >&2
        exit 1
    fi
    if [ ${#categorical[@]} -gt 0 ]; then
        echo "Categorical variables in a 'Title|Group' category of CLINICAL_CATEGORIES:" \
            "${categorical[*]}" >&2
        exit 1
    fi
    if [ ${#untested[@]} -gt 0 ]; then
        echo "Not in CLINICAL_INTERACTIONS, for CLINICAL_CATEGORIES: ${untested[*]}" >&2
        exit 1
    fi
}

# The sets of CLINICAL_CATEGORIES but empty ones: SET_NAMES[i] ("" unnamed) and
# SET_CATEGORIES[i].
SET_NAMES=()
SET_CATEGORIES=()
named_set='^[[:space:]]*([A-Za-z0-9_-]+):(.*)$'
for figure_set in "${CLINICAL_CATEGORIES[@]}"; do
    if [ -z "${figure_set//[[:space:]]/}" ]; then continue; fi
    if [[ "$figure_set" =~ $named_set ]]; then
        SET_NAMES+=("${BASH_REMATCH[1]}")
        SET_CATEGORIES+=("${BASH_REMATCH[2]}")
    else
        SET_NAMES+=("")
        SET_CATEGORIES+=("$figure_set")
    fi
done

# Exit unless the sets of CLINICAL_CATEGORIES have different names, one at most none, and each
# passes check_categories <names...>.
check_figure_sets() {
    local i repeated
    repeated=$(printf '%s\n' "${SET_NAMES[@]}" | sort | uniq -d | paste -sd' ')
    if [ -n "$repeated" ] || [ "$(printf '%s\n' "${SET_NAMES[@]}" | grep -c '^$')" -gt 1 ]; then
        echo "CLINICAL_CATEGORIES: name each set differently ('name: Title=...');" \
            "one at most may be unnamed" >&2
        exit 1
    fi
    for i in "${!SET_CATEGORIES[@]}"; do check_categories "${SET_CATEGORIES[i]}" "$@"; done
}

# Exit unless each pair of CLINICAL_INTERACTIONS is 'clinical:primary' of a CLINICAL_NUMERICAL
# and a PRIMARY_CATEGORICAL variable, each pair once.
check_interactions() {
    local pair repeated primary numerical=" ${CLINICAL_NUMERICAL//[$'\n\t']/ } "
    primary=" $(variable_names "$PRIMARY_CATEGORICAL" | paste -sd' ') "
    # shellcheck disable=SC2086  # one word per pair
    for pair in $CLINICAL_INTERACTIONS; do
        if ! [[ "$pair" =~ ^[^:]+:[^:]+$ ]]; then
            echo "CLINICAL_INTERACTIONS: expected 'clinical:primary', got '$pair'" >&2
            exit 1
        fi
        if [[ "$numerical" != *" ${pair%:*} "* ]]; then
            echo "CLINICAL_INTERACTIONS: ${pair%:*} (in $pair) is not in CLINICAL_NUMERICAL" >&2
            exit 1
        fi
        if [[ "$primary" != *" ${pair#*:} "* ]]; then
            echo "CLINICAL_INTERACTIONS: ${pair#*:} (in $pair) is not in PRIMARY_CATEGORICAL" >&2
            exit 1
        fi
    done
    # shellcheck disable=SC2086  # one word per pair
    repeated=$(printf '%s\n' $CLINICAL_INTERACTIONS | sort | uniq -d | paste -sd' ')
    if [ -n "$repeated" ]; then
        echo "Pairs named more than once in CLINICAL_INTERACTIONS: $repeated" >&2
        exit 1
    fi
}

# Exit unless the variables (with the categorical primary ones <primary...>) name at least one
# primary column, no column twice, and only columns of subject_info.csv, so a wrong name stops
# here rather than in a job.
check_variables() {
    local primary=$1 named=() clinical=() column missing=() repeated header names
    names=$(variable_names "$primary $PRIMARY_NUMERICAL") || exit 1
    mapfile -t named <<< "$names"
    if [[ "$CLINICAL_CATEGORICAL $CLINICAL_NUMERICAL" == *[][]* ]]; then
        echo "Units and labels of clinical variables go in CLINICAL_CATEGORIES:" \
            "$CLINICAL_CATEGORICAL $CLINICAL_NUMERICAL" >&2
        exit 1
    fi
    # shellcheck disable=SC2206  # one word per column
    clinical=($CLINICAL_CATEGORICAL $CLINICAL_NUMERICAL)
    named+=("${clinical[@]}")
    if [ -z "${primary}${PRIMARY_NUMERICAL}" ]; then
        echo "No primary variables: set PRIMARY_CATEGORICAL and/or PRIMARY_NUMERICAL" >&2
        exit 1
    fi
    repeated=$(printf '%s\n' "${named[@]}" | sort | uniq -d | paste -sd' ')
    if [ -n "$repeated" ]; then
        echo "Variables named more than once: $repeated" >&2
        exit 1
    fi
    PRIMARY_CATEGORICAL=$primary check_interactions
    check_figure_sets "${clinical[@]}"
    header=",$(head -n 1 "$SUBJECT_INFO" | tr -d '\r"'),"
    for column in "${named[@]}" ${MARKER_VARIABLE:+"$MARKER_VARIABLE"}; do
        if [[ "$header" != *",$column,"* ]]; then missing+=("$column"); fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "Not columns of $SUBJECT_INFO: ${missing[*]}" >&2
        exit 1
    fi
}

# The plot job's units and labels: each primary variable given one, and the categories.
PLOT_VARIABLES=()
while IFS= read -r spec; do
    if [[ "$spec" == *\[* ]]; then PLOT_VARIABLES+=(--variable "$spec"); fi
done < <(split_outside " " "${PRIMARY_CATEGORICAL:-} $PRIMARY_NUMERICAL")
for i in "${!SET_CATEGORIES[@]}"; do
    # On one line, as commands.log holds one command per line.
    PLOT_VARIABLES+=(--categories "${SET_NAMES[i]:+${SET_NAMES[i]}:}$(
        split_outside ";" "${SET_CATEGORIES[i]}" | paste -sd';')")
done

# Every group's categorical primary variables, all checked before anything is submitted.
declare -A GROUP_PRIMARY
for group in $1; do
    if [ -n "${PRIMARY_CATEGORICAL+set}" ]; then
        GROUP_PRIMARY[$group]=$PRIMARY_CATEGORICAL
    else
        GROUP_PRIMARY[$group]=$(group_columns "$group" covariates)
    fi
    check_variables "${GROUP_PRIMARY[$group]}"
done

# subject_stats <group>: per rank, gmri stats and then its figures.
subject_stats() {
    local group=$1 rank model output after=${DEPENDENCY:-} DEPENDENCY names=()
    local primary=${GROUP_PRIMARY[$1]}
    mapfile -t names < <(variable_names "$primary")  # the first is the group variable
    for rank in $RANKS; do
        DEPENDENCY=$after
        model="$(model_dir "$group" "$rank")/cp_rank_$rank"
        output="$RESULTS_DIR/plots/decomposition/cp/rank_${rank}_$group"
        if [ "$PLOTS_ONLY" = "no" ]; then
            local cmd=(gmri stats --model "$model" --subject-info "$SUBJECT_INFO"
                --output-dir "$output")
            add_optional cmd --primary-categorical "$(variable_names "$primary")"
            add_optional cmd --primary-numerical "$(variable_names "$PRIMARY_NUMERICAL")"
            add_optional cmd --clinical-categorical "$CLINICAL_CATEGORICAL"
            add_optional cmd --clinical-numerical "$CLINICAL_NUMERICAL"
            add_optional cmd --interaction "$CLINICAL_INTERACTIONS"
            cmd+=(--n-permutations "$N_PERMUTATIONS" --seed "$SEED" --alpha "$ALPHA")
            run_or_submit "$(job_name subject_stats "$group" "$rank")" "${cmd[@]}"
            if [ -n "$JOB_ID" ]; then DEPENDENCY="afterany:$JOB_ID"; fi
        fi
        run_or_submit "$(job_name plot_subject_stats "$group" "$rank")" --sbatch PLOT_JOB \
            python code/plot_stats/plot_subject_stats.py --model "$model" \
            --subject-info "$SUBJECT_INFO" --stats-dir "$output" \
            --group-variable "${names[0]:-Group}" --marker-variable "$MARKER_VARIABLE" \
            "${PLOT_VARIABLES[@]}"
    done
}

for group in $1; do subject_stats "$group"; done
