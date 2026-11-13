"""The inputs of the subject-statistics figures: `gmri stats` tables, its variables, and the
subject coefficients it tested."""

import json
import re
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from gMRItensor.model_io import load_decomposition, resolve_model, subject_coefficients
from gMRItensor.plotting.utils import scale_mode
from gMRItensor.subject_statistics import add_fdr

VARIABLE_KINDS = (
    "primary_categorical",
    "primary_numerical",
    "clinical_categorical",
    "clinical_numerical",
)
# The columns per component of an `fdr_sheet`
FDR_COLUMNS = ("partial_r", "partial_r_ci_low", "partial_r_ci_high", "p_perm", "p_fdr")


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `gmri stats` first")
    return path


def stats_variables(stats_dir: Path) -> dict[str, list[str]]:
    """The variables of the `gmri stats` run in `stats_dir`, by kind (`VARIABLE_KINDS`), as
    recorded in its `stats.json`."""
    options = json.loads(_require(Path(stats_dir) / "stats.json").read_text())[
        "options"
    ]
    return {kind: list(options.get(kind) or []) for kind in VARIABLE_KINDS}


def stats_interactions(stats_dir: Path) -> list[tuple[str, str]]:
    """The `(clinical, primary)` pairs of the `gmri stats` run in `stats_dir` (`--interaction`),
    as recorded in its `stats.json`."""
    options = json.loads(_require(Path(stats_dir) / "stats.json").read_text())[
        "options"
    ]
    return [
        (clinical, primary) for clinical, primary in options.get("interactions") or []
    ]


def read_tables(stats_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The primary table of model `stem` in `<stats_dir>/statistics/`, and its clinical table
    or None without one."""
    folder = Path(stats_dir) / "statistics"
    primary = pd.read_csv(_require(folder / f"{stem}__subject_stats_primary.csv"))
    clinical_path = folder / f"{stem}__subject_stats_clinical.csv"
    clinical = pd.read_csv(clinical_path) if clinical_path.is_file() else None
    return primary, clinical


def read_interactions(stats_dir: Path, stem: str) -> pd.DataFrame:
    """The interactions table of model `stem` in `<stats_dir>/statistics/`."""
    path = Path(stats_dir) / "statistics" / f"{stem}__subject_stats_interactions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `gmri stats` with --interaction first",
        )
    return pd.read_csv(path)


def model_stem(model: Path) -> str:
    """The name gmri gives the tables of `model` (a model directory or .h5 file)."""
    return resolve_model(Path(model))[1]


def subject_frame(model: Path, subject_info: Path) -> pd.DataFrame:
    """One row per subject of `model` with a row in `subject_info`: its info columns and
    c1, c2, ... the coefficients as `gmri stats` tests them (unit norm per component over
    these subjects)."""
    saved = load_decomposition(resolve_model(Path(model))[0])
    info = pd.read_csv(subject_info, dtype={"subjects": str}).set_index("subjects")
    subjects = [str(s) for s in saved.subjects]
    keep = [i for i, s in enumerate(subjects) if s in info.index]
    coefficients = scale_mode(subject_coefficients(saved)[keep])
    frame = info.loc[[subjects[i] for i in keep]].reset_index()
    for c in range(coefficients.shape[1]):
        frame[f"c{c + 1}"] = coefficients[:, c]
    return frame


def labels(values: pd.Series) -> pd.Series:
    """`values` as text, integer-valued numbers without decimals (as gmri names levels)."""
    if pd.api.types.is_float_dtype(values) and (values.dropna() % 1 == 0).all():
        values = values.astype("Int64")
    return values.astype("string")


def levels(values: pd.Series) -> list[str]:
    """The sorted levels of `values` (numeric codes numerically), as text."""
    present = values.dropna()
    if pd.api.types.is_float_dtype(present) and (present % 1 == 0).all():
        present = present.astype("Int64")
    return [str(level) for level in sorted(present.unique())]


def comparisons(table: pd.DataFrame) -> list[tuple[str, str]]:
    """`(term, level)` of each comparison in `table`, in table order; level is "" for a
    numerical term."""
    return list(
        dict.fromkeys(zip(table["term"], table["level"].fillna("").astype(str))),
    )


class Variable(NamedTuple):
    """A variable as named in 10_subject_stats.sh: `name`, `name[unit]` or `name[unit,label]`;
    "" where not given."""

    name: str
    unit: str = ""
    label: str = ""


def split_outside(text: str, separators: str) -> list[str]:
    """The parts of `text` between any of the `separators` outside square brackets, stripped,
    without empty parts."""
    parts, current, depth = [], "", 0
    for char in text:
        depth += {"[": 1, "]": -1}.get(char, 0)
        if char in separators and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return [part.strip() for part in [*parts, current] if part.strip()]


def parse_variable(spec: str) -> Variable:
    """The `Variable` of `name`, `name[unit]` or `name[unit,label]` (the name may hold
    parentheses, as in `TMT_B(sek)[s]`)."""
    name, bracket, inside = spec.strip().partition("[")
    balanced = inside.endswith("]") and inside.count("[") == inside.count("]") - 1
    if not name.strip() or "]" in name or (bracket and not balanced):
        raise ValueError(f"Expected a variable as 'name[unit,label]', got '{spec}'")
    unit, _, label = inside[:-1].partition(",")
    return Variable(name.strip(), unit.strip(), label.strip())


def parse_figure_set(text: str) -> tuple[str, str]:
    """The name and the categories of a set of figures, `name:Title=...;...` (name of letters,
    digits, "_" and "-"), or "" and `text` for an unnamed set."""
    named = re.fullmatch(r"\s*([A-Za-z0-9_-]+):(.*)", text, re.DOTALL)
    return (named[1], named[2]) if named else ("", text)


def parse_categories(
    text: str,
) -> tuple[dict[str, list[str]], list[Variable], dict[str, str]]:
    """Title -> variable names of `Title=variable,variable;Title|Group=...` (variables as in
    `parse_variable`), its variables, and title -> group of the categories with one (plotted
    per level of that primary variable)."""
    categories: dict[str, list[str]] = {}
    variables: list[Variable] = []
    groups: dict[str, str] = {}
    for entry in split_outside(text, ";"):
        head, equals, specs = (part.strip() for part in entry.partition("="))
        title, bar, group = (part.strip() for part in head.partition("|"))
        members = [parse_variable(spec) for spec in split_outside(specs, ",")]
        if not equals or not title or (bar and not group) or not members:
            raise ValueError(
                f"Expected a category as 'Title=variable,variable' or 'Title|Group=...', "
                f"got '{entry}'",
            )
        if title in categories:
            raise ValueError(f"Category '{title}' named more than once")
        categories[title] = [variable.name for variable in members]
        variables += members
        if group:
            groups[title] = group
    return categories, variables, groups


def variable_texts(
    variables: Sequence[Variable],
) -> tuple[dict[str, str], dict[str, str]]:
    """The units and the labels of `variables` given one, by name."""
    units = {v.name: v.unit for v in variables if v.unit}
    labels = {v.name: v.label for v in variables if v.label}
    return units, labels


def check_categories(terms: Sequence[str], categories: dict[str, list[str]]) -> None:
    """Raise ValueError unless the categories name only `terms`, each at most once (terms in no
    category are left out of the figures)."""
    named = [term for members in categories.values() for term in members]
    problems = {
        "Variables in more than one category": sorted(
            {t for t in named if named.count(t) > 1},
        ),
        "Not clinical variables": [t for t in dict.fromkeys(named) if t not in terms],
    }
    found = [
        f"{problem}: {', '.join(names)}" for problem, names in problems.items() if names
    ]
    if found:
        raise ValueError("; ".join(found))


def check_interaction_categories(
    categories: dict[str, list[str]],
    groups: dict[str, str],
    numerical: Sequence[str],
    interactions: Sequence[tuple[str, str]],
) -> None:
    """Raise ValueError unless every variable of a category with a group (`groups`) is
    numerical and was tested with that group (`interactions`)."""
    members = [(t, group) for title, group in groups.items() for t in categories[title]]
    problems = {
        "Categorical variables in an interaction category": [
            t for t, _ in members if t not in numerical
        ],
        "Not tested as interactions": [
            f"{t}:{g}"
            for t, g in members
            if t in numerical and (t, g) not in interactions
        ],
    }
    found = [
        f"{problem}: {', '.join(names)}" for problem, names in problems.items() if names
    ]
    if found:
        raise ValueError("; ".join(found))


class InteractionBlock(NamedTuple):
    """The rows of one test of an interactions table: `test` "slope" or "difference", `level`
    the group ("PD") or the groups compared ("PD vs Control"), and `rows` as a clinical table
    (`term` the clinical variable, `level` "")."""

    test: str
    level: str
    rows: pd.DataFrame


def interaction_blocks(
    table: pd.DataFrame,
    terms: Sequence[str],
    group: str,
) -> list[InteractionBlock]:
    """The slope blocks and then the difference blocks of `terms` with `group` in an
    interactions table, in table order, each with only the terms (in the order of `terms`) it
    has a slope (difference) for; blocks without one are left out."""
    clinical = table["term"].str.rsplit(" × ", n=1).str[0]
    chosen = table[(table["term"] == clinical + f" × {group}") & clinical.isin(terms)]
    chosen = chosen.assign(term=clinical[chosen.index])
    blocks = []
    for test in ("slope", "difference"):
        part = chosen[chosen["test"] == test]
        for level in dict.fromkeys(part["level"]):
            rows = part[part["level"] == level]
            tested = rows.groupby("term")["p_perm"].apply(lambda p: p.notna().any())
            shown = [t for t in terms if tested.get(t, False)]
            if shown:
                position = {t: i for i, t in enumerate(shown)}
                rows = rows[rows["term"].isin(shown)].assign(level="")
                rows = rows.sort_values(
                    "term",
                    key=lambda t: t.map(position),
                    kind="stable",
                )
                text = level.removeprefix("slope in ") if test == "slope" else level
                blocks.append(InteractionBlock(test, text, rows))
    return blocks


def standardised_effects(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with each row's effect on a scale comparable across components, `effect` with
    its 95% CI `effect_low` and `effect_high`, of `effect_kind` "r" or "d": a numerical term's
    `partial_r` and its CI, or a contrast's `standardised` (covariate-adjusted Cohen's d) with
    the CI of `estimate` divided by the same residual SD. NaN (kind "") for a row with
    neither."""
    table = table.copy()
    numerical = table["partial_r"].notna()
    contrast = ~numerical & table["standardised"].notna() & (table["std_kind"] == "d")
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = table["standardised"] / table["estimate"]
    table["effect"] = table["partial_r"].where(
        numerical,
        table["standardised"].where(contrast),
    )
    table["effect_low"] = table["partial_r_ci_low"].where(
        numerical,
        table["ci_low"] * scale,
    )
    table["effect_high"] = table["partial_r_ci_high"].where(
        numerical,
        table["ci_high"] * scale,
    )
    table.loc[~numerical & ~contrast, ["effect_low", "effect_high"]] = np.nan
    table["effect_kind"] = np.where(numerical, "r", np.where(contrast, "d", ""))
    return table


def family_fdr(
    table: pd.DataFrame,
    families: dict[str, list[str]],
    keys: Sequence[str] = ("component", "term"),
) -> pd.DataFrame:
    """The rows of `table` whose term is in a family of `families` (title: terms), in the order
    of the families and their terms, with `category` the family's title and `p_fdr` from
    Benjamini-Hochberg over the family's tests in all components (`add_fdr`, rows sharing
    `keys` one test)."""
    parts = []
    for title, terms in families.items():
        position = {t: i for i, t in enumerate(terms)}
        part = table[table["term"].isin(terms)]
        if len(part):
            part = part.sort_values(
                "term",
                key=lambda t: t.map(position),
                kind="stable",
            )
            parts.append(add_fdr(part, keys=keys).assign(category=title))
    if not parts:
        return table.iloc[:0].assign(category="")
    return pd.concat(parts, ignore_index=True)


def fdr_sheet(rows: pd.DataFrame, group: bool = False) -> pd.DataFrame:
    """One row per category and term of `family_fdr` rows, with per component k its
    `c<k>_<column>` of `FDR_COLUMNS`; a factor's contrasts, which share their p-values, as one
    row. With `group`, the rows are interaction slopes ("X × Group", level "slope in <g>"):
    one row per variable X and group g."""
    rows = rows.assign(variable=rows["term"])
    index = ["category", "variable"]
    if group:
        rows = rows.assign(
            variable=rows["term"].str.rsplit(" × ", n=1).str[0],
            group=rows["level"].str.removeprefix("slope in "),
        )
        index.append("group")
    first = rows.drop_duplicates([*index, "component"])
    components = sorted(first["component"].unique())
    order = [(column, k) for k in components for column in FDR_COLUMNS]
    wide = first.pivot(index=index, columns="component", values=list(FDR_COLUMNS))
    wide = wide.reindex(pd.MultiIndex.from_frame(first[index].drop_duplicates()))
    wide = wide.reindex(columns=order)
    wide.columns = [f"c{k}_{column}" for column, k in order]
    return wide.reset_index()


def _residuals(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def _design(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    categorical: Collection[str],
) -> np.ndarray:
    """The intercept and `covariates` of `frame`, a categorical one as 0/1 columns but its first
    level's: the column space of gMRI's treatment coding."""
    blocks = [np.ones((len(frame), 1))]
    for covariate in covariates:
        if covariate in categorical:
            blocks.append(
                pd.get_dummies(labels(frame[covariate]), drop_first=True).to_numpy(
                    float,
                ),
            )
        else:
            blocks.append(pd.to_numeric(frame[covariate]).to_numpy(float)[:, None])
    return np.column_stack(blocks)


def added_variables(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    categorical: Collection[str],
) -> pd.DataFrame:
    """The added-variable residuals of each tested numerical row (`partial_r`) of a clinical or
    interactions table, per subject of its model (`frame`, with `c1`, ...) and component: `x`
    the term's and `y` the coefficient's residual on the row's other model columns, in their
    own units, so that the slope of `y` on `x` is the row's `estimate` and their correlation
    its `partial_r`. An interaction slope in group g (`level` "slope in g") is the pooled
    model's: its other columns are the covariates and the other groups' slopes, and `in_group`
    marks g's subjects. `term` is the clinical variable."""
    rows = []
    tested = table[table["partial_r"].notna()].fillna({"level": "", "covariates": ""})
    for (term, level), part in tested.groupby(["term", "level"], sort=False):
        covariates = [c for c in str(part["covariates"].iloc[0]).split(" + ") if c]
        level = str(level)
        clinical, _, group = str(term).partition(" × ")
        needed = [clinical, *covariates, *([group] if group else [])]
        data = frame.dropna(subset=list(dict.fromkeys(needed)))
        x = pd.to_numeric(data[clinical]).to_numpy(float)
        design = _design(data, covariates, categorical)
        inside = np.ones(len(data), bool)
        if group:
            members = labels(data[group])
            sloped = tested.loc[tested["term"] == term, "level"].str.removeprefix(
                "slope in ",
            )
            own = level.removeprefix("slope in ")
            others = [
                x * (members == g).to_numpy(float)
                for g in dict.fromkeys(sloped)
                if g != own
            ]
            design = np.column_stack([design, *others])
            inside = (members == own).to_numpy()
            x = x * inside
        e_x = _residuals(x, design)
        for component in sorted(part["component"].astype(int).unique()):
            e_y = _residuals(data[f"c{component}"].to_numpy(float), design)
            rows.append(
                pd.DataFrame(
                    {
                        "term": clinical,
                        "level": level,
                        "component": component,
                        "x": e_x,
                        "y": e_y,
                        "in_group": inside,
                    },
                ),
            )
    columns = ["term", "level", "component", "x", "y", "in_group"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
