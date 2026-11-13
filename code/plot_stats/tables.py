"""LaTeX tables of the clinical variables of a set of figures: each variable's mean ± SD, or the
number of subjects at each level of a categorical one, within each level of the categorical primary
variables, in a booktabs `table`."""

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

from plot_stats.data import labels, levels

LATEX_SPECIAL = "&%#_{}"
DEMOGRAPHICS = "Demographics"

Column = tuple[str, str]  # (categorical primary variable, level)


def significant(value: float, digits: int = 2) -> str:
    """`value` to `digits` significant digits, without an exponent (e.g. 480, 26, 0.034)."""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    rounded = round(value, digits - 1 - exponent)
    if int(np.floor(np.log10(abs(rounded)))) > exponent:
        exponent += 1  # rounding gained a digit, as 9.96 to 10
        rounded = round(value, digits - 1 - exponent)
    return f"{rounded:.{max(digits - 1 - exponent, 0)}f}"


def latex_escape(text: str) -> str:
    """`text` with the LaTeX special characters escaped outside $...$."""
    parts = text.split("$")
    for i in range(0, len(parts), 2):
        for char in LATEX_SPECIAL:
            parts[i] = parts[i].replace(char, "\\" + char)
    return "$".join(parts)


def group_columns(frame: pd.DataFrame, group_variables: Sequence[str]) -> list[Column]:
    """`(variable, level)` of each level of each of the `group_variables`."""
    return [
        (variable, level)
        for variable in group_variables
        for level in levels(frame[variable])
    ]


def _members(frame: pd.DataFrame, column: Column) -> pd.Series:
    variable, level = column
    return (labels(frame[variable]) == level).fillna(False).astype(bool)


def numerical_cells(
    frame: pd.DataFrame,
    term: str,
    columns: Sequence[Column],
) -> list[str]:
    """`mean $\\pm$ SD` of `term` per column, each to two significant digits; followed by `(k)`
    where only k of the column's subjects have a value, and "--" where none has."""
    cells = []
    for column in columns:
        group = frame.loc[_members(frame, column), term]
        present = group.dropna()
        if present.empty:
            cells.append("--")
            continue
        text = significant(present.mean())
        if len(present) > 1:
            text += f" $\\pm$ {significant(present.std(ddof=1))}"
        if len(present) < len(group):
            text += f" ({len(present)})"
        cells.append(text)
    return cells


def categorical_cells(
    frame: pd.DataFrame,
    term: str,
    columns: Sequence[Column],
) -> list[str]:
    """The number of each column's subjects at each level of `term`, as "k / k"; "--" in the
    columns of `term` itself and where none has a value."""
    order = levels(frame[term])
    cells = []
    for column in columns:
        values = labels(frame.loc[_members(frame, column), term]).dropna()
        if column[0] == term or values.empty:
            cells.append("--")
            continue
        cells.append(" / ".join(str(int((values == level).sum())) for level in order))
    return cells


def summary_table(
    frame: pd.DataFrame,
    blocks: Sequence[tuple[str, Sequence[str]]],
    group_variables: Sequence[str],
    primary_numerical: Sequence[str],
    categorical: Collection[str],
    units: dict[str, str],
    names: dict[str, str],
    label: str,
) -> str:
    """A booktabs `table` (labelled `label`) of the subjects of `frame`: one column per level of
    each of the `group_variables` with its n, a Demographics block of the `primary_numerical` and
    `group_variables`, then a block per `(title, terms)` of `blocks` with terms. Each block is
    spaced from the one before: its title in bold, and a row per term, `name (unit)` (else a.u.)
    with its `numerical_cells`, or `name (level / level)` of a `categorical` or group variable
    with its `categorical_cells`. Terms are shown by `names` with their `units`."""
    columns = group_columns(frame, group_variables)
    width = len(columns) + 1

    def name(term: str) -> str:
        return names.get(term, term)

    def row(term: str) -> str:
        if term in categorical or term in group_variables:
            text = f"{name(term)} ({' / '.join(levels(frame[term]))})"
            cells = categorical_cells(frame, term, columns)
        else:
            text = f"{name(term)} ({units.get(term) or 'a.u.'})"
            cells = numerical_cells(frame, term, columns)
        return " & ".join([rf"\quad {latex_escape(text)}", *cells]) + r" \\"

    spans = [
        (v, sum(variable == v for variable, _ in columns)) for v in group_variables
    ]
    ends = np.cumsum([1] + [span for _, span in spans])
    sizes = [int(_members(frame, column).sum()) for column in columns]
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{l{'c' * len(columns)}}}",
        r"\toprule",
        " & "
        + " & ".join(
            rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(name(v))}}}"
            for v, span in spans
        )
        + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{start + 1}-{end}}}"
            for start, end in zip(ends[:-1], ends[1:])
        ),
        "Group & "
        + " & ".join(
            latex_escape(f"{level} (n={size})")
            for (_, level), size in zip(columns, sizes)
        )
        + r" \\",
        r"\midrule",
    ]
    demographics = (DEMOGRAPHICS, [*primary_numerical, *group_variables])
    for i, (title, terms) in enumerate(
        [block for block in [demographics, *blocks] if block[1]],
    ):
        if i:
            lines.append(r"\addlinespace")
        lines.append(
            rf"\multicolumn{{{width}}}{{l}}{{\textbf{{{latex_escape(title)}}}}} \\",
        )
        lines += [row(term) for term in terms]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"
