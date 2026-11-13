"""Figures of one model's `gmri stats` results, next to its tables in `<stats-dir>/statistics/`:

- `<model>__subject_stats_groups`: the coefficients by each primary variable
- `<model>__subject_stats_primary`: standardised effects of the primary variables per
  component, Cohen's d of a categorical one and the partial correlation of a numerical one
- `<model>__subject_stats_clinical`: standardised effects of the clinical variables (with a
  clinical table)
- `<model>__subject_stats_clinical_partial_r`: correlations of the numerical clinical variables,
  adjusted for the primary variables
- `<model>__subject_stats_added_variable_<Title>`: per category of `--categories`, the
  coefficients against each numerical clinical variable, both adjusted for the primary variables
- `<model>__subject_stats_interaction_<Title>` and `..._partial_r`: per category `Title|Group` of
  `--categories`, the partial correlations of its variables within each level of the primary
  variable `Group` (`gmri stats --interaction`), and `..._added_variable_<level>` their
  added-variable plots per level; spaces in `Title` and `level` as "_"
- `..._p_fdr.csv` next to `..._primary`, `..._clinical` and `..._interaction_<Title>`: per
  variable (and group of a slope) and component its partial correlation with 95% CI, `p_perm`,
  and `p_fdr` from Benjamini-Hochberg within its category (the primary variables one) over all
  components; the figures show the uncorrected `p_perm`

- `<model>__subject_stats_clinical_table.tex`: a LaTeX table of the clinical variables by
  category, mean ± SD (counts per level of a categorical one) within each level of the categorical
  primary variables, after a Demographics block of the primary variables

Variables are shown by the units and display names of `--variable` and `--categories`, and the
clinical ones in one block per category of `--categories`. The effects and correlation figures
take 45 mm per column of panels and 90 mm per column of blocks (180 mm for a full line) and,
where they fit, at most 170 mm in height. Each `--categories` is a set of the
clinical figures and table; a named one (`name:Title=...`) goes in `statistics/<name>/`.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from gMRItensor.plotting.utils import save_figure
from matplotlib.figure import Figure
from plot_stats.data import (
    added_variables,
    check_categories,
    check_interaction_categories,
    family_fdr,
    fdr_sheet,
    interaction_blocks,
    model_stem,
    parse_categories,
    parse_figure_set,
    parse_variable,
    read_interactions,
    read_tables,
    stats_interactions,
    stats_variables,
    subject_frame,
    variable_texts,
)
from plot_stats.figures import (
    MAX_H,
    PAD,
    plot_added_variables,
    plot_effects,
    plot_interaction_added_variables,
    plot_interaction_effects,
    plot_interaction_partial_r,
    plot_partial_r,
    plot_primary_groups,
)
from plot_stats.tables import summary_table

# Clinical comparisons per row of their effects figure
CLINICAL_COLUMNS = 4


def save_page_figure(fig: Figure, stem: str) -> list[Path]:
    """`save_figure`, with a note where `fig` is taller than a page's 170 mm (`MAX_H`)."""
    if fig.get_figheight() > MAX_H + 1e-6:
        height = (fig.get_figheight() + 2 * PAD) * 25.4
        print(f"{Path(stem).name} is {height:.0f} mm tall, over 170 mm")
    return save_figure(fig, stem)


def write_fdr(path: Path, sheet: pd.DataFrame) -> list[Path]:
    """Write `sheet` (`fdr_sheet`) to `path`; the written file, none for an empty sheet."""
    if sheet.empty:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False)
    return [path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="The model gmri stats tested",
    )
    parser.add_argument(
        "--subject-info",
        type=Path,
        required=True,
        help="Its --subject-info",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        required=True,
        help="Its --output-dir",
    )
    parser.add_argument(
        "--group-variable",
        required=True,
        help="Subject info column of groups",
    )
    parser.add_argument(
        "--marker-variable",
        default="Sex",
        help="Subject info column that sets the dot shapes ('' for one shape)",
    )
    parser.add_argument(
        "--variable",
        action="append",
        default=[],
        help="A primary variable's unit and display name as 'name[unit,label]'; repeatable",
    )
    parser.add_argument(
        "--categories",
        action="append",
        default=[],
        help="A set of clinical figures: the clinical variables by category, "
        "'[name:]Title=name[unit,label],name;Title|Group=...'; repeatable, one unnamed at most",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw the figures; return the written files."""
    args = parse_args(argv)
    stem = model_stem(args.model)
    variables = stats_variables(args.stats_dir)
    primary, clinical = read_tables(args.stats_dir, stem)
    frame = subject_frame(args.model, args.subject_info)
    statistics = args.stats_dir / "statistics"
    sets = [parse_figure_set(text) for text in args.categories] or [("", "")]
    set_names = [name for name, _ in sets]
    if len(set(set_names)) < len(set_names):
        raise ValueError(
            f"--categories: name each set differently, got names {set_names}",
        )
    parsed = {name: parse_categories(text) for name, text in sets}
    clinical_terms = variables["clinical_categorical"] + variables["clinical_numerical"]
    for categories, _, groups in parsed.values():
        if categories:
            check_categories(clinical_terms, categories)
        if groups:
            check_interaction_categories(
                categories,
                groups,
                variables["clinical_numerical"],
                stats_interactions(args.stats_dir),
            )
    primary_variables = [parse_variable(spec) for spec in args.variable]
    units, names = variable_texts(primary_variables)
    primary_terms = variables["primary_categorical"] + variables["primary_numerical"]

    out = statistics / f"{stem}__subject_stats"
    written = save_figure(
        plot_primary_groups(
            frame,
            variables["primary_categorical"],
            variables["primary_numerical"],
            args.group_variable,
            args.marker_variable or None,
            units,
            names,
        ),
        f"{out}_groups",
    )
    written += save_page_figure(
        plot_effects(primary, names=names, show_statistic=True),
        f"{out}_primary",
    )
    written += write_fdr(
        Path(f"{out}_primary_p_fdr.csv"),
        fdr_sheet(family_fdr(primary, {"Primary variables": primary_terms})),
    )
    for name, (categories, clinical_variables, groups) in parsed.items():
        units, names = variable_texts(primary_variables + clinical_variables)
        set_out = (statistics / name if name else statistics) / f"{stem}__subject_stats"
        written += clinical_table(
            Path(f"{set_out}_clinical_table.tex"),
            frame,
            list(categories.items()) or [("Clinical variables", clinical_terms)],
            variables,
            units,
            names,
            f"tab:{stem}{f'_{name}' if name else ''}_clinical",
        )
        written += clinical_figures(
            set_out,
            clinical,
            read_interactions(args.stats_dir, stem) if groups else None,
            frame,
            variables["primary_categorical"],
            categories,
            groups,
            variables["clinical_numerical"],
            primary_terms,
            units,
            names,
        )
    for path in written:
        print(f"Wrote {path}")
    return written


def clinical_table(
    path: Path,
    frame: pd.DataFrame,
    blocks: list[tuple[str, list[str]]],
    variables: dict[str, list[str]],
    units: dict[str, str],
    names: dict[str, str],
    label: str,
) -> list[Path]:
    """Write the `summary_table` of `blocks` by the categorical primary variables (`variables` by
    kind, as `stats_variables`) to `path`; the written file, none without such variables.
    """
    if not variables["primary_categorical"]:
        print(f"No categorical primary variables to group {path.name} by")
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        summary_table(
            frame,
            blocks,
            variables["primary_categorical"],
            variables["primary_numerical"],
            variables["clinical_categorical"],
            units,
            names,
            label,
        ),
    )
    return [path]


def _slug(text: str) -> str:
    return re.sub(r"\s+", "_", text)


def clinical_figures(
    out: Path,
    clinical: pd.DataFrame | None,
    interactions: pd.DataFrame | None,
    frame: pd.DataFrame,
    categorical: list[str],
    categories: dict[str, list[str]],
    groups: dict[str, str],
    numerical_terms: list[str],
    primary_terms: list[str],
    units: dict[str, str],
    names: dict[str, str],
) -> list[Path]:
    """The clinical figures of one set of `categories`, as files from `out`, and those of its
    categories with a group (`groups`) from the `interactions` table, each effects figure with
    its `fdr_sheet` (one family per category); the added-variable plots from the subject
    `frame` (`subject_frame`), whose `categorical` primary variables are dummy-coded. The
    written files."""
    written: list[Path] = []
    plain = {title: terms for title, terms in categories.items() if title not in groups}
    if clinical is not None and (plain or not categories):
        written += save_page_figure(
            plot_effects(
                clinical,
                show_n=True,
                columns=CLINICAL_COLUMNS,
                categories=plain or None,
                names=names,
            ),
            f"{out}_clinical",
        )
        families = plain or {
            "Clinical variables": list(dict.fromkeys(clinical["term"])),
        }
        written += write_fdr(
            Path(f"{out}_clinical_p_fdr.csv"),
            fdr_sheet(family_fdr(clinical, families)),
        )
        partial_r = plot_partial_r(
            clinical,
            numerical_terms,
            primary_terms,
            plain or None,
            names,
        )
        if partial_r is not None:
            written += save_page_figure(partial_r, f"{out}_clinical_partial_r")
        shown = plain or {"": list(dict.fromkeys(clinical["term"]))}
        chosen = clinical[
            clinical["term"].isin([t for terms in shown.values() for t in terms])
        ]
        residuals = added_variables(frame, chosen, categorical)
        for title, terms in shown.items():
            figure = plot_added_variables(residuals, clinical, terms, units, names)
            if figure is not None:
                stem = f"{out}_added_variable{f'_{_slug(title)}' if title else ''}"
                written += save_page_figure(figure, stem)
    if interactions is None:
        return written
    for title, group in groups.items():
        blocks = interaction_blocks(interactions, categories[title], group)
        if not blocks:
            print(f"No slopes of '{title}' to plot")
            continue
        figure = f"{out}_interaction_{_slug(title)}"
        written += save_page_figure(
            plot_interaction_effects(
                blocks,
                title,
                categories[title],
                True,
                CLINICAL_COLUMNS,
                names,
            ),
            figure,
        )
        pairs = [f"{term} × {group}" for term in categories[title]]
        slopes = family_fdr(
            interactions[interactions["test"] == "slope"],
            {title: pairs},
            keys=("component", "term", "level"),
        )
        written += write_fdr(Path(f"{figure}_p_fdr.csv"), fdr_sheet(slopes, group=True))
        partial_r = plot_interaction_partial_r(
            blocks,
            title,
            group,
            categories[title],
            primary_terms,
            names,
        )
        if partial_r is not None:
            written += save_page_figure(partial_r, f"{figure}_partial_r")
        residuals = added_variables(
            frame,
            interactions[interactions["term"].isin(pairs)],
            categorical,
        )
        for block in blocks:
            if block.test != "slope":
                continue
            added = plot_interaction_added_variables(
                residuals,
                interactions,
                categories[title],
                block.level,
                units,
                names,
            )
            if added is not None:
                written += save_page_figure(
                    added,
                    f"{figure}_added_variable_{_slug(block.level)}",
                )
    return written


if __name__ == "__main__":
    main()
