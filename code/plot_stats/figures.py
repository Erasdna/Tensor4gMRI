"""Figures of `gmri stats` results: the coefficients by primary variable, the standardised
effects of the primary and clinical variables per component, the clinical partial correlations,
and the clinical added-variable plots. Variables are shown by their display names and units
where given (10_subject_stats.sh `name[unit,label]`), clinical ones grouped into lettered blocks
by category."""

import math
import re
from collections.abc import Sequence
from string import ascii_uppercase
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")  # Prevent GUI display errors in containers/terminals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from gMRItensor.plotting.utils import get_color_palette
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import linregress

from plot_stats.data import (
    InteractionBlock,
    check_categories,
    comparisons,
    labels,
    levels,
    standardised_effects,
)

plt.style.use(["science", "no-latex"])

# Effects with a permutation p below this are drawn filled (not corrected for multiple tests).
FILLED_P = 0.05
OTHER_COLOR = "0.25"
TEXT_COLOR = "0.15"
CATEGORY_COLOR = "0.85"
MISSING_GROUP_COLOR = "0.6"
NOT_AVAILABLE_COLOR = "0.85"  # a correlation of a group without a slope
MARKERS = ("o", "^", "s", "D", "v", "P")
STATISTICS = {"r": "Correlation", "d": "Cohen's d"}  # by `standardised_effects` kind

# Layout of the effects and correlation figures, in inches: saved (with the style's padding) a
# share of a 180 mm page per column of panels or blocks and, where they fit, at most 170 mm tall.
PAD = plt.rcParams["savefig.pad_inches"]
PAGE_W = 180 / 25.4
MAX_H = 170 / 25.4 - 2 * PAD
COLUMNS = 4  # effects panels per line, each PAGE_W / COLUMNS wide
Y_LABEL_W = 0.16  # the "Component" beside a line of effects panels
CORRELATION_COLUMNS = (
    2  # correlation blocks per line, each PAGE_W / CORRELATION_COLUMNS wide
)
EDGE = 0.06  # figure edge to the block titles and keys
TITLE_H = 0.17  # a block's lettered title
KEY_H = 0.2  # a key
SPACING = 0.06  # between blocks
BLOCK_GAP = 0.3  # between correlation blocks side by side
LABEL_PAD = 0.06  # row labels to their axes
PANEL_GAP = 0.22
PANEL_H_PER_COMPONENT = 0.25
MIN_H_PER_COMPONENT = 0.15  # as far as panels shrink to fit MAX_H
PANEL_H_EXTRA = 0.08
N_TITLE_H = 0.15  # the n= above a panel
NO_TITLE_H = 0.04
FOOT_H = 0.34  # x tick labels and a one-line x label
X_LABEL_LINE_H = 0.12  # each further line of an x label
X_LABEL_SIZE = 7
DIRECTION_H = 0.13  # the levels named under a contrast
HEATMAP_CELL_H = 0.26
COLORBAR_LABEL_H = 0.3
COLORBAR_H = 0.09
COLORBAR_GAP = 0.06
HEATMAP_TICKS_H = 0.32  # component numbers and their label
AV_PANEL_H = 0.95  # an added-variable panel, before shrinking to fit MAX_H
MIN_AV_PANEL_H = 0.6
AV_ROW_GAP = 0.08
AV_Y_LABEL_W = 0.3  # the two-line "Component k / adjusted (a.u.)"
OTHER_GROUP_COLOR = "0.8"
KEY_STYLE: dict[str, Any] = {
    "frameon": False,
    "fontsize": 6.5,
    "borderaxespad": 0,
    "borderpad": 0,
    "handletextpad": 0.4,
    "columnspacing": 1.5,
}


def group_palette(group_levels: Sequence[str]) -> dict[str, str]:
    """Group level -> colour, in the order of gMRI's subject figures."""
    return dict(zip(group_levels, get_color_palette(len(group_levels))))


def component_color(component: int) -> str:
    """The colour of `component` (from 1), as in gMRI's mode-grid figures."""
    return f"C{component - 1}"


def p_text(p: float) -> str:
    return "$p<0.001$" if p < 0.001 else f"$p={p:.2g}$"


def _boxes(ax: Axes, values: list[np.ndarray], colors: Sequence[str]) -> None:
    drawn = [(i, v, c) for i, (v, c) in enumerate(zip(values, colors)) if len(v)]
    if not drawn:
        return
    positions, data, box_colors = zip(*drawn)
    boxes = ax.boxplot(
        list(data),
        positions=list(positions),
        widths=0.4,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)


def _dots(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    colors: np.ndarray,
    markers: np.ndarray,
) -> None:
    for marker in dict.fromkeys(markers):
        chosen = markers == marker
        ax.scatter(
            x[chosen],
            y[chosen],
            c=list(colors[chosen]),
            marker=marker,
            s=14,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )


def _category_panel(
    ax: Axes,
    frame: pd.DataFrame,
    column: str,
    y: str,
    box_colors: dict[str, str] | None,
    rng: np.random.Generator,
) -> None:
    present = frame.dropna(subset=[column, y])
    names = levels(frame[column])
    position = labels(present[column]).map({name: i for i, name in enumerate(names)})
    values = present[y].to_numpy(float)
    _boxes(
        ax,
        [values[(position == i).to_numpy()] for i in range(len(names))],
        [box_colors[n] if box_colors else CATEGORY_COLOR for n in names],
    )
    x = position.to_numpy(float) + rng.uniform(-0.07, 0.07, len(present))
    _dots(ax, x, values, present["_color"].to_numpy(), present["_marker"].to_numpy())
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def _numerical_panel(ax: Axes, frame: pd.DataFrame, column: str, y: str) -> None:
    present = frame.dropna(subset=[column, y])
    x = pd.to_numeric(present[column]).to_numpy(float)
    values = present[y].to_numpy(float)
    _dots(ax, x, values, present["_color"].to_numpy(), present["_marker"].to_numpy())
    if len(x) >= 3 and np.unique(x).size >= 2:
        fit = linregress(x, values)
        xs = np.array([x.min(), x.max()])
        ax.plot(
            xs,
            fit.slope * xs + fit.intercept,
            color="black",
            linewidth=1,
            label=rf"$R^2$={fit.rvalue**2:.2f}",
        )
        ax.legend(
            loc="upper left",
            fontsize=6,
            handlelength=1.2,
            frameon=True,
            framealpha=0.9,
        )


def plot_primary_groups(
    frame: pd.DataFrame,
    categorical: Sequence[str],
    numerical: Sequence[str],
    group_variable: str,
    marker_variable: str | None = "Sex",
    units: dict[str, str] | None = None,
    names: dict[str, str] | None = None,
) -> Figure:
    """The coefficients (columns c1, c2, ... of `frame`) by each primary variable: one row per
    component, one column per variable, titled by its display name (`names`, else its column
    name). Boxes by `group_variable` in the group colours and by another categorical variable
    in grey; a numerical variable (labelled with its unit from `units`, else a.u.) against the
    coefficient with one fit over every subject. Every subject is a dot in its group's colour,
    shaped by `marker_variable` when `frame` has it."""
    units, names = units or {}, names or {}
    components = [c for c in frame.columns if c[:1] == "c" and c[1:].isdigit()]
    variables = [*categorical, *numerical]
    group_levels = levels(frame[group_variable])
    palette = group_palette(group_levels)
    frame = frame.assign(
        _color=labels(frame[group_variable]).map(palette).fillna(MISSING_GROUP_COLOR),
    )
    shapes: dict[str, str] = {}
    if marker_variable and marker_variable in frame.columns:
        marker_levels = levels(frame[marker_variable])
        shapes = {
            level: MARKERS[i % len(MARKERS)] for i, level in enumerate(marker_levels)
        }
        frame["_marker"] = labels(frame[marker_variable]).map(shapes).fillna(MARKERS[0])
    else:
        frame["_marker"] = MARKERS[0]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(
        len(components),
        len(variables),
        squeeze=False,
        sharey="row",
        figsize=(2.7 * len(variables) + 0.4, 2.1 * len(components) + 0.4),
    )
    for row, component in enumerate(components):
        for ax, variable in zip(axes[row], variables):
            if variable in categorical:
                box_colors = palette if variable == group_variable else None
                _category_panel(ax, frame, variable, component, box_colors, rng)
            else:
                _numerical_panel(ax, frame, variable, component)
        axes[row, 0].set_ylabel(f"Component {row + 1}\ncoefficient (a.u.)")
    for ax, variable in zip(axes[0], variables):
        ax.set_title(names.get(variable, variable))
    for ax, variable in zip(axes[-1], variables):
        if variable not in categorical:
            ax.set_xlabel(
                variable_label(names.get(variable, variable), units.get(variable)),
            )

    handles: list[Patch | Line2D] = [
        Patch(facecolor=color, alpha=0.7, label=level)
        for level, color in palette.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=m,
            linestyle="",
            color="0.5",
            markeredgecolor="black",
            label=s,
        )
        for s, m in shapes.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    return fig


def _effects_panel(ax: Axes, rows: pd.DataFrame, rank: int) -> bool:
    """Draw one comparison's rows (one per component); whether any was tested."""
    tested = False
    columns = ["component", "effect", "effect_low", "effect_high", "p_perm"]
    for component, estimate, ci_low, ci_high, p_perm in rows[columns].to_numpy(float):
        y = rank - int(component)
        if np.isnan(p_perm):
            ax.text(
                0.5, y, "not tested", ha="center", va="center", fontsize=6, color="0.5",
                transform=ax.get_yaxis_transform(),
            )  # fmt: skip
            continue
        tested = True
        color = component_color(int(component))
        ax.plot([ci_low, ci_high], [y, y], color=color, linewidth=1.3, zorder=2)
        filled = p_perm < FILLED_P
        ax.scatter(
            [estimate],
            [y],
            s=28,
            zorder=3,
            linewidths=1.3,
            edgecolors=color,
            facecolors=color if filled else "white",
        )
        ax.text(
            estimate, y + 0.16, p_text(p_perm), ha="center", va="bottom", fontsize=5.5,
            color=TEXT_COLOR, zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.3},
        )  # fmt: skip
    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
    return tested


def _direction(ax: Axes, level: str) -> None:
    """Name the levels a contrast compares at the ends of the x axis: the reference (lower
    estimates) on the left, the compared level (higher estimates) on the right."""
    compared, reference = level.split(" vs ", 1)
    for x, text, ha in (
        (0, rf"$\leftarrow$ {reference}", "left"),
        (1, rf"{compared} $\rightarrow$", "right"),
    ):
        ax.annotate(
            text, xy=(x, 0), xycoords="axes fraction", xytext=(0, -13),
            textcoords="offset points", ha=ha, va="top", fontsize=6.5,
        )  # fmt: skip


def _text_width(
    texts: Sequence[str],
    fontsize: float,
    fontweight: str = "normal",
) -> float:
    """Width in inches of the widest of `texts` at `fontsize`, measured before the figure
    they go in is laid out."""
    fig = Figure()
    renderer = FigureCanvasAgg(fig).get_renderer()
    widths = []
    for text in texts:
        artist = Text(text=text, fontsize=fontsize, fontweight=fontweight)
        artist.set_figure(fig)
        widths.append(artist.get_window_extent(renderer).width)
    return max(widths, default=0.0) / fig.dpi


def _head_width(
    titles: Sequence[str],
    keys: Sequence[Sequence[Patch | Line2D]],
) -> float:
    """Width in inches of the widest of the block `titles` and one-line `keys`."""
    fig = Figure()
    renderer = FigureCanvasAgg(fig).get_renderer()
    widths = [_text_width(titles, 9, "bold") * fig.dpi]
    for handles in keys:
        legend = fig.legend(handles=handles, ncol=len(handles), **KEY_STYLE)
        widths.append(legend.get_window_extent(renderer).width)
    return max(widths) / fig.dpi


def _wrap(text: str, width: float, fontsize: float) -> str:
    """`text` on two lines where it is wider than `width` inches (never inside $...$)."""
    words = re.findall(r"(?:\$[^$]*\$|[^\s$])+", text)
    if _text_width([text], fontsize) <= width or len(words) < 2:
        return text
    fits = [
        i
        for i in range(1, len(words))
        if _text_width([" ".join(words[:i])], fontsize) <= width
    ]
    split = fits[-1] if fits else 1
    return f"{' '.join(words[:split])}\n{' '.join(words[split:])}"


def _place(
    fig: Figure,
    ax: Axes,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    """Put `ax` `left` and `top` inches from the top left corner of `fig`, `width` by
    `height` inches."""
    fig_w, fig_h = fig.get_size_inches()
    ax.set_position(
        (left / fig_w, 1 - (top + height) / fig_h, width / fig_w, height / fig_h),
    )


def _figure_width(columns: int, per_page: int, content: float = 0.0) -> float:
    """The width of a figure of `columns` columns of the page's `per_page`, saved padded to
    `PAGE_W` / `per_page` per column (to `PAGE_W` at most), widened by whole columns to hold
    `content` inches."""
    needed = math.ceil((content + 2 * PAD) * per_page / PAGE_W - 1e-9)
    return PAGE_W * min(max(columns, needed), per_page) / per_page - 2 * PAD


def _frame(fig: Figure) -> None:
    """An invisible box over the whole of `fig`, so that it is saved at its full size."""
    fig.add_artist(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=fig.transFigure,
            fill=False,
            edgecolor="none",
        ),
    )


def _block_head(
    fig: Figure,
    title: str,
    handles: Sequence[Patch | Line2D],
    top: float,
    x: float = EDGE,
) -> float:
    """Draw a block's title and key (each if any) from `top` inches below the top of `fig` and
    `x` inches from its left; return where the block's content starts. Placed in figure
    fractions, which the tight bounding box of a saved figure keeps in step with the axes.
    """
    width, height = fig.get_size_inches()
    if title:
        fig.text(
            x / width, 1 - top / height, title, va="top", fontweight="bold", fontsize=9,
        )  # fmt: skip
        top += TITLE_H
    if not handles:
        return top
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(x / width, 1 - top / height),
        ncol=len(handles), **KEY_STYLE,
    )  # fmt: skip
    return top + KEY_H


def _blocks(
    terms: Sequence[str],
    categories: dict[str, list[str]] | None,
) -> list[tuple[str, list[str]]]:
    """(title, terms) of each block of a figure: one per category (`check_categories`), or one
    untitled block of every term without `categories`."""
    if not categories:
        return [("", list(terms))]
    check_categories(terms, categories)
    return list(categories.items())


def _lettered(titles: Sequence[str]) -> list[str]:
    """`titles` as "A  title", "B  title", ...; untitled blocks stay untitled."""
    return [
        f"{letter}  {title}" if title else ""
        for letter, title in zip(ascii_uppercase, titles)
    ]


def _effects_key() -> list[Patch | Line2D]:
    return [
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=OTHER_COLOR,
            markeredgecolor=OTHER_COLOR, label=rf"$p_{{perm}}<{FILLED_P}$",
        ),
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="white",
            markeredgecolor=OTHER_COLOR, label=rf"$p_{{perm}}\geq{FILLED_P}$ (uncorrected)",
        ),
        Line2D([], [], color=OTHER_COLOR, linewidth=1.3, label="95% CI"),
    ]  # fmt: skip


def variable_label(name: str, unit: str | None) -> str:
    """`name (unit)`, with "a.u." without a unit."""
    return f"{name} ({unit or 'a.u.'})"


def _x_label(ax: Axes, text: str, width: float) -> str:
    """Format the x ticks of `ax` and return its label `text`, wrapped to the panel `width`.
    Values below 0.01 or from 1000 are shown in units of a power of ten named in the label.
    """
    largest = max(abs(limit) for limit in ax.get_xlim())
    exponent = int(np.floor(np.log10(largest))) if largest > 0 else 0
    scale = 10.0**exponent if exponent <= -3 or exponent >= 3 else 1.0
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / scale:g}"))
    if scale != 1.0:
        text += rf" $\times 10^{{{exponent}}}$"
    return _wrap(text, width, X_LABEL_SIZE)


class _EffectsBlock(NamedTuple):
    """A block of an effects figure: its `title` and the `pairs` (term, level) of `table` it
    shows."""

    title: str
    table: pd.DataFrame
    pairs: list[tuple[str, str]]


def plot_effects(
    table: pd.DataFrame,
    show_n: bool = False,
    columns: int | None = None,
    categories: dict[str, list[str]] | None = None,
    names: dict[str, str] | None = None,
    show_statistic: bool = False,
) -> Figure:
    """One panel per comparison of a `gmri stats` table: each component's standardised effect
    (`standardised_effects`: the partial correlation of a numerical term, Cohen's d of a
    contrast) and its 95% CI in its colour, filled when its permutation p is below `FILLED_P`;
    on top, the statistic with `show_statistic` and `n=` subjects with `show_n`. The x label
    names the term (by `names`, else its column name), or the levels a contrast compares,
    which also name the ends of its x axis; the correlation panels share their x range, as do
    the Cohen's d ones. With `categories` (title -> terms of `table`, each in at most one) one
    block per category, titled A, B, ..., and terms in none left out; otherwise one block,
    untitled as is a figure's only block. One key on top; each block fills lines of at most
    `columns` panels (`COLUMNS` by default), as tall as fit in `MAX_H`; each column of panels
    is `PAGE_W` / `COLUMNS` wide, and the figure wider by whole columns where its key or titles
    need it."""
    pairs = comparisons(table)
    blocks = [
        _EffectsBlock(
            title,
            table,
            [pair for term in terms for pair in pairs if pair[0] == term],
        )
        for title, terms in _blocks(list(dict.fromkeys(table["term"])), categories)
    ]
    return _draw_effects(blocks, show_n, columns, names or {}, show_statistic)


def plot_interaction_effects(
    blocks: Sequence[InteractionBlock],
    title: str,
    terms: Sequence[str],
    show_n: bool = True,
    columns: int | None = None,
    names: dict[str, str] | None = None,
) -> Figure:
    """`plot_effects` of the slope blocks of the `interaction_blocks` of the `terms` of
    category `title`, one per group ("title: PD") with the partial correlations of its slopes
    in the pooled model, each term in the same column of every block, blank where a block has
    no slope for it."""
    return _draw_effects(
        [
            _EffectsBlock(
                f"{title}: {block.level}",
                block.rows,
                [(term, "") for term in terms],
            )
            for block in blocks
            if block.test == "slope"
        ],
        show_n,
        columns,
        names or {},
    )


def _draw_effects(
    blocks: Sequence[_EffectsBlock],
    show_n: bool,
    columns: int | None,
    names: dict[str, str],
    show_statistic: bool = False,
) -> Figure:
    if len(blocks) == 1:
        blocks = [blocks[0]._replace(title="")]
    blocks = [
        block._replace(table=standardised_effects(block.table)) for block in blocks
    ]
    rank = max(int(block.table["component"].max()) for block in blocks)
    columns = min(columns or COLUMNS, max(len(block.pairs) for block in blocks))
    left = EDGE + Y_LABEL_W + _text_width([str(rank)], 7) + LABEL_PAD
    titles = _lettered([block.title for block in blocks])
    width = _figure_width(
        columns, COLUMNS, 2 * EDGE + _head_width(titles, [_effects_key()]),
    )  # fmt: skip
    panel_w = (width - left) / columns - PANEL_GAP
    fig = plt.figure()
    axes: list[list[Axes]] = []
    shared: dict[str, list[Axes]] = {}
    for block in blocks:
        level_text = block.table["level"].fillna("").astype(str)
        block_axes: list[Axes] = []
        for i, (term, level) in enumerate(block.pairs):
            rows = block.table[(block.table["term"] == term) & (level_text == level)]
            ax = fig.add_axes(
                (0, 0, 1, 1),
                sharey=block_axes[0] if block_axes else None,
            )
            ax.tick_params(axis="y", which="minor", left=False, right=False)
            ax.tick_params(axis="y", labelleft=i % columns == 0, labelsize=7)
            if i % columns == 0:
                ax.set_ylabel("Component", fontsize=7, labelpad=3)
            block_axes.append(ax)
            if rows.empty:
                # A blank slot, keeping the component numbers of its line.
                ax.set_frame_on(False)
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                )
                ax.tick_params(axis="y", left=False, right=False)
                continue
            kind = next((k for k in rows["effect_kind"] if k), "")
            if _effects_panel(ax, rows.sort_values("component"), rank):
                shared.setdefault(kind, []).append(ax)
            heading = [STATISTICS[kind]] if show_statistic and kind else []
            heading += [f"n={int(rows['n'].iloc[0])}"] if show_n else []
            ax.set_title(", ".join(heading), fontsize=7, pad=3)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(axis="x", labelsize=6)
        if block_axes:
            block_axes[0].set_yticks(range(rank), [str(rank - i) for i in range(rank)])
            block_axes[0].set_ylim(-0.6, rank - 0.2)
        axes.append(block_axes)
    for kind_axes in shared.values():
        lows, highs = zip(*(ax.get_xlim() for ax in kind_axes))
        for ax in kind_axes:
            ax.set_xlim(min(lows), max(highs))
    x_labels = [
        [
            _x_label(
                ax,
                f"{names.get(term, term)}: {level}" if level else names.get(term, term),
                panel_w,
            )
            if ax.get_frame_on()
            else ""
            for ax, (term, level) in zip(block_axes, block.pairs)
        ]
        for block_axes, block in zip(axes, blocks)
    ]

    def lines(items: list) -> list[list]:
        return [items[i : i + columns] for i in range(0, len(items), columns)]

    head_h = N_TITLE_H if show_n or show_statistic else NO_TITLE_H

    def foot_h(labels_: list[str], pairs: list[tuple[str, str]]) -> float:
        extra_lines = max(label.count("\n") for label in labels_)
        contrast = any(level for _, level in pairs)
        return FOOT_H + X_LABEL_LINE_H * extra_lines + (DIRECTION_H if contrast else 0)

    feet = [
        [
            foot_h(line_labels, line_pairs) for line_labels, line_pairs in
            zip(lines(block_labels), lines(block.pairs))
        ]
        for block_labels, block in zip(x_labels, blocks)
    ]  # fmt: skip
    n_lines = sum(len(block_feet) for block_feet in feet)
    height = 2 * EDGE + KEY_H + SPACING * (len(blocks) - 1)
    height += sum(
        (TITLE_H if block.title else 0) + sum(head_h + f for f in block_feet)
        for block, block_feet in zip(blocks, feet)
    )
    fit = ((MAX_H - height) / n_lines - PANEL_H_EXTRA) / rank
    panel_h = (
        max(min(PANEL_H_PER_COMPONENT, fit), MIN_H_PER_COMPONENT) * rank + PANEL_H_EXTRA
    )
    fig.set_size_inches(width, height + n_lines * panel_h)
    _frame(fig)
    top = _block_head(fig, "", _effects_key(), EDGE)
    for title, block_axes, block_feet in zip(titles, axes, feet):
        top = _block_head(fig, title, (), top)
        for line, line_foot in zip(lines(block_axes), block_feet):
            for slot, ax in enumerate(line):
                _place(
                    fig, ax, left + slot * (panel_w + PANEL_GAP), top + head_h, panel_w, panel_h,
                )  # fmt: skip
            top += head_h + panel_h + line_foot
        top += SPACING
    for block_axes, block_labels, block in zip(axes, x_labels, blocks):
        for ax, label, (_, level) in zip(block_axes, block_labels, block.pairs):
            ax.set_xlabel(label, fontsize=X_LABEL_SIZE, labelpad=2)
            if level:
                _direction(ax, level)
                ax.xaxis.labelpad = 10
    return fig


def _heatmap(
    ax: Axes,
    r: np.ndarray,
    p: np.ndarray,
    not_available: Sequence[bool] = (),
) -> AxesImage:
    """Draw `r` (terms x components) with its values, boxed and with their `p` where it is
    below `FILLED_P` ("n.t." where `r` is missing), and the rows marked in `not_available` as
    "N/A"."""
    image = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            value, p_value = r[i, j], p[i, j]
            if i < len(not_available) and not_available[i]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=NOT_AVAILABLE_COLOR,
                        edgecolor="none",
                        zorder=2,
                    ),
                )
                ax.text(
                    j,
                    i,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=TEXT_COLOR,
                )
                continue
            if np.isnan(value):
                ax.text(j, i, "n.t.", ha="center", va="center", fontsize=6, color="0.5")
                continue
            significant = p_value < FILLED_P
            color = "white" if abs(value) > 0.55 else "black"
            ax.text(
                j, i - (0.13 if significant else 0), f"{value:.2f}", ha="center", va="center",
                fontsize=7, color=color, fontweight="bold" if significant else "normal",
            )  # fmt: skip
            if significant:
                ax.text(
                    j, i + 0.22, p_text(p_value), ha="center", va="center", fontsize=5.5,
                    color=color,
                )  # fmt: skip
            if significant:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.6,
                    ),
                )
    return image


def correlation_label(covariates: Sequence[str]) -> str:
    """The colourbar label of the partial correlations, adjusted for `covariates`."""
    return (
        f"Correlation (adjusted for {', '.join(covariates)})"
        if covariates
        else "Correlation"
    )


class _CorrelationBlock(NamedTuple):
    """A block of a correlation figure: its `title`, the numerical terms of `table` it shows,
    and the display names of its categorical ones."""

    title: str
    table: pd.DataFrame
    terms: list[str]
    categorical: list[str]


def plot_partial_r(
    table: pd.DataFrame,
    numerical_terms: Sequence[str],
    covariates: Sequence[str] = (),
    categories: dict[str, list[str]] | None = None,
    names: dict[str, str] | None = None,
) -> Figure | None:
    """Heatmaps of the partial correlation of each of the `numerical_terms` of a clinical table
    with each component, adjusted for `covariates`, boxed and with its permutation p where that
    is below `FILLED_P` ("n.t." where it was not tested); terms and covariates are shown by
    `names`.
    With `categories` (as for `plot_effects`) one block per category with numerical terms,
    titled A, B, ...; otherwise one block, untitled as is a figure's only block. The blocks
    fill lines of `CORRELATION_COLUMNS`, each column `PAGE_W` / `CORRELATION_COLUMNS` wide (a
    lone block wider where its title or key need it); each has a colourbar on top and a key
    naming its categorical terms, which have no partial r. None without numerical terms.
    """
    names = names or {}
    blocks = [
        _CorrelationBlock(
            title, table, [t for t in members if t in numerical_terms],
            [names.get(t, t) for t in members if t not in numerical_terms],
        )
        for title, members in _blocks(list(dict.fromkeys(table["term"])), categories)
    ]  # fmt: skip
    return _draw_partial_r(blocks, covariates, names)


def plot_interaction_partial_r(
    blocks: Sequence[InteractionBlock],
    title: str,
    group: str,
    terms: Sequence[str],
    covariates: Sequence[str] = (),
    names: dict[str, str] | None = None,
) -> Figure | None:
    """`plot_partial_r` of the slope blocks of the `interaction_blocks` of the `terms` of
    category `title`, one per level of `group` ("title: PD"), with the correlations within that
    level: a row per term, "N/A" where the level has no slope for it. The colourbar names the
    `covariates` but `group`, which only gives each level its own mean. None without slopes.
    """
    return _draw_partial_r(
        [
            _CorrelationBlock(f"{title}: {block.level}", block.rows, list(terms), [])
            for block in blocks
            if block.test == "slope"
        ],
        [covariate for covariate in covariates if covariate != group],
        names or {},
    )


def _correlation_key(block: _CorrelationBlock) -> list[Patch | Line2D]:
    """The key of a correlation block: the box of a significant correlation, the N/A of terms
    without rows, and its categorical terms."""
    handles: list[Patch | Line2D] = [
        Patch(
            facecolor="white", edgecolor="black", linewidth=1.6,
            label=rf"$p_{{perm}}<{FILLED_P}$ (uncorrected)",
        ),
    ]  # fmt: skip
    if not set(block.terms) <= set(block.table["term"]):
        handles.append(
            Patch(
                facecolor=NOT_AVAILABLE_COLOR,
                edgecolor="none",
                label="N/A: no slope in this group",
            ),
        )
    if block.categorical:
        text = f"No correlation (categorical): {', '.join(block.categorical)}"
        handles.append(Line2D([], [], linestyle="", marker="", label=text))
    return handles


def _draw_partial_r(
    blocks: Sequence[_CorrelationBlock],
    covariates: Sequence[str],
    names: dict[str, str],
) -> Figure | None:
    blocks = [block for block in blocks if block.terms]
    if not blocks:
        return None
    if len(blocks) == 1:
        blocks = [blocks[0]._replace(title="")]
    rank = max(int(block.table["component"].max()) for block in blocks)
    components = list(range(1, rank + 1))
    # A term without rows (no slope in the group of an interaction block) has the block's n.
    row_labels = []
    for block in blocks:
        n = block.table.drop_duplicates("term").set_index("term")["n"]
        row_labels.append(
            [f"{names.get(t, t)} (n={int(n.get(t, n.iloc[0]))})" for t in block.terms],
        )
    # Every heatmap starts after the widest row labels of its slot of the line.
    label_w = (
        _text_width([label for labels_ in row_labels for label in labels_], 7)
        + LABEL_PAD
    )
    titles = _lettered([block.title for block in blocks])
    keys = [_correlation_key(block) for block in blocks]
    columns = min(len(blocks), CORRELATION_COLUMNS)
    width = _figure_width(
        columns, CORRELATION_COLUMNS, EDGE + _head_width(titles, keys) + PANEL_GAP,
    )  # fmt: skip
    slot_w = (width - EDGE - PANEL_GAP - (columns - 1) * BLOCK_GAP) / columns
    map_w = slot_w - label_w
    above = COLORBAR_LABEL_H + COLORBAR_H + COLORBAR_GAP
    heights = [
        (TITLE_H if block.title else 0) + KEY_H + above + HEATMAP_CELL_H * len(block.terms)
        + HEATMAP_TICKS_H
        for block in blocks
    ]  # fmt: skip
    line_heights = [
        max(heights[i : i + columns]) for i in range(0, len(blocks), columns)
    ]
    fig = plt.figure(
        figsize=(
            width,
            2 * EDGE + sum(line_heights) + SPACING * (len(line_heights) - 1),
        ),
    )
    _frame(fig)
    label = correlation_label([names.get(c, c) for c in covariates])
    for i, (block, title, handles, block_labels) in enumerate(
        zip(blocks, titles, keys, row_labels),
    ):
        line, slot = divmod(i, columns)
        x = EDGE + slot * (slot_w + BLOCK_GAP)
        left = x + label_w
        top = EDGE + sum(line_heights[:line]) + SPACING * line
        not_available = [t not in set(block.table["term"]) for t in block.terms]
        bar_top = _block_head(fig, title, handles, top, x) + COLORBAR_LABEL_H
        ax = fig.add_axes((0, 0, 1, 1))
        cax = fig.add_axes((0, 0, 1, 1))
        _place(fig, cax, left, bar_top, map_w, COLORBAR_H)
        _place(
            fig,
            ax,
            left,
            bar_top + COLORBAR_H + COLORBAR_GAP,
            map_w,
            HEATMAP_CELL_H * len(block.terms),
        )
        part = block.table[block.table["term"].isin(block.terms)]
        r = part.pivot(index="term", columns="component", values="partial_r")
        p = part.pivot(index="term", columns="component", values="p_perm")
        image = _heatmap(
            ax,
            r.reindex(index=block.terms, columns=components).to_numpy(float),
            p.reindex(index=block.terms, columns=components).to_numpy(float),
            not_available,
        )
        ax.set_xticks(range(rank), [str(j) for j in components])
        ax.set_xlabel("Component", fontsize=7, labelpad=2)
        ax.set_yticks(range(len(block.terms)), block_labels)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(top=False, right=False, labelsize=7)
        bar = fig.colorbar(
            image,
            cax=cax,
            orientation="horizontal",
            ticks=[-1, -0.5, 0, 0.5, 1],
        )
        bar.set_label(label, fontsize=7)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=6)
        cax.tick_params(which="minor", top=False, bottom=False)
    return fig


def plot_added_variables(
    residuals: pd.DataFrame,
    table: pd.DataFrame,
    terms: Sequence[str],
    units: dict[str, str] | None = None,
    names: dict[str, str] | None = None,
) -> Figure | None:
    """The added-variable plots of one category's clinical `terms`: a column per term with
    `added_variables` rows (categorical terms have none), a row per component, each panel the
    component's coefficient against the term, both adjusted for the covariates of the term's
    rows in `table` (in their own units: `units`, else a.u.), with the fitted slope, titled
    with the partial r and its 95% CI (bold where its permutation p is below `FILLED_P`).
    None without rows."""
    columns = []
    for term in terms:
        rows = residuals[residuals["term"] == term]
        if not rows.empty:
            stats = table[table["term"] == term]
            columns.append((f"n={int(stats['n'].iloc[0])}", term, rows, stats))
    return _draw_added_variables(columns, units or {}, names or {}, False)


def plot_interaction_added_variables(
    residuals: pd.DataFrame,
    table: pd.DataFrame,
    terms: Sequence[str],
    group: str,
    units: dict[str, str] | None = None,
    names: dict[str, str] | None = None,
) -> Figure | None:
    """`plot_added_variables` of the slopes within `group` (a level) of the `terms` in an
    interactions table, a column per term with a slope there: the pooled model's residuals,
    the group's subjects in the component's colour and the other groups' in grey. None
    without slopes."""
    level = f"slope in {group}"
    columns = []
    for term in terms:
        rows = residuals[(residuals["term"] == term) & (residuals["level"] == level)]
        if rows.empty:
            continue
        stats = table[
            (table["term"].str.split(" × ").str[0] == term) & (table["level"] == level)
        ]
        n = int(
            rows.loc[rows["component"] == rows["component"].min(), "in_group"].sum(),
        )
        columns.append((f"{group} (n={n})", term, rows, stats))
    return _draw_added_variables(columns, units or {}, names or {}, True)


def _added_variable_panel(
    ax: Axes,
    points: pd.DataFrame,
    row: pd.Series,
    component: int,
    others: bool,
) -> None:
    inside = points["in_group"].to_numpy(bool)
    x, y = points["x"].to_numpy(float), points["y"].to_numpy(float)
    color = component_color(component)
    if others:
        ax.scatter(x[~inside], y[~inside], s=6, color=OTHER_GROUP_COLOR, linewidths=0)
    ax.scatter(
        x[inside],
        y[inside],
        s=6,
        color=color,
        alpha=0.75,
        linewidths=0,
        zorder=2,
    )
    span = np.array([x.min(), x.max()])
    ax.plot(span, (x @ y / (x @ x)) * span, color=color, linewidth=1.2, zorder=3)
    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.6, zorder=0)
    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.6, zorder=0)
    ax.set_title(
        f"r = {row['partial_r']:.2f} [{row['partial_r_ci_low']:.2f}, "
        f"{row['partial_r_ci_high']:.2f}]",
        fontsize=6.5, pad=3, fontweight="bold" if row["p_perm"] < FILLED_P else "normal",
    )  # fmt: skip
    ax.tick_params(labelsize=6)
    ax.tick_params(which="minor", left=False, right=False, top=False, bottom=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def _draw_added_variables(
    columns: Sequence[tuple[str, str, pd.DataFrame, pd.DataFrame]],
    units: dict[str, str],
    names: dict[str, str],
    others: bool,
) -> Figure | None:
    """A column per `(title, term, residuals, table rows)` of `columns`, a row per component,
    each column `PAGE_W` / `COLUMNS` wide (to `PAGE_W` at most); the rows share their y range
    and the columns their x range, and the panels are as tall as fit in `MAX_H`."""
    if not columns:
        return None
    rank = max(int(rows["component"].max()) for _, _, rows, _ in columns)
    width = _figure_width(len(columns), COLUMNS)
    left = EDGE + AV_Y_LABEL_W + _text_width(["-0.05"], 6) + LABEL_PAD
    panel_w = (width - left) / len(columns) - PANEL_GAP
    x_labels = [
        _wrap(
            f"{names.get(term, term)}, adjusted ({units.get(term) or 'a.u.'})", panel_w,
            X_LABEL_SIZE,
        )
        for _, term, _, _ in columns
    ]  # fmt: skip
    foot = FOOT_H + X_LABEL_LINE_H * max(label.count("\n") for label in x_labels)
    fixed = 2 * EDGE + N_TITLE_H + rank * (N_TITLE_H + AV_ROW_GAP) + foot
    panel_h = max(min(AV_PANEL_H, (MAX_H - fixed) / rank), MIN_AV_PANEL_H)
    fig = plt.figure(figsize=(width, fixed + rank * panel_h))
    _frame(fig)
    fig_w, fig_h = fig.get_size_inches()
    for c, (title, _, _, _) in enumerate(columns):
        x = left + c * (panel_w + PANEL_GAP) + panel_w / 2
        fig.text(
            x / fig_w,
            1 - EDGE / fig_h,
            title,
            ha="center",
            va="top",
            fontsize=7.5,
        )
    grid: list[list[Axes]] = []
    for k in range(1, rank + 1):
        top = (
            EDGE + N_TITLE_H + (k - 1) * (N_TITLE_H + panel_h + AV_ROW_GAP) + N_TITLE_H
        )
        row_axes = []
        for c, (_, _, rows, stats) in enumerate(columns):
            ax = fig.add_axes((0, 0, 1, 1))
            _place(fig, ax, left + c * (panel_w + PANEL_GAP), top, panel_w, panel_h)
            _added_variable_panel(
                ax, rows[rows["component"] == k], stats[stats["component"] == k].iloc[0], k,
                others,
            )  # fmt: skip
            ax.tick_params(labelbottom=k == rank, labelleft=c == 0)
            if k == rank:
                ax.set_xlabel(x_labels[c], fontsize=X_LABEL_SIZE, labelpad=2)
            row_axes.append(ax)
        row_axes[0].set_ylabel(
            f"Component {k}\nadjusted (a.u.)",
            fontsize=7,
            labelpad=2,
        )
        grid.append(row_axes)
    for row_axes in grid:
        lows, highs = zip(*(ax.get_ylim() for ax in row_axes))
        for ax in row_axes:
            ax.set_ylim(min(lows), max(highs))
    for column_axes in zip(*grid):
        lows, highs = zip(*(ax.get_xlim() for ax in column_axes))
        for ax in column_axes:
            ax.set_xlim(min(lows), max(highs))
    return fig
