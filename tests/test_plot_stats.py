"""The figure of the coefficients by primary variable (plot_stats.figures.plot_primary_groups)."""

import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.text import Text
from plot_stats.figures import (
    BOX_LINE,
    GROUP_FILL_ALPHA,
    GROUPS_FOOT,
    GROUPS_PANEL_H,
    GROUPS_TOP,
    MAX_H,
    MIN_GROUPS_PANEL_H,
    PAGE_W,
    plot_primary_groups,
)

VARIABLES = ["Group", "Sex", "Age"]


def subjects(rank: int, seed: int = 0) -> pd.DataFrame:
    """92 subjects, 49 REF and 43 iNPH, with `rank` components of positive coefficients."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "subjects": [f"S{i:03d}" for i in range(92)],
            "Group": ["REF"] * 49 + ["iNPH"] * 43,
            "Sex": rng.choice(["F", "M"], 92),
            "Age": rng.uniform(40, 85, 92),
        },
    )
    for c in range(rank):
        frame[f"c{c + 1}"] = rng.gamma(4, 0.03, 92)
    return frame


def draw(rank: int = 3) -> Figure:
    fig = plot_primary_groups(
        subjects(rank), ["Group", "Sex"], ["Age"], "Group", "Sex", {"Age": "years"}, {},
    )  # fmt: skip
    fig.canvas.draw()
    return fig


def panels(fig: Figure, rank: int) -> np.ndarray:
    return np.array(fig.axes).reshape(rank, len(VARIABLES))


@pytest.mark.parametrize("rank", [2, 3])
def test_small_ranks_keep_the_regular_panel_height(rank: int) -> None:
    width, height = draw(rank).get_size_inches()
    assert width == pytest.approx(PAGE_W)
    assert height == pytest.approx(GROUPS_TOP + GROUPS_FOOT + rank * GROUPS_PANEL_H)
    assert height <= MAX_H


def test_large_ranks_shrink_to_fit_the_page() -> None:
    height = draw(6).get_size_inches()[1]
    assert height == pytest.approx(MAX_H)
    assert height < GROUPS_TOP + GROUPS_FOOT + 6 * GROUPS_PANEL_H


def test_panels_do_not_shrink_below_the_minimum() -> None:
    height = draw(12).get_size_inches()[1]
    assert height == pytest.approx(GROUPS_TOP + GROUPS_FOOT + 12 * MIN_GROUPS_PANEL_H)
    assert height > MAX_H


def test_boxes_have_transparent_fills_and_solid_outlines_as_thick_as_the_median() -> (
    None
):
    grid = panels(draw(2), 2)
    group_ax, sex_ax, _ = grid[0]
    for ax, faces in ((group_ax, ["#0c5da5", "#ff2c00"]), (sex_ax, ["0.88", "0.88"])):
        boxes = [p for p in ax.patches if isinstance(p, PathPatch)]
        assert len(boxes) == 2
        for box, face in zip(boxes, faces):
            alpha = GROUP_FILL_ALPHA if ax is group_ax else 1.0
            assert box.get_facecolor() == pytest.approx(to_rgba(face, alpha), abs=0.01)
            assert box.get_edgecolor()[3] == 1.0
            assert box.get_linewidth() == pytest.approx(BOX_LINE)
        assert {line.get_linewidth() for line in ax.lines} == {
            BOX_LINE,
        }  # median, whiskers, caps
    assert [t.get_text() for t in grid[-1, 0].get_xticklabels()] == [
        "REF",
        "iNPH",
    ]  # no "All"


def test_dots_are_shaped_by_sex_without_outlines() -> None:
    for ax in panels(draw(2), 2)[0]:
        dots = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(dots) == 2  # one per sex
        assert sorted(float(d.get_sizes()[0]) for d in dots) == [14.0, 18.0]
        assert all(np.all(d.get_linewidths() == 0) for d in dots)


def test_age_panels_have_one_fit_and_an_r2_label_but_no_legend() -> None:
    for ax in panels(draw(3), 3)[:, 2]:
        assert len(ax.lines) == 1 and ax.lines[0].get_color() == "black"
        labels = [t.get_text() for t in ax.texts]
        assert len(labels) == 1 and re.fullmatch(r"\$R\^2 = \d\.\d\d\$", labels[0])
        assert ax.texts[0].get_fontsize() == pytest.approx(8.5)
        assert ax.get_legend() is None


def test_no_p_values_anywhere() -> None:
    texts = [t.get_text() for t in draw(3).findobj(Text)]
    assert not [t for t in texts if re.search(r"(^|\W)p\s*[=<]", t)]


def test_rows_share_y_and_only_the_last_row_has_x_labels() -> None:
    grid = panels(draw(3), 3)
    for row in grid:
        assert all(row[0].get_shared_y_axes().joined(row[0], ax) for ax in row[1:])
    for r, row in enumerate(grid):
        for ax in row:
            visible = [t.get_visible() and t.get_text() for t in ax.get_xticklabels()]
            assert any(visible) == (r == len(grid) - 1)
    assert [ax.get_title() for ax in grid[0]] == VARIABLES
    assert grid[-1, 2].get_xlabel() == "Age (years)"


def test_one_legend_row_of_groups_and_sexes() -> None:
    (legend,) = draw(2).legends
    assert [t.get_text() for t in legend.get_texts()] == ["REF", "iNPH", "F", "M"]


@pytest.mark.parametrize("rank", [3, 6, 12])
def test_row_labels_never_overlap(rank: int) -> None:
    fig = draw(rank)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    grid = panels(fig, rank)
    labels = [ax.yaxis.label for ax in grid[:, 0]]
    assert [label.get_text() for label in labels] == [
        f"Component {r + 1}" for r in range(rank)
    ]
    boxes = [label.get_window_extent(renderer) for label in labels]
    for upper, lower in zip(boxes, boxes[1:]):
        assert upper.y0 > lower.y1  # rows run top to bottom
    assert fig.get_supylabel() == "Subject coefficient (a.u.)"


def test_row_labels_are_aligned() -> None:
    """Rows of different scales have tick labels of different widths."""
    frame = subjects(3)
    frame["c2"] *= 100
    frame["c3"] *= 0.01
    fig = plot_primary_groups(frame, ["Group", "Sex"], ["Age"], "Group", "Sex")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    lefts = {
        round(ax.yaxis.label.get_window_extent(renderer).x0, 1)
        for ax in panels(fig, 3)[:, 0]
    }
    assert len(lefts) == 1
