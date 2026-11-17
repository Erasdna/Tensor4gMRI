"""Correlations between components' subject coefficients (code/plot_stats/components.py, its CLI
code/plot_stats/component_correlations.py, and plot_stats.figures.plot_component_correlations)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from plot_stats.components import PAIR_COLUMNS, component_correlations
from plot_stats.figures import PAD, plot_component_correlations

REPO = Path(__file__).resolve().parents[1]
MAX_W = 170 / 25.4 - 2 * PAD


def subjects(rank: int = 3, seed: int = 0) -> pd.DataFrame:
    """92 subjects; c2 follows c1 beyond Group and Age, c3 is independent of both."""
    rng = np.random.default_rng(seed)
    group = np.array(["REF"] * 49 + ["iNPH"] * 43)
    age = np.where(group == "REF", rng.normal(58, 10, 92), rng.normal(74, 6, 92))
    shared = 0.3 * (group == "iNPH") + 0.01 * (age - 65)
    c1 = 1 + shared + rng.normal(0, 0.2, 92)
    frame = pd.DataFrame(
        {
            "subjects": [f"S{i:03d}" for i in range(92)],
            "Group": group,
            "Sex": rng.choice(["F", "M"], 92),
            "Age": age,
            "c1": c1,
            "c2": 1 - shared + 0.8 * c1 + rng.normal(0, 0.1, 92),
            "c3": 1 + rng.normal(0, 0.2, 92),
        },
    )
    return frame[[*frame.columns[:4], *[f"c{k + 1}" for k in range(rank)]]]


def residuals(values: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["Group"] == "iNPH",
            frame["Sex"] == "M",
            frame["Age"],
        ],
    ).astype(float)
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


@pytest.fixture(scope="module")
def pairs() -> pd.DataFrame:
    return component_correlations(
        subjects(),
        ["Group", "Sex"],
        ["Age"],
        n_permutations=500,
    )


def test_one_row_per_pair_in_order(pairs: pd.DataFrame) -> None:
    assert list(pairs.columns) == PAIR_COLUMNS
    assert list(zip(pairs["component_a"], pairs["component_b"])) == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert (pairs["n"] == 92).all()


def test_partial_r_is_the_correlation_adjusted_for_the_primary_variables(
    pairs: pd.DataFrame,
) -> None:
    frame = subjects()
    for row in pairs.itertuples():
        a = residuals(frame[f"c{row.component_a}"].to_numpy(), frame)
        b = residuals(frame[f"c{row.component_b}"].to_numpy(), frame)
        assert row.partial_r == pytest.approx(np.corrcoef(a, b)[0, 1])
        unadjusted = np.corrcoef(
            frame[f"c{row.component_a}"],
            frame[f"c{row.component_b}"],
        )
        assert row.r_unadjusted == pytest.approx(unadjusted[0, 1])
        assert row.partial_r_ci_low < row.partial_r < row.partial_r_ci_high


def test_only_the_coupled_pair_is_significant(pairs: pd.DataFrame) -> None:
    coupled = pairs.set_index(["component_a", "component_b"])
    assert (
        coupled.loc[(1, 2), "partial_r"] > 0.5 and coupled.loc[(1, 2), "p_perm"] < 0.01
    )
    assert (pairs["p_fdr"] >= pairs["p_perm"] - 1e-12).all()
    assert pairs["significant"].tolist() == [True, False, False]


def test_one_component_has_no_pairs() -> None:
    table = component_correlations(
        subjects(1),
        ["Group", "Sex"],
        ["Age"],
        n_permutations=100,
    )
    assert table.empty and list(table.columns) == PAIR_COLUMNS
    assert plot_component_correlations(table) is None


def test_cli_writes_the_table(tmp_path: Path) -> None:
    from gMRItensor.model_io import SavedDecomposition, save_decomposition
    from plot_stats.data import subject_frame

    frame = subjects()
    coefficients = frame[["c1", "c2", "c3"]].to_numpy()
    model = tmp_path / "cp_rank_3"
    model.mkdir()
    save_decomposition(
        model / "model.h5",
        SavedDecomposition(
            method="cp", rank=3, error=0.1, weights=np.ones(3), subject_mode=coefficients,
            label_mode=np.ones((4, 3)), subjects=frame["subjects"].to_numpy(),
            timepoints=np.array([2, 4, 8, 24]), time_mode=np.ones((4, 3)),
            labels=np.arange(4), label_index=np.zeros(4, dtype=int),
        ),
    )  # fmt: skip
    info = tmp_path / "subject_info.csv"
    frame.drop(columns=["c1", "c2", "c3"]).to_csv(info, index=False)
    result = subprocess.run(
        [
            sys.executable, str(REPO / "code/plot_stats/component_correlations.py"),
            "--model", str(model), "--subject-info", str(info), "--output-dir", str(tmp_path),
            "--primary-categorical", "Group", "Sex", "--primary-numerical", "Age",
            "--n-permutations", "500", "--seed", "0", "--alpha", "0.05",
        ],
        env={**os.environ, "PYTHONPATH": str(REPO / "code")},
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(
        tmp_path / "statistics/cp_rank_3__subject_stats_components.csv",
    )
    assert written[["component_a", "component_b"]].values.tolist() == [
        [1, 2],
        [1, 3],
        [2, 3],
    ]
    # gmri reloads the model with its components sorted largest first
    expected = component_correlations(
        subject_frame(model, info), ["Group", "Sex"], ["Age"], n_permutations=500,
    )  # fmt: skip
    assert written["partial_r"].to_numpy() == pytest.approx(
        expected["partial_r"].to_numpy(),
    )
    assert written["p_perm"].to_numpy() == pytest.approx(expected["p_perm"].to_numpy())


def made_up_pairs(rank: int, coupled: dict[tuple[int, int], float]) -> pd.DataFrame:
    rows = []
    for a in range(1, rank + 1):
        for b in range(a + 1, rank + 1):
            r = coupled.get((a, b), 0.05)
            rows.append(
                {
                    "component_a": a, "component_b": b, "partial_r": r,
                    "partial_r_ci_low": r - 0.2, "partial_r_ci_high": r + 0.2,
                    "p_perm": 0.001 if (a, b) in coupled else 0.6,
                },
            )  # fmt: skip
    return pd.DataFrame(rows).reindex(columns=PAIR_COLUMNS)


def test_figure_draws_the_lower_triangle_with_ci_and_p_in_every_cell() -> None:
    fig = plot_component_correlations(made_up_pairs(4, {(1, 4): 0.74}))
    assert fig is not None
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    estimates = [t for t in texts if "[" in t]
    assert len(estimates) == 6  # pairs of 4 components
    assert "0.74 [0.54, 0.94]" in estimates
    assert [t for t in texts if t.startswith("$p_{perm}")] == [
        t for t in texts if "perm" in t
    ] and len([t for t in texts if "perm" in t]) == 6
    bold = [t for t in ax.texts if t.get_fontweight() == "bold"]
    assert [t.get_text() for t in bold] == ["0.74 [0.54, 0.94]"]
    thick = [
        p for p in ax.patches if isinstance(p, Rectangle) and p.get_linewidth() > 1
    ]
    assert len(thick) == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [r"$p_{perm} < 0.05$"]
    assert fig.axes[1].get_ylabel() == "Correlation"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["2", "3", "4"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "3"]
    labels = [t.get_text() for t in fig.findobj(Text)]
    assert not [t for t in labels if "uncorrected" in t or "adjusted" in t]


@pytest.mark.parametrize("rank", [3, 6, 8])
def test_figure_is_at_most_17_cm_wide(rank: int) -> None:
    fig = plot_component_correlations(made_up_pairs(rank, {(1, 2): 0.6}))
    assert fig is not None
    assert fig.get_figwidth() <= MAX_W + 1e-9


def test_p_perm_text() -> None:
    fig = plot_component_correlations(
        made_up_pairs(2, {(1, 2): 0.6}).assign(p_perm=0.0004),
    )
    assert fig is not None
    assert r"$p_{perm}<0.001$" in [t.get_text() for t in fig.axes[0].texts]
