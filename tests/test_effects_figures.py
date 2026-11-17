"""The effects, partial-r and added-variable figures of 05 (plot_stats.figures), drawn from
gMRItensor's clinical and interaction tables of synthetic subjects."""

import re

import numpy as np
import pandas as pd
import pytest
from gMRItensor.subject_statistics import clinical_effects, interaction_effects
from matplotlib.figure import Figure
from matplotlib.text import Text
from plot_stats.data import added_variables, interaction_blocks
from plot_stats.figures import (
    plot_added_variables,
    plot_effects,
    plot_interaction_added_variables,
    plot_interaction_effects,
    plot_interaction_partial_r,
    plot_partial_r,
)

PRIMARY = ["Group", "Sex", "Age"]
CATEGORICAL = ["Group", "Sex"]
CLINICAL = ["Height", "ventricles_volume_ml"]
CATEGORIES = {"Anthropometrics": ["Height"], "Volume (mL)": ["ventricles_volume_ml"]}


def subjects(rank: int = 3, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """92 subjects (49 REF, 43 iNPH) and their coefficients; c1 grows with the ventricles."""
    rng = np.random.default_rng(seed)
    n = 92
    group = np.array(["REF"] * 49 + ["iNPH"] * 43)
    ventricles = np.where(group == "iNPH", rng.normal(90, 20, n), rng.normal(35, 10, n))
    info = pd.DataFrame(
        {
            "subjects": [f"S{i:03d}" for i in range(n)],
            "Group": group,
            "Sex": rng.choice(["F", "M"], n),
            "Age": rng.normal(68, 9, n),
            "Height": rng.normal(172, 9, n),
            "ventricles_volume_ml": ventricles,
        },
    )
    coefficients = 1 + rng.normal(0, 0.2, (n, rank))
    coefficients[:, 0] += 0.01 * ventricles
    return info, coefficients


def frame(info: pd.DataFrame, coefficients: np.ndarray) -> pd.DataFrame:
    columns = {f"c{k + 1}": coefficients[:, k] for k in range(coefficients.shape[1])}
    return info.assign(**columns)


@pytest.fixture(scope="module")
def figures() -> list[Figure]:
    info, coefficients = subjects()
    clinical = clinical_effects(
        coefficients, info, CLINICAL, PRIMARY, CATEGORICAL, n_permutations=99,
    )  # fmt: skip
    interactions = interaction_effects(
        coefficients, info, [("ventricles_volume_ml", "Group")], PRIMARY, CATEGORICAL,
        n_permutations=99,
    )  # fmt: skip
    subject_frame = frame(info, coefficients)
    residuals = added_variables(subject_frame, clinical, CATEGORICAL)
    blocks = interaction_blocks(interactions, ["ventricles_volume_ml"], "Group")
    slopes = interactions[interactions["test"] == "slope"]
    slope_residuals = added_variables(subject_frame, slopes, CATEGORICAL)
    drawn = [
        plot_effects(clinical, categories=CATEGORIES),
        plot_partial_r(clinical, CLINICAL, PRIMARY, CATEGORIES),
        plot_added_variables(residuals, clinical, ["Height"]),
        plot_interaction_effects(blocks, "Volume (mL)", ["ventricles_volume_ml"]),
        plot_interaction_partial_r(
            blocks, "Volume (mL)", "Group", ["ventricles_volume_ml"], PRIMARY,
        ),  # fmt: skip
        plot_interaction_added_variables(
            slope_residuals, interactions, ["ventricles_volume_ml"], "iNPH",
        ),  # fmt: skip
    ]
    assert all(figure is not None for figure in drawn)
    return drawn


def texts(figure: Figure) -> list[str]:
    return [t.get_text() for t in figure.findobj(Text) if t.get_text()]


def test_no_figure_counts_subjects(figures: list[Figure]) -> None:
    for figure in figures:
        assert not [t for t in texts(figure) if re.search(r"\bn\s*=", t)]


def test_no_figure_calls_its_p_values_uncorrected(figures: list[Figure]) -> None:
    for figure in figures:
        assert not [t for t in texts(figure) if "uncorrected" in t]


def test_the_keys_still_name_the_permutation_p(figures: list[Figure]) -> None:
    effects, partial_r = figures[0], figures[1]
    assert r"$p_{perm}<0.05$" in texts(effects)
    assert r"$p_{perm}\geq0.05$" in texts(effects)
    assert r"$p_{perm}<0.05$" in texts(partial_r)


def test_the_interaction_added_variables_still_name_their_group(
    figures: list[Figure],
) -> None:
    assert "iNPH" in texts(figures[-1])
