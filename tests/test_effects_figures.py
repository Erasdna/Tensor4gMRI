"""The effects, partial-r and added-variable figures of 05 (plot_stats.figures), the
`_p_fdr.csv` sheets next to them (plot_stats.data.fdr_sheet) and the associations with
p_perm < 0.05 it prints (plot_stats.data.association_lines), from gMRItensor's primary,
clinical and interaction tables of synthetic subjects."""

import re

import numpy as np
import pandas as pd
import pytest
from gMRItensor.subject_statistics import (
    clinical_effects,
    interaction_effects,
    primary_effects,
)
from matplotlib.figure import Figure
from matplotlib.text import Text
from plot_stats.data import (
    added_variables,
    association_lines,
    associations,
    family_fdr,
    fdr_sheet,
    interaction_blocks,
    one_row_per_test,
    standardised_effects,
)
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


R_COLUMNS = ("partial_r", "partial_r_ci_low", "partial_r_ci_high")
D_COLUMNS = ("d", "d_ci_low", "d_ci_high")


def per_component(columns: tuple[str, ...], k: int) -> list[str]:
    return [f"c{k}_{column}" for column in columns]


def test_the_primary_sheet_holds_the_plotted_d_and_its_ci() -> None:
    info, coefficients = subjects()
    rank = coefficients.shape[1]
    primary = primary_effects(
        coefficients,
        info,
        PRIMARY,
        CATEGORICAL,
        n_permutations=99,
    )
    sheet = fdr_sheet(family_fdr(primary, {"Primary variables": PRIMARY}))
    columns = (*R_COLUMNS, *D_COLUMNS, "p_perm", "p_fdr")
    assert list(sheet.columns) == [
        "category",
        "variable",
        *[c for k in range(1, rank + 1) for c in per_component(columns, k)],
    ]
    sheet = sheet.set_index("variable")
    plotted = standardised_effects(primary).set_index(["term", "component"])
    for term in CATEGORICAL:
        for k in range(1, rank + 1):
            row = plotted.loc[(term, k)]
            assert row["effect_kind"] == "d"
            assert sheet.loc[
                term,
                per_component(D_COLUMNS, k),
            ].tolist() == pytest.approx(
                [row["effect"], row["effect_low"], row["effect_high"]],
            )
            assert sheet.loc[term, per_component(R_COLUMNS, k)].isna().all()
    for k in range(1, rank + 1):
        assert sheet.loc["Age", per_component(R_COLUMNS, k)].notna().all()
        assert sheet.loc["Age", per_component(D_COLUMNS, k)].isna().all()


def test_the_slope_sheet_has_no_d_columns() -> None:
    """Interaction slopes are of numerical variables: a partial r, never a d."""
    info, coefficients = subjects()
    interactions = interaction_effects(
        coefficients, info, [("ventricles_volume_ml", "Group")], PRIMARY, CATEGORICAL,
        n_permutations=99,
    )  # fmt: skip
    slopes = family_fdr(
        interactions[interactions["test"] == "slope"],
        {"Volume (mL)": ["ventricles_volume_ml × Group"]},
        keys=("component", "term", "level"),
    )
    sheet = fdr_sheet(slopes, group=True)
    assert len(sheet)
    assert set(per_component(R_COLUMNS, 1)) <= set(sheet.columns)
    assert not [c for c in sheet.columns if re.search(r"_d(_ci_\w+)?$", c)]


def test_associations_are_the_tests_below_the_threshold() -> None:
    info, coefficients = subjects()
    primary = primary_effects(
        coefficients, info, PRIMARY, CATEGORICAL, n_permutations=99,
    )  # fmt: skip
    clinical = clinical_effects(
        coefficients, info, CLINICAL, PRIMARY, CATEGORICAL, n_permutations=99,
    )  # fmt: skip
    tests = pd.concat(
        [
            one_row_per_test(family_fdr(primary, {"Primary variables": PRIMARY})),
            one_row_per_test(family_fdr(clinical, CATEGORIES)),
        ],
        ignore_index=True,
    )
    hits = associations(tests, 0.05)
    below = tests[tests["p_perm"] < 0.05]
    assert 0 < len(hits) < len(tests)
    assert sorted(zip(hits["variable"], hits["component"])) == sorted(
        zip(below["variable"], below["component"]),
    )
    stats = pd.concat([primary, clinical]).set_index(["term", "component"])
    for row in hits.itertuples():
        stat = stats.loc[(row.variable, row.component)]
        if row.variable in CATEGORICAL:
            assert (row.effect_kind, row.effect) == (
                "d",
                pytest.approx(stat["standardised"]),
            )
            assert row.effect_low / row.effect == pytest.approx(
                stat["ci_low"] / stat["estimate"],
            )
        else:
            assert (row.effect_kind, row.effect) == (
                "r",
                pytest.approx(stat["partial_r"]),
            )
            assert row.effect_low == pytest.approx(stat["partial_r_ci_low"])


def hand_made_tests() -> pd.DataFrame:
    """`one_row_per_test` rows as family_fdr orders slopes: component by component."""
    columns = [
        "category", "variable", "group", "component", "effect_kind", "effect", "effect_low",
        "effect_high", "p_perm", "p_fdr",
    ]  # fmt: skip
    slopes = "Volume (mL) by Group"
    rows = [
        ("Primary variables", "Group", "", 1, "d", 2.091, 1.67, 2.512, 0.001, 0.012),
        (
            "Primary variables",
            "Sex",
            "",
            1,
            "d",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ),
        ("Primary variables", "Age", "", 1, "r", 0.1, -0.1, 0.3, 0.2, 0.4),
        (
            slopes,
            "ventricles_volume_ml",
            "REF",
            1,
            "r",
            0.356,
            0.158,
            0.526,
            0.001,
            0.004,
        ),
        (
            slopes,
            "ventricles_volume_ml",
            "iNPH",
            1,
            "r",
            0.453,
            0.269,
            0.605,
            0.0001,
            0.002,
        ),
        (
            slopes,
            "ventricles_volume_ml",
            "REF",
            2,
            "r",
            -0.297,
            -0.477,
            -0.094,
            0.0049,
            0.013,
        ),
        (slopes, "ventricles_volume_ml", "iNPH", 2, "r", 0.01, -0.2, 0.22, 0.9, 0.9),
    ]
    return pd.DataFrame(rows, columns=columns)


def test_associations_list_each_variables_components_together() -> None:
    hits = associations(hand_made_tests(), 0.05)
    assert list(zip(hits["variable"], hits["group"], hits["component"])) == [
        ("Group", "", 1),
        ("ventricles_volume_ml", "REF", 1),
        ("ventricles_volume_ml", "REF", 2),
        ("ventricles_volume_ml", "iNPH", 1),
    ]


def test_the_associations_printed() -> None:
    assert association_lines(hand_made_tests(), 0.05, "cp_rank_4") == [
        "cp_rank_4: 4 of 6 tests with p_perm < 0.05",
        "  Primary variables     Group                        c1  d =  2.09 [ 1.67,  2.51]"
        "  p_perm = 0.0010  p_fdr = 0.0120",
        "  Volume (mL) by Group  ventricles_volume_ml (REF)   c1  r =  0.36 [ 0.16,  0.53]"
        "  p_perm = 0.0010  p_fdr = 0.0040",
        "  Volume (mL) by Group  ventricles_volume_ml (REF)   c2  r = -0.30 [-0.48, -0.09]"
        "  p_perm = 0.0049  p_fdr = 0.0130",
        "  Volume (mL) by Group  ventricles_volume_ml (iNPH)  c1  r =  0.45 [ 0.27,  0.60]"
        "  p_perm = 0.0001  p_fdr = 0.0020",
    ]


def test_nothing_below_the_threshold_prints_only_the_count() -> None:
    assert association_lines(hand_made_tests(), 1e-5, "cp_rank_4") == [
        "cp_rank_4: 0 of 6 tests with p_perm < 1e-05",
    ]
