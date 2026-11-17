"""Correlations between the components of a model: each pair of components' subject coefficients,
adjusted for the primary variables with `gmri stats`' own test (`clinical_effects`), and
unadjusted."""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from gMRItensor.subject_statistics import add_fdr, clinical_effects

PAIR_COLUMNS = [
    "component_a",
    "component_b",
    "n",
    "covariates",
    "partial_r",
    "partial_r_ci_low",
    "partial_r_ci_high",
    "p_perm",
    "p_fdr",
    "significant",
    "r_unadjusted",
    "r_unadjusted_ci_low",
    "r_unadjusted_ci_high",
    "p_unadjusted",
    "note",
]


def coefficient_columns(frame: pd.DataFrame) -> list[str]:
    """The coefficient columns c1, c2, ... of `frame` (`plot_stats.data.subject_frame`)."""
    return [c for c in frame.columns if re.fullmatch(r"c\d+", str(c))]


def component_correlations(
    frame: pd.DataFrame,
    primary_categorical: Sequence[str],
    primary_numerical: Sequence[str],
    n_permutations: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per pair of components a < b of `frame` (`PAIR_COLUMNS`): the partial correlation
    of their coefficients adjusted for the primary variables, with its 95% CI and Freedman-Lane
    permutation p (component a's coefficients regressed on component b's and the primary
    variables, as `gmri stats` tests a clinical variable), `p_fdr` (Benjamini-Hochberg over all
    pairs), and the unadjusted Pearson r with its CI and p. The partial r is the same either
    way round. Empty (with the columns) for fewer than two components."""
    components = coefficient_columns(frame)
    coefficients = frame[components].to_numpy(float)
    primary = [*primary_categorical, *primary_numerical]
    rows = []
    for a in range(len(components)):
        for b in range(a + 1, len(components)):
            term = f"Component {b + 1}"
            table = clinical_effects(
                coefficients[:, [a]],
                frame.assign(**{term: coefficients[:, b]}),
                [term],
                primary,
                primary_categorical,
                n_permutations=n_permutations,
                seed=seed,
            )
            rows.append(table.assign(component_a=a + 1, component_b=b + 1))
    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    pairs = add_fdr(
        pd.concat(rows, ignore_index=True),
        alpha=alpha,
        keys=("component_a", "component_b"),
    )
    pairs["n"] = pairs["n"].astype(np.int64)
    return pairs.loc[:, PAIR_COLUMNS]
