"""The LaTeX table of a CP model's components (04_plot_decomp.sh): each component's weight
lambda_r and its share of the data's sum of squares, then the shares' sum and the model fit.
Everything is read from the metrics gMRI's `save_decomposition` writes into model.h5
(`component_weights`, `component_shares`, `relative_squared_error`); only the sum and the fit
(1 - relative squared error) are computed. The table is a booktabs `tabular` without caption
or label, to `\\input` inside a `table` of the document; the names column is left blank."""

from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

HEADER = (
    r"Component number & Component short name & $\lambda_r$ & $\hat{\lambda}_r^2$ \\"
)
SUM_LABEL = r"Sum over components: $\sum_{r=1}^R \hat{\lambda}_r^2$"
FIT_LABEL = (
    r"Model fit: $1 - \|\mathcal{X} - \hat{\mathcal{X}}\|_F^2 / \|\mathcal{X}\|_F^2$"
)


class ComponentMetrics(NamedTuple):
    """A model's `weights` (lambda_r) and `shares` (lambda_r^2 / ||X||^2), component 1 first,
    and its `relative_squared_error` (||X - X_hat||^2 / ||X||^2)."""

    weights: np.ndarray
    shares: np.ndarray
    relative_squared_error: float


def read_component_metrics(path: Path) -> ComponentMetrics:
    """The metrics stored in the model file `path`. Raises ValueError where they are missing
    (a model saved without the data's norm) or the components are not stored largest first
    (the numbering of the figures, `load_decomposition`'s order)."""
    with h5py.File(path, "r") as file:
        missing = [
            name
            for name in ("component_weights", "component_shares")
            if name not in file
        ]
        if missing or "relative_squared_error" not in file.attrs:
            raise ValueError(
                f"{path} has no {', '.join(missing) or 'relative_squared_error'}: gMRI "
                "stores them only with the data's norm; refit with 02_decompose.sh",
            )
        weights = np.asarray(file["component_weights"][()], dtype=float)
        shares = np.asarray(file["component_shares"][()], dtype=float)
        error = float(file.attrs["relative_squared_error"])
    if np.any(np.diff(weights) > 0):
        raise ValueError(
            f"{path}: components are not stored largest first ({weights}); refit with "
            "02_decompose.sh",
        )
    return ComponentMetrics(weights, shares, error)


def component_table(metrics: ComponentMetrics) -> str:
    """The booktabs `tabular` of `metrics`: one row per component (number, blank name, lambda_r
    to one decimal, its share to two), then the sum of the unrounded shares and the fit.
    """
    rows = [
        rf"{k} &  & {weight:.1f} & {share:.2f} \\"
        for k, (weight, share) in enumerate(
            zip(metrics.weights, metrics.shares),
            start=1,
        )
    ]
    total = float(np.sum(metrics.shares))
    fit = 1 - metrics.relative_squared_error
    return "\n".join(
        [
            r"\begin{tabular}{clcc}",
            r"\toprule",
            HEADER,
            r"\midrule",
            *rows,
            r"\midrule",
            rf"\multicolumn{{3}}{{l}}{{{SUM_LABEL}}} & {total:.2f} \\",
            rf"\multicolumn{{3}}{{l}}{{{FIT_LABEL}}} & {fit:.2f} \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ],
    )
