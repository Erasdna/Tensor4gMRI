"""The LaTeX table of a CP model's components (tensor4gmri.component_table and its CLI
code/analysis/component_table.py), read from the metrics gMRI writes into model.h5."""

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from gMRItensor.model_io import (
    SavedDecomposition,
    component_weights,
    data_norm,
    load_decomposition,
    save_decomposition,
    sort_components,
)
from tensor4gmri.component_table import component_table, read_component_metrics

REPO = Path(__file__).resolve().parents[1]


def save_model(path: Path, rank: int = 3, with_norm: bool = True) -> Path:
    """A CP model of a toy tensor, saved as the pipeline saves it (sorted, with ||X||)."""
    rng = np.random.default_rng(0)
    subjects, times, labels = (
        rng.uniform(0.1, 1, (n, rank)) * rng.uniform(1, 4, rank) for n in (8, 4, 12)
    )
    data = np.einsum("ir,jr,kr->ijk", subjects, times, labels)
    saved = SavedDecomposition(
        method="cp", rank=rank, error=0.2, weights=np.ones(rank), subject_mode=subjects,
        time_mode=times, label_mode=labels, subjects=np.array([f"S{i}" for i in range(8)]),
        timepoints=np.array([2, 4, 8, 24]), labels=np.arange(12), label_index=np.arange(12),
        data_norm=data_norm(list(data)) if with_norm else None,
    )  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    save_decomposition(path, sort_components(saved))
    return path


def overwrite(
    path: Path,
    weights: list[float],
    shares: list[float],
    rse: float,
) -> None:
    """Replace the stored metrics, so that a table showing them was read, not recomputed."""
    with h5py.File(path, "r+") as file:
        del file["component_weights"], file["component_shares"]
        file.create_dataset("component_weights", data=np.array(weights))
        file.create_dataset("component_shares", data=np.array(shares))
        file.attrs["relative_squared_error"] = rse


TEMPLATE = r"""\begin{tabular}{clcc}
\toprule
Component number & Component short name & $\lambda_r$ & $\hat{\lambda}_r^2$ \\
\midrule
1 &  & 107.1 & 0.24 \\
2 &  & 90.0 & 0.18 \\
3 &  & 85.4 & 0.16 \\
\midrule
\multicolumn{3}{l}{Sum over components: $\sum_{r=1}^R \hat{\lambda}_r^2$} & 0.58 \\
\multicolumn{3}{l}{Model fit: $1 - \|\mathcal{X} - \hat{\mathcal{X}}\|_F^2 / \|\mathcal{X}\|_F^2$} & 0.94 \\
\bottomrule
\end{tabular}
"""  # noqa: E501 (the template's fit line)


def test_the_table_shows_the_stored_metrics(tmp_path: Path) -> None:
    model = save_model(tmp_path / "cp_rank_3/model.h5")
    overwrite(model, [107.12, 90.01, 85.43], [0.24, 0.18, 0.16], 0.06)
    assert component_table(read_component_metrics(model)) == TEMPLATE


def test_the_sum_is_of_the_unrounded_shares(tmp_path: Path) -> None:
    model = save_model(tmp_path / "cp_rank_3/model.h5")
    overwrite(model, [3.0, 2.0, 1.0], [0.244, 0.244, 0.244], 0.1)
    table = component_table(read_component_metrics(model))
    assert table.count(" & 0.24 \\\\") == 3
    assert r"\hat{\lambda}_r^2$} & 0.73 \\" in table


def test_components_are_numbered_as_in_the_figures(tmp_path: Path) -> None:
    """The file's order is load_decomposition's (largest first), which the figures use."""
    model = save_model(tmp_path / "cp_rank_3/model.h5")
    metrics = read_component_metrics(model)
    loaded = component_weights(load_decomposition(model))
    np.testing.assert_allclose(metrics.weights, loaded)
    with h5py.File(model, "r") as file:
        np.testing.assert_allclose(metrics.shares, file["component_shares"][()])
        assert metrics.relative_squared_error == file.attrs["relative_squared_error"]


def test_a_model_without_the_data_norm_is_refused(tmp_path: Path) -> None:
    model = save_model(tmp_path / "cp_rank_3/model.h5", with_norm=False)
    with pytest.raises(ValueError, match="component_shares"):
        read_component_metrics(model)


def test_unsorted_components_are_refused(tmp_path: Path) -> None:
    model = save_model(tmp_path / "cp_rank_3/model.h5")
    overwrite(model, [2.0, 3.0, 1.0], [0.2, 0.3, 0.1], 0.1)
    with pytest.raises(ValueError, match="largest first"):
        read_component_metrics(model)


def test_the_script_writes_the_table_of_a_model_directory(tmp_path: Path) -> None:
    model = save_model(tmp_path / "cp_rank_3/model.h5")
    output = tmp_path / "figures/decomposition/cp_rank_3__components.tex"
    result = subprocess.run(
        [
            sys.executable, str(REPO / "code/analysis/component_table.py"),
            "--model", str(model.parent), "--output", str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(REPO / "code")},
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert output.read_text() == component_table(read_component_metrics(model))
    assert str(output) in result.stdout
