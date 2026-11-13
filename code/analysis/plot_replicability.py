"""Factor match score between split halves against rank, from replicability.csv."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Prevent GUI display errors in containers/terminals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib.colors import to_rgb

plt.style.use(["science", "no-latex"])

RANK_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
REFERENCE_FMS = 0.9


def lighten(color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """`color` blended toward white by `amount` (0: unchanged, 1: white)."""
    rgb = np.asarray(to_rgb(color))
    r, g, b = rgb + (1 - rgb) * amount
    return (float(r), float(g), float(b))


def fms_summary(fms: pd.DataFrame) -> pd.DataFrame:
    """Median, quartiles and number of finite splits of the FMS per rank."""
    grouped = fms.dropna(subset=["fms"]).groupby("rank")["fms"]
    return pd.DataFrame(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
            "n_splits": grouped.size(),
            "n_failed": fms["fms"].isna().groupby(fms["rank"]).sum(),
        },
    ).reset_index()


def plot_fms(fms: pd.DataFrame) -> plt.Figure:
    finite = fms.dropna(subset=["fms"])
    ranks = sorted(fms["rank"].unique())
    colors = {r: RANK_COLORS[i % len(RANK_COLORS)] for i, r in enumerate(ranks)}
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.axhline(REFERENCE_FMS, color="black", linestyle="--", linewidth=1, zorder=1)
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(finite))
    ax.scatter(
        finite["rank"] + jitter,
        finite["fms"],
        s=14,
        alpha=0.7,
        color=finite["rank"].map(colors).tolist(),
        linewidths=0,
        zorder=2,
    )
    for r in ranks:
        color = colors[r]
        line = {"color": color, "linewidth": 1.5}
        ax.boxplot(
            [finite.loc[finite["rank"] == r, "fms"]],
            positions=[r],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": (*lighten(color), 0.5),
                "edgecolor": color,
                "linewidth": 1.5,
            },
            whiskerprops=line,
            capprops=line,
            medianprops={"color": color, "linewidth": 2},
            zorder=3,
        )
    ax.set_xticks(ranks)
    ax.set_xlabel("Rank")
    ax.set_ylabel("FMS score")
    ax.set_ylim(top=1.02)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replicability_csv",
        type=Path,
        help="gmri replicability output",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Where to write the plot and summary",
    )
    args = parser.parse_args()

    fms = pd.read_csv(args.replicability_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = fms_summary(fms)
    summary.to_csv(args.output_dir / "fms_summary.csv", index=False)
    print(summary.to_string(index=False))
    fig = plot_fms(fms)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"fms_by_rank.{ext}", dpi=300)
    plt.close(fig)
