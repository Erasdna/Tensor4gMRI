import pandas as pd
from pathlib import Path
import argparse
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot replicability outcomes")
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder containing relevant files",
    )
    parser.add_argument("tracer", type=Path, help="Path to tracer data")

    args = parser.parse_args()
    tracer_data = pd.read_csv(args.tracer)

    # Use only iNPH and REF for the moment
    labels = tracer_data.query(
        "Group in ['REF','iNPH']",
    )
    counts_dict = {
        "Male": len(labels[labels["Sex"] == "M"]),
        "Female": len(labels[labels["Sex"] == "F"]),
        "REF": len(labels[labels["Group"] == "REF"]),
        "iNPH": len(labels[labels["Group"] == "iNPH"]),
    }
    counts = pd.DataFrame.from_dict(counts_dict, orient="index", columns=["Subjects"])
    print(counts)

    data_dict: dict[str, list[Any]] = {
        "Group": [],
        "rank": [],
        "splitting_method": [],
        "FMS": [],
    }
    # Iterate over .h5 files in folder:
    for file in args.folder.glob("CP_*.h5"):
        try:
            _, group, rank, splitting_method = file.name.split(".")[0].split("_")
        except ValueError:
            print("Invalid file name...skipping")
            continue

        if int(rank) > 6:
            continue
        with h5py.File(file, "r") as f:
            for k, v in f.items():
                data_dict["Group"].append(group)
                data_dict["rank"].append(int(rank))
                data_dict["splitting_method"].append(splitting_method)
                data_dict["FMS"].append(v[()])

    df = pd.DataFrame(data_dict)
    print(df)

    df["Group"] = df["Group"].replace({"M": "Male", "F": "Female"})
    group_order = sorted(df["Group"].unique())

    # Create a fixed color palette dictionary mapping each group to a specific color
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    group_palette = dict(zip(group_order, default_colors[: len(group_order)]))

    fig_save_path = args.folder.joinpath("Figures")
    fig_save_path.mkdir(exist_ok=True, parents=True)
    # Make boxplot by splitting method
    for method in df["splitting_method"].unique():
        plot_df = df.query(f"splitting_method=='{method}'")
        fig, ax = plt.subplots()
        sns.boxplot(
            plot_df,
            x="rank",
            y="FMS",
            hue="Group",
            hue_order=group_order,  # Keeps legend/box order static
            palette=group_palette,  # Keeps group colors static
            gap=0.1,
            showfliers=False,
        )

        print(plot_df.groupby(["rank", "Group"])["FMS"].mean().unstack())

        fill_alpha = 0.6
        for container in ax.containers:
            # Extract base RGBA color from the box face
            r, g, b, _ = container.boxes[0].get_facecolor()

            solid_color = (r, g, b, 1.0)  # 100% opaque for outlines & lines
            fill_color = (r, g, b, fill_alpha)  # Reduced alpha for box fill

            # Apply semi-transparent fill to boxes
            plt.setp(container.boxes, facecolor=fill_color)

            # Apply solid color to outlines, whiskers, caps, and medians
            plt.setp(container.boxes, edgecolor=solid_color)
            plt.setp(container.whiskers, color=solid_color)
            plt.setp(container.caps, color=solid_color)
            plt.setp(container.medians, color=solid_color)

        # Update legend handles to match fill and outline transparency
        if ax.legend_:
            ax.legend_.set_title(None)
            for handle in ax.legend_.legend_handles:
                r, g, b, _ = handle.get_facecolor()
                handle.set_facecolor((r, g, b, fill_alpha))
                handle.set_edgecolor((r, g, b, 1.0))

        ax.set_xlabel("Number of components")
        fig.savefig(fig_save_path.joinpath(f"{method}.png"), dpi=500)
        fig.savefig(fig_save_path.joinpath(f"{method}.pdf"), dpi=500)

        # Make latex table of mean FMS values across groups and ranks
        latex_df = plot_df.groupby(["rank", "Group"])["FMS"].mean().unstack().T

        latex_df.columns = [rf"$R={col}$" for col in latex_df.columns]
        for col in latex_df.columns:
            latex_df[col] = latex_df[col].apply(
                lambda x: (
                    r"$\textbf{" + f"{x:.2f}" + r"}$" if x > 0.9 else rf"${x:.2f}$"
                ),
            )  # Set values > 0.9 to bold

        latex_table_df = latex_df.reset_index()
        latex_table_df = pd.merge(
            counts.reset_index().rename(columns={"index": "Group"}),
            latex_table_df,
            on="Group",
        )

        print(latex_table_df)
        latex_code = latex_table_df.to_latex(
            # float_format="%.2f",
            column_format="l" + "c" * (len(latex_df.columns) + 1),
            escape=False,
            index=False,
        )

        # Optional: Save directly to a .tex file
        with open(fig_save_path.joinpath(f"{method}_mean.tex"), "w") as f:
            f.write(latex_code)
