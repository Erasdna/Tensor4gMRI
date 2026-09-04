from gMRItensor import replicability, decomposition
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import h5py
import sys
import torch
import os
from gMRItensor.preprocessing import prepare_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse tracer data and identify files")
    parser.add_argument("tracer", type=str, help="Path to tracer data")
    parser.add_argument(
        "group",
        type=str,
        help="Sub-group to consider",
        choices=["M", "F", "REF", "iNPH", "all"],
    )
    parser.add_argument(
        "rank",
        type=int,
        help="Number of components to compute",
    )
    parser.add_argument(
        "splitting_method",
        type=str,
        help="Approach for computing replicability",
        choices=["half-half", "CV"],
    )
    parser.add_argument("save", type=str, help="Path to save file folder")

    # Prevent oversubscribing
    torch.set_num_threads(1)

    args = parser.parse_args()

    # Load csv data
    tracer_data = pd.read_csv(args.tracer)
    tracer_data = tracer_data.query(
        "Group in ['REF','iNPH']",
    )

    data, subjects, time_points, regions = prepare_tensor(
        tracer_data,
        ("Sex" if args.group in ["M", "F"] else "Group", args.group),
    )
    stratification_column = "Group" if args.group in ["M", "F"] else "Sex"
    subject_to_group = (
        tracer_data.groupby("subject")[stratification_column].first().to_dict()
    )
    stratification = [subject_to_group[s] for s in subjects]

    # Scale by standard deviation only
    std_scaling = np.nanstd(data, axis=(0, 1), keepdims=False)
    data = data / std_scaling

    # Setup backend and move tensor to pytorch
    device = decomposition.setup_backend()
    tensor = torch.from_numpy(data).to(device)

    CP_init_repeats = 50
    CP_max_iter = 5000
    CP_verbose_level = 0
    CP_tolerance = 1e-7

    if args.splitting_method == "half-half":
        engine = replicability.HalfHalfEngine(
            repeats=10,
            device=device,
        )
    elif args.splitting_method == "CV":
        engine = replicability.CrossValidationEngine(
            splits=10,
            repeats=10,
            device=device,
        )
    else:
        print("Invalid dataset splitting strategy... exiting")
        exit(1)

    # Create file
    sv = Path(args.save)
    sv.mkdir(exist_ok=True, parents=True)
    save_file = sv.joinpath(
        f"CP_{args.group}_{args.rank}_{args.splitting_method}",
    ).with_suffix(".h5")

    n_procs = (
        1
        if os.environ.get("MULTIPROCESSING_TASKS") is None
        else int(str(os.environ.get("MULTIPROCESSING_TASKS")))
    )

    fms = replicability.evaluate_replicability_multiproc(
        engine,
        tensor,
        args.rank,
        stratification,
        n_procs=n_procs,
        CP_init_repeats=CP_init_repeats,
        CP_max_iter=CP_max_iter,
        CP_verbose_level=CP_verbose_level,
        CP_tolerance=CP_tolerance,
        progress_bar=False,
    )
    for n, it in enumerate(fms):
        print(f"FMS: {it[-1]}")
        with h5py.File(save_file, "a" if n > 0 else "w") as f:
            f.create_dataset(f"{n}", data=it[-1])
        sys.stdout.flush()
