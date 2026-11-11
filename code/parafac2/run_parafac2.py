from gMRItensor import decomposition
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import torch
import h5py
from gMRItensor.preprocessing import prepare_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse tracer data and identify files")
    parser.add_argument("tracer", type=Path, help="Path to tracer data")
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
    parser.add_argument("save", type=Path, help="Path to save file folder")

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
    # Scale by standard deviation only
    std_scaling = np.nanstd(data, axis=(0, 1), keepdims=False)
    data = data / std_scaling

    # Setup backend and move tensor to pytorch
    device = decomposition.setup_backend()
    list_data = [torch.from_numpy(d).to(device) for d in data]  # List of subject slices

    for d in list_data:
        print(d.shape)
    print(data.shape)

    init_repeats = 50
    max_iter = 5000
    verbose_level = 0
    tolerance = 1e-7

    (
        weights,
        factors,
        projections,
        err,
    ) = decomposition.run_PARAFAC2_decomposition_repeated(
        list_data,
        args.rank,
        max_iter=max_iter,
        init_repeats=init_repeats,
        device=device,
        verbose_level=verbose_level,
        tolerance=tolerance,
        nn_modes=[-1],
    )
    print(weights)

    order = np.argsort(weights, descending=True)
    print(order)

    save_path = args.save / f"PARAFAC2_{args.rank}_{args.group}.h5"
    args.save.mkdir(exist_ok=True, parents=True)
    string_dt = h5py.string_dtype(encoding="utf-8", length=None)
    with h5py.File(save_path, "w") as f:
        factor_group = f.create_group("factors")
        factor_group.create_dataset("subject_mode", data=factors[0][:, order])
        factor_group.create_dataset("time_mode", data=factors[1][:, order])
        factor_group.create_dataset("roi_mode", data=factors[2][:, order])
        factor_group.create_dataset("projections", data=projections)
        factor_group.create_dataset("error", data=err)
        factor_group.create_dataset("scaling", data=std_scaling)
        factor_group.create_dataset("weights", data=weights[order])

        f.create_dataset("subjects", data=subjects.astype(object), dtype=string_dt)
        f.create_dataset("time_points", data=time_points)
        f.create_dataset("rois", data=regions)
