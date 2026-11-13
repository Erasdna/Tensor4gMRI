"""Find each subject's scans closest to TIME_POINTS hours after the first one, and merge its
SAS and parenchyma segmentations.

Writes, under <results_dir>:
- `<subject>/<subject>-segmentation.nii.gz` (SAS labels + 10000, else the parenchyma labels),
  `<subject>-brainmask.nii.gz` (labelled voxels) and `<subject>-T1w.nii.gz`, a copy of the
  pre-injection (baseline) scan, the background of the spatial maps
- `files_to_process.csv`: one row per subject and time point, with the baseline (first) and
  post-injection scan, mask and segmentation, the actual time in hours and the old id

Subjects marked `exclude` in the labels are left out; subjects without a scan for every time
point, or with two time points on one scan, are skipped with a note.
"""

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tensor4gmri.config import TIME_POINTS
from tensor4gmri.subject_info import included_labels, read_labels


def compute_time(filename: Path, n: int) -> float:
    t_raw = int(filename.stem[-10:-4])
    # Compute time in hours
    return (
        (t_raw % 100)
        + (t_raw // 100) % 100 * 60
        + (t_raw // 10000) % 100 * 3600
        + n * 24 * 3600
    ) / 3600


def find_images(folder: Path) -> tuple[str, dict[str, list], Path, Path]:
    n = 0
    img_list: list[Path] = []
    # Find images from first day
    while len(img_list) == 0 and n < 10:
        n = n + 1
        img_list = list(folder.glob("*normalized-201?010" + str(n) + "*"))

    if len(img_list) == 0:
        raise FileNotFoundError

    times = []
    for img in img_list:
        times.append(compute_time(img, 0))
    # Add second day
    second_day = list(folder.glob("*normalized-201?010" + str(n + 1) + "*"))
    if len(second_day) < 1:
        raise FileNotFoundError
    img_list.append(second_day[0])
    times.append(compute_time(img_list[-1], 1))
    time = np.array(times)
    images = np.array(img_list)

    sorted_ids = np.argsort(time)
    images = images[sorted_ids]
    time = time[sorted_ids]

    reference = str(images[0].resolve().absolute())
    time = time - time[0]

    tracer: dict[str, list] = {"actual_time": [], "path": [], "time_point": []}
    prev = None
    # Find image closest to each time point
    for time_point in TIME_POINTS:
        img_id = np.argmin(np.abs(time - time_point))
        tracer["path"].append(str(images[img_id].resolve().absolute()))
        tracer["actual_time"].append(time[img_id])
        tracer["time_point"].append(time_point)

        if prev is not None:
            if prev == str(images[img_id]):
                raise FileNotFoundError

        print(f"Image after {time_point} hours")
        print("     Actual time ", time[img_id], " filename ", images[img_id].name)

        prev = str(images[img_id])

    subj = folder.name
    sas = folder.joinpath(subj + "-sasparc.nii.gz").resolve().absolute()
    parenchyma = folder.joinpath(subj + "-refroi.nii.gz").resolve().absolute()
    for f in [sas, parenchyma]:
        if not f.exists():
            raise FileNotFoundError

    return reference, tracer, sas, parenchyma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", type=Path, help="Raw data: one folder per subject")
    parser.add_argument(
        "labels",
        type=Path,
        help="Subject labels csv ($SUBJECT_LABELS)",
    )
    parser.add_argument("save", type=Path, help="Results directory")
    args = parser.parse_args()

    labels = included_labels(read_labels(args.labels))

    save_df: dict[str, list] = {
        "subject": [],
        "time_point": [],
        "baseline_path": [],
        "post_injection_path": [],
        "mask_path": [],
        "segmentation_path": [],
        "actual_time": [],
        "old_id": [],
    }
    for subj, old_id in zip(labels["new_id"], labels["old_id"]):
        subj_path = args.folder.joinpath(subj)
        print(subj)
        try:
            reference, tracer, sas, parenchyma = find_images(subj_path)
        except FileNotFoundError:
            print("Missing files...skipping")
            continue

        # Merge
        parenchyma_nifti = nib.load(parenchyma)
        parenchyma_data = parenchyma_nifti.get_fdata()
        sas_data = nib.load(sas).get_fdata()
        seg = np.where(sas_data > 0, sas_data + 10000, parenchyma_data)
        out_dir = args.save.joinpath(subj)
        out_dir.mkdir(parents=True, exist_ok=True)
        segmentation_path = out_dir.joinpath(f"{subj}-segmentation.nii.gz").absolute()
        brainmask_path = out_dir.joinpath(f"{subj}-brainmask.nii.gz").absolute()
        # Integer types, rounded, so the labels are stored exactly (no scaling to the source's
        # type, and a label read back from a float or scaled image as 1.9999 stays 2)
        for data, dtype, path in (
            (seg, np.int32, segmentation_path),
            (seg > 0, np.uint8, brainmask_path),
        ):
            image = nib.Nifti1Image(
                np.rint(data).astype(dtype),
                parenchyma_nifti.affine,
                parenchyma_nifti.header,
            )
            image.set_data_dtype(dtype)
            nib.save(image, path)
        shutil.copyfile(reference, out_dir.joinpath(f"{subj}-T1w.nii.gz"))

        for time, path, actual_time in zip(
            tracer["time_point"],
            tracer["path"],
            tracer["actual_time"],
        ):
            save_df["subject"].append(subj)
            save_df["time_point"].append(time)
            save_df["baseline_path"].append(reference)
            save_df["post_injection_path"].append(str(path))
            save_df["mask_path"].append(str(brainmask_path))
            save_df["segmentation_path"].append(str(segmentation_path))
            save_df["actual_time"].append(actual_time)
            save_df["old_id"].append(old_id)

    files = pd.DataFrame(save_df)
    files.to_csv(args.save.joinpath("files_to_process.csv"), index=False)
    print(f"{files['subject'].nunique()} of {len(labels)} subjects found")
