"""The subject table of the pipeline (subject_info.csv) and the subjects of each group."""

from collections.abc import Collection
from pathlib import Path

import pandas as pd

from tensor4gmri.config import GROUP_FILTER, GROUP_LABELS, GROUPS, SEX_LABELS


def read_labels(path: Path) -> pd.DataFrame:
    """$SUBJECT_LABELS, with the subject ids as text. Raises `ValueError` if a `new_id` is
    repeated."""
    labels = pd.read_csv(path, dtype={"new_id": str, "old_id": str})
    repeated = sorted(set(labels.loc[labels["new_id"].duplicated(), "new_id"]))
    if repeated:
        raise ValueError(f"Subjects more than once in {path}: {repeated}")
    return labels


def included_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """The rows of `labels` not marked `exclude`."""
    return labels[labels["exclude"] == 0]


def subject_info_from_labels(
    labels: pd.DataFrame,
    subjects: Collection[str],
) -> pd.DataFrame:
    """The included rows of `labels` of `subjects`, keyed by `subjects` as gmri expects, with
    Group and Sex named (REF/iNPH, M/F) and Stratum their combination, e.g. REF_M.

    Raises `ValueError` for a subject of `subjects` that is excluded or not in `labels`, and for
    a Group or Sex code without a name.
    """
    included = included_labels(labels)
    missing = sorted(set(subjects) - set(included["new_id"]))
    if missing:
        raise ValueError(f"Subjects excluded or missing in the labels: {missing}")
    info = included[included["new_id"].isin(subjects)].rename(
        columns={"new_id": "subjects"},
    )
    for column, names in (("Group", GROUP_LABELS), ("Sex", SEX_LABELS)):
        named = info[column].map(names)
        if named.isna().any():
            codes = sorted(set(info.loc[named.isna(), column]))
            raise ValueError(f"{column} codes without a name: {codes} (known: {names})")
        info[column] = named
    info["Stratum"] = info["Group"] + "_" + info["Sex"]
    return info.reset_index(drop=True)


def read_subject_info(path: Path) -> pd.DataFrame:
    """subject_info.csv with every cell as text."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def group_members(info: pd.DataFrame, group: str) -> list[str]:
    """The subjects of `info` in `group` (config.GROUP_FILTER), in table order."""
    if group not in GROUPS:
        raise ValueError(
            f"Unknown group '{group}' (expected one of {', '.join(GROUPS)})",
        )
    selection = GROUP_FILTER[group]  # type: ignore[index]
    if selection is None:
        return info["subjects"].tolist()
    column, value = selection
    return info.loc[info[column] == value, "subjects"].tolist()


def excluded_for_group(
    info: pd.DataFrame,
    group: str,
    present: Collection[str],
) -> list[str]:
    """The subjects of `present` (those in the data) not in `group`, sorted: what a fit of
    `group` leaves out with `--exclude-subjects`.

    Raises `ValueError` for an unknown group or a subject of `present` not in `info`.
    """
    unknown = sorted(set(present) - set(info["subjects"]))
    if unknown:
        raise ValueError(f"Subjects of the data not in the subject info: {unknown}")
    members = set(group_members(info, group))
    return sorted(s for s in present if s not in members)
