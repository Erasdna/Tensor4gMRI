from snakemake.script import snakemake
import pandas as pd
import numpy as np

# Extract only subjects after cleaning of the dataset
data_subjects = pd.read_csv(
    snakemake.input.DATA,
).query("Group in ['REF','iNPH']")["subject"].unique()
subject_info = pd.read_csv(snakemake.input.INFO).query("new_id in @data_subjects")

print(subject_info)


def build_table_for_column(column, column_naming, category, category_naming):
    # Compute distribution of age, height and weight
    measurements = (
        subject_info[[column, "Age", "Height", "Weight"]]
        .groupby(column)
        .agg(
            [
                ("median", lambda x: np.median(x)),
                ("pc25", lambda x: np.percentile(x, 25)),
                ("pc75", lambda x: np.percentile(x, 75)),
            ],
        )
    )
    print(measurements)

    counts = subject_info[[column, category]].groupby(category).agg("count")

    print(counts)
    latex_table = {"Group": [], "Measurement": [], "String": []}

    for measurement in measurements.columns.get_level_values(0).unique():
        for group in measurements[measurement].index:
            latex_table["Group"].append(column_naming[group])
            latex_table["Measurement"].append(measurement)
            data = measurements[measurement].loc[group]
            latex_table["String"].append(
                rf"${int(data["median"])}~[{int(data["pc25"])},{int(data["pc75"])}]$",
            )

    for m in ["Age", "Height", "Weight"]:
        latex_table["Group"].append("Total")
        latex_table["Measurement"].append(m)
        data = subject_info[m]
        latex_table["String"].append(
            rf"${int(np.median(data))}~[{int(np.percentile(data,25))},{int(np.percentile(data,75))}]$",
        )
    latex_table = pd.DataFrame(latex_table).pivot(
        index="Group", values="String", columns="Measurement",
    )

    print(latex_table)

    category_counts = {"Group": [], "Count": []}
    for group_id, group_name in column_naming.items():
        total = len(subject_info.query(f"{column} == {group_id}"))
        M_count = len(subject_info.query(f"{column} == {group_id} & {category} == 0"))
        # Sanity check:
        F_count = len(subject_info.query(f"{column} == {group_id} & {category} == 1"))
        assert total == M_count + F_count
        category_counts["Count"].append(
            rf"{total} ({M_count}~{category_naming[0]}/ {F_count}~{category_naming[1]})",
        )
        category_counts["Group"].append(group_name)

    category_counts["Group"].append("Total")
    category_counts["Count"].append(
        rf"{len(subject_info)} ({len(subject_info[subject_info[category]==0])} {category_naming[0]}/{len(subject_info[subject_info[category]==1])} {category_naming[1]})", # noqa: E501
    )

    latex_table = pd.merge(
        pd.DataFrame(category_counts), latex_table, on="Group",
    ).reset_index()
    return latex_table


group_naming = {0: "REF", 1: "iNPH"}
sex_naming = {0: "Male", 1: "Female"}

diagnosis_table = build_table_for_column("Group", group_naming, "Sex", sex_naming)
sex_table = build_table_for_column("Sex", sex_naming, "Group", group_naming)
print(sex_table)
latex_table = pd.concat([diagnosis_table, sex_table])
print(latex_table)
latex_table.to_latex(
    snakemake.output.TABLE,
    index=False,
    columns=latex_table.columns[1:].values,
    header=["Group", "Subjects", "Age", "Height [cm]", "Weight [kg]"],
    bold_rows=[False,False,True] * 2,
)
