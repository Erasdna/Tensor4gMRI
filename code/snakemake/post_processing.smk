
configfile: "code/snakemake/post_processing.yaml"

import os

def expand_env_vars(d):
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = os.path.expandvars(v)
        elif isinstance(v, dict):
            expand_env_vars(v)

expand_env_vars(config)

CP_output = config["CP_output"]
savepath = config["savepath"]
subject_info = config["subject_info"]
example_subject = config["example_subject"]
data_path = config["data_path"]

groups = ["M"]
Rs = [3]
#groups = ["F"]
#Rs = [3,4]

rule all:
    input:
        savepath + "subject_info.tex",
        expand(
            savepath + "{group}/R_{R}/{subfolder}/done.png",
            R=Rs,
            group=groups,
            subfolder=["components", "spatial", "subject"],
            allow_missing=True
        ),
        expand(
            savepath + "{group}/R_{R}/component_correlations_done.png",
            R=Rs,
            group=groups,
            allow_missing=True
        ),


rule subject_characteristics:
    input:
        INFO=subject_info,
        DATA=data_path,
    output:
        TABLE=savepath + "subject_info.tex",  # Latex table with subject info
    script:
        "scripts/subject_characteristics.py"


rule plot_components:
    input:
        CP_h5=CP_output + "CP_{R}_{group}.h5",
        INFO=subject_info,
        parenchyma_segmentation=example_subject + "parenchyma_seg.nii.gz",
        csf_segmentation=example_subject + "csf_seg.nii.gz",
        T1w=example_subject + "T1w.nii.gz",
    output:
        savepath + "{group}/R_{R}/components/done.png"
    params:
        Rs="{R}",
        save=savepath + "{group}",
        group="{group}",
    script:
        "scripts/plot_components.py"


rule spatial_mode:
    input:
        CP_h5=CP_output + "CP_{R}_{group}.h5",
        parenchyma_segmentation=example_subject + "parenchyma_seg.nii.gz",
        csf_segmentation=example_subject + "csf_seg.nii.gz",
        T1w=example_subject + "T1w.nii.gz",
    output:
        savepath + "{group}/R_{R}/spatial/done.png"
    params:
        Rs="{R}",
        save=savepath + "{group}",
    script:
        "scripts/plot_spatial.py"


rule subject_mode:
    input:
        CP_h5=CP_output + "CP_{R}_{group}.h5",
        INFO=subject_info,
    output:
        savepath + "{group}/R_{R}/subject/done.png",
    params:
        Rs="{R}",
        save=savepath + "{group}",
        group="{group}",
    script:
        "scripts/plot_subject.py"


# rule sort_subject_mode:
#     input:
#         CP_h5=CP_output + "CP_{R}_{group}.h5",
#         INFO=subject_info,
#     output:
#         savepath + "{group}/R_{R}/subject/sorted.csv",
#     params:
#         Rs="{R}",
#         save=savepath + "{group}",
#         group="{group}",
#     script:
#         "scripts/sort_subjects.py"


rule plot_correlations:
    input:
        CP_h5=CP_output + "CP_{R}_{group}.h5",
        INFO=subject_info,
    output:
        savepath + "{group}/R_{R}/component_correlations_done.png"
    params:
        Rs="{R}",
        save=savepath + "{group}",
        group="{group}",
    script:
        "scripts/plot_correlations.py"
