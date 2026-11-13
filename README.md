# Tensor4gMRI
Code for the paper "Low-rank tensor decompositions reveal coupled spatiotemporal patterns of CSF tracer transport in the human brain" by Solheim et al. 2026

**Abstract**
> Intrathecal contrast-enhanced longitudinal MRI, or glymphatic MRI (gMRI), provides a unique clinical window into human cerebrospinal fluid (CSF) transport, interstitial fluid (ISF) interaction, and the glymphatic system. However, standard analyses flatten these multi-subject longitudinal data into univariate comparisons across regions of interest, destroying the underlying multi-way structure and obscuring coupled spatiotemporal dynamics. Here we analyze gMRI data from 92 patients (43 diagnosed with idiopathic normal pressure hydrocephalus [iNPH] and 49 reference subjects [REF]) using unsupervised non-negative low-rank tensor decompositions that treat tracer signal in CSF and brain parenchyma simultaneously. The analysis is stratified by sex to avoid confounding diagnosis with the sex imbalance between cohorts. We recover four replicable components, three of which resolve distinct transport pathways capturing CSF--ISF interaction: distribution of tracer in the supratentorial subarachnoid space and cerebral gray and white matter at 24 hours; transient early influx in regions consistent with transport along major cerebral arteries; and tracer influx to ventricular CSF, also called ventricular reflux. The first two components are positively correlated and thus establish a direct quantitative link between early-stage influx and 24-hour tracer distribution. Moreover, the expression of the two influx pathways varies significantly between REF and iNPH cohorts, demonstrating how unsupervised tensor decompositions provide an automated, data-driven framework to stratify patient groups and derive quantitative markers of solute transport in the human brain.

## Running the code in this project

The data used in this paper is not made available due to data privacy concerns. Accordingly, we are not able to provide reproducible examples.

If you want to use this code on your own project, we recommend using `uv` to install the dependencies by running `uv sync`. We recommend running code using `uv run`.

The analysis runs inside an Apptainer container, built with `apptainer build apptainer/app.sif apptainer/app.def`. See [`code/pipeline/README.md`](code/pipeline/README.md) for the stages, their options and outputs. In short:

```bash
bash code/pipeline/00_find_files.sh
bash code/pipeline/01_preprocessing.sh
bash code/pipeline/02_decompose.sh "all REF iNPH M F"
bash code/pipeline/03_replicability.sh "all REF iNPH M F"
bash code/pipeline/05_subject_stats.sh "all REF iNPH M F" "3 4"
```

`uv run pytest` runs the tests. `bash code/scripts/update_gmritensor.sh` moves the gMRI-tensor pin in `uv.lock` to the head of its branch, tests and rebuilds the container.

## What is in this repo

This repo contains scripts and code to transform gMRI data containing contrast enhanced imaging from multiple subjects at multiple time points into tensors and then decompose them. This code is mainly built around the [gMRI-tensor](https://github.com/Erasdna/gMRI-tensor) project, whose `gmri` command line does the preprocessing, decompositions, replicability analysis, figures and statistics.

The `code` directory contains:
- `code/pipeline`: the analysis, stage by stage (`00_find_files.sh` ... `05_subject_stats.sh`). Start here.
- `code/tensor4gmri`: shared helpers: time points and subject groups, the image manifest and the subject table
- `code/preprocessing`: finding the images, the subject table and the manifest (stage `00`), and the subjects of each group
- `code/analysis`: the replicability plot (stage `03`)
- `code/plot_stats`: figures and tables of the subject statistics (stage `05`)
- `code/scripts`: the container wrappers used by every stage (`common.sh`) and `update_gmritensor.sh`

Legacy code, written for an earlier version of gMRI-tensor and not yet ported to the pipeline:
- `code/data_preparation`, `code/decomposition`, `code/replicability`: the previous preprocessing, CP decomposition and replicability scripts
- `code/snakemake`, `code/parafac2` and the `code/plot_*.py`, `code/brain_anatomy.py` scripts: the figures of the paper

## Citation

If you found the software in this repo useful and applied it to your own work, please cite it as

```
@software{Solheim2026,
  author = {Solheim, Andreas and Ringstad, Geir and Eide, Per Kristian and Rognes, Marie and Acar, Evrim and Mardal, Kent-Andre},
  doi = {10.5281/zenodo.22669838},
  month = {9},
  title = {{Low-rank tensor decompositions reveal coupled spatiotemporal patterns of CSF tracer transport in the human brain}},
  url = {https://github.com/Erasdna/Tensor4gMRI},
  version = {1.0.0},
  year = {2026}
}
```
