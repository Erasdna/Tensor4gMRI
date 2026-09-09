# Tensor4gMRI
Code for the paper "Low-rank tensor decompositions reveal coupled spatiotemporal patterns of CSF tracer transport in the human brain" by Solheim et al. 2026

**Abstract**
> Intrathecal contrast-enhanced longitudinal MRI, or glymphatic MRI (gMRI), provides a unique clinical window into human cerebrospinal fluid (CSF) transport, interstitial fluid (ISF) interaction, and the glymphatic system. However, standard analyses flatten these multi-subject longitudinal data into univariate comparisons across regions of interest, destroying the underlying multi-way structure and obscuring coupled spatiotemporal dynamics. Here we analyze gMRI data from 92 patients (43 diagnosed with idiopathic normal pressure hydrocephalus [iNPH] and 49 reference subjects [REF]) using unsupervised non-negative low-rank tensor decompositions that treat tracer signal in CSF and brain parenchyma simultaneously. The analysis is stratified by sex to avoid confounding diagnosis with the sex imbalance between cohorts. We recover four replicable components, three of which resolve distinct transport pathways capturing CSF--ISF interaction: distribution of tracer in the supratentorial subarachnoid space and cerebral gray and white matter at 24 hours; transient early influx in regions consistent with perivascular transport along major cerebral arteries; and tracer influx to ventricular CSF, also called ventricular reflux. The first two components are positively correlated and thus establish a direct quantitative link between early-stage influx and 24-hour tracer distribution. Moreover, the expression of the two influx pathways varies significantly between REF and iNPH cohorts, demonstrating how unsupervised tensor decompositions provide an automated, data-driven framework to stratify patient groups and derive quantitative markers of solute transport in the human brain.

## Running the code in this project

The data used in this paper is not made available due to data privacy concerns. Accordingly, we are not able to provide reproducible examples.

If you want to use this code on your own project, we recommend using `uv` to install the dependencies by running `uv sync`. We recommend running code using `uv run`.

We also provide an apptainer `.def` file which can be built using `apptainer build apptainer/app.sif apptainer/app.def`

## What is in this repo

This repo contains scripts and code to transform gMRI data containing contrast enhanced imaging from multiple subjects at multiple time points into tensors and then decompose them. This code is mainly built around the [gMRI-tensor](https://github.com/Erasdna/gMRI-tensor) project which wraps functionality for this purpose.

The `code` directory contains the key elements of this project, including:
- `code/data_preparation`: Scripts to identify relevant files, read them and transform them into tensors
- `code/decomposition`: Scripts and code to compute the CP decomposition of the tensor
- `code/replicability`: Scripts and code to perform a replicability analysis of the CP model
- `code/snakemake`: Scripts and code to reproduce all the figures in the paper
