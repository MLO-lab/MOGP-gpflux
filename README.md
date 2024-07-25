# MOMO-GP

A Multi-Omic Multi-output Gaussian Processes for Integration of Multi Omics Data:
- which learns the nonlinear structure of data by combining the neural network layer with the Gaussian Process layer,
- in the single-view version, it learns separate latent representations for both cells and genes, and
- in the multi-view version, it learns a shared representation of cells and separate representations of features for each view in an interpretable manner.

## Basic usage

The `Running_MOGP.ipynb` file is the main entry point for loading the data and performing the inference.
This file is located in `./experiments/CITEseq/RNA` folder.
In this file, you can see the cell and gene embedding of the MOGP on RNA-seq data of sampled CITE-seq dataset. Running MOGP on the sampled data takes about 1 hour for 200 iterations. 

Then to see the Gene Relevance Map results, you have to run `Running_GeneRelevanceMAP.ipynb` script, located in `./experiments/CITEseq/RNA` folder.

## Installation

We suggest using [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your environments. Follow these steps to get `MOGP` up and running!

1. Create a python environment in `conda`:

```bash
conda env create -f environment_MOGP.yml
```

2. Activate freshly created environment:

```bash
source activate MOGP-GPFLUX
```

3. Create a python environment in `conda`:

```bash
conda env create -f seaCell.yml
```

4. Activate freshly created environment:

```bash
source activate seaCell
```

## Citation

This paper is under review

## Results on the paper

All figures presented in this paper are available in the `./experiments` folder for both the PBMC and CITE-seq datasets.

-	PBMC 10k Dataset
We utilized the PBMC 10k dataset from 10x Genomics, which includes paired single-cell multiome ATAC and gene expression sequencing. This dataset comprises:

11,909 cells
36,601 genes
134,726 peaks. 
The dataset can be accessed from the following link: [PBMC 10k](https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k).

-	5k PBMC CITE-seq Dataset
We also utilized the 5k PBMC CITE-seq dataset, which provides transcriptome-wide measurements for single cells, including gene expression data and surface protein levels for several dozen proteins. This dataset includes:

5,247 cells
33,538 genes
32 proteins. 
The dataset is available from [CITE-seq](https://support.10xgenomics.com/single-cell-gene-expression/datasets/3.0.2/5k_pbmc_protein_v3).

Details about our preprocessing of these datasets can be found in the `./data` folder.

