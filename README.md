This Python-based software tool, SiQ-3D (Single-cell image Quantifier for 3D), optimizes Deep Learning (DL)-based 3D image segmentation, single-cell phenotype classification and tracking to automatically quantify and convert 3D live-cell imaging movies into multi-dimensional dynamic data for different interacting cell types in a 3D tissue/organ microenvironment. The SiQ-3D quantified results are output in an excel file of cell position and phenotype at each time point together with labelled images of the segmented single cells for each cell type in the imaging dataset. SiQ-3D can be easily customized to analyze 3D microscopy data from both in vitro and in vivo imaging of diverse tissue/organ models that comprise multiple interacting cell types.

# Installation
1. Computational requirements: a computer with CUDA-enabled GPU, anaconda installed

2. Create a new conda environment
  ```
  $ conda env create -f SiQ-3D.yml
  ```
3. Activate the created conda environment
  ```
  $ conda activate SiQ-3D
  ```
4. Install the SiQ3D package
  ```
  $ pip install SiQ3D
  ```
  
