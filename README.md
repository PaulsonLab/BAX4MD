# BAX4MD
This repository contains code supporting the paper "MD-BAX: A General-Purpose Bayesian Design Framework for Molecular Dynamics Simulations with Input-Dependent Noise".

## Overview
- `branin/` contains UQ calibration results based on a synthetic problem with varying levels of input-dependent noise.
- `data/copolymer/` contains the estimated mean and noise data for all the points in the parameter grid. We used this precomputed dataset for both UQ calibration analysis and case studies on BAX applications.
- `examples/` contains notebooks illustrating two MD-BAX application: level set estimation and manifold crawling.
- `src/` contains the source code for MD-BAX. [`UQtools.py`](./src/UQtools.py) was adapted from [https://github.com/jensengroup/UQ_validation_methods](https://github.com/jensengroup/UQ_validation_methods).
- `uq_calibration` contains UQ calibration results based on MD simulation data.


## Packages
- botorch 0.10.0
- gpytorch 1.11
- matplotlib 3.9.2
- numpy 1.26.4
- pandas 2.2.3
- pytorch 2.4.0 
- scikit-learn 1.1.3
- scipy 1.13.1


## Citation
```
@article{
author = {Tianhong Tan, Ting-Yeh Chen, Jacob R. Breese, Lisa M. Hall, Joel A. Paulson},
title = {MD-BAX: A General-Purpose Bayesian Design Framework for Molecular Dynamics Simulations with Input-Dependent Noise},
journal = {},
volume = {},
number = {},
pages = {},
doi = {},
abstract = {}}
```
