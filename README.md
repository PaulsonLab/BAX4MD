# BAX4MD
This repository contains code supporting the paper "MD-BAX: A General-Purpose Bayesian Design Framework for Molecular Dynamics Simulations with Input-Dependent Noise".

## Overview
- `/branin` contains UQ calibration results based on a synthetic problem with varying levels of input-dependent noise.
- `/data` contains the estimated Rg value and noise data for all the points in the parameter grid. We used the precomputed diblock dataset for both UQ calibration analysis and case studies on BAX applications.
- `/examples` contains notebooks illustrating two MD-BAX application: level set estimation (diblock) and manifold crawling (both diblock and triblock).
- `/src` contains the source code for MD-BAX. [`UQtools.py`](./src/UQtools.py) was adapted from [https://github.com/jensengroup/UQ_validation_methods](https://github.com/jensengroup/UQ_validation_methods).
- `/uq_calibration` contains UQ calibration results based on MD simulation data.


## Packages
- botorch 0.10.0
- gpytorch 1.11
- matplotlib 3.9.2
- numpy 1.26.4
- pandas 2.2.3
- pytorch 2.4.0 
- scikit-learn 1.1.3
- scipy 1.13.1


## Usage
To efficiently obtain meaningful statistical performance results, we used a pseudo simulator that directly reads the pre-computed real simulation outcomes from the data files when given specified inputs on a predefined grid. The  [`copolymer_simulator.py`](./src/copolymer_simulator.py) file can be adapted to a real simulator seamlessly integrated with our framework.
```sh
from copolymer_simulator import Simulator
simulator = Simulator(polymer='diblock') # diblock case
# simulator = Simulator(polymer='triblock') # triblock case
```

Our general-purpose framework can extend beyond tasks of level set estimation and manifold crawling shown in the paper. We provided the following *TaskHandler* class in [`bax.py`](./src/bax.py) as a template which can be easily adapted to customized tasks for MD simulation experiment design.
```sh
class TaskHandler:
    def __init__(self, task_kwargs):
        self.task_kwargs = task_kwargs

    def algorithm(self, sample_Y, max_var_indices):
        pass

    def get_valid_indices(self, bax, X, sample_Y, pred_Ymean, pred_Yvar, X1_range, X2_range, max_var_indices):
        raise NotImplementedError

    def post_step(self, bax, **kwargs):
        pass
```


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
