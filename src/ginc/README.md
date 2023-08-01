# GINC small-scale in-context learning dataset

The code in this subdirectory is sourced from [the original GINC repo](https://github.com/p-lambda/incontext-learning) and [Rylan Schaeffer](http://rylanschaeffer.github.io/).

## Installation & Quickstart

Please create a conda environment or virtualenv using the information in `conda-env.yml`, then install `transformers` by going into the `transformers/` directory and running `pip install -e .`.
Modify `consts.sh` to change the default output locations and insert code to activate the environment of choice.
Run `scripts/runner_generate.sh` to generate GINC datasets, and `scripts/runner_train.sh` to train GINC models.

[Source](https://github.com/p-lambda/incontext-learning#quickstart) of instructions above from original GINC repo. Note: the transformers repository included is required to reproduce GINC paper results.

## Usage
1. Run `scripts/runner_generate.sh` to generate GINC datasets using varied number of HMMs and number of symbols.
2. See `sweeps/train_reproduction_small.yaml`, `train_reproduction_mid.yaml`, `train_reproduction_large.yaml` specify hyperparameters for training 4-, 12-, and 16-layer GPT-2 Transformers respectively on GINC data. These files will train models with same parameters as in GINC paper.
4. Run `scripts/runner_train.sh` to kick off model training via creating wandb agents.

## Acknowledgements
We acknowledge [Rylan Schaeffer](http://rylanschaeffer.github.io/) for providing updated scripts `ginc/run_clm.py` and `ginc/run_eval.py`, and wandb integration.