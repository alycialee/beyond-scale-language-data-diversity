actively maintained repo: https://github.com/brando90/beyond-scale-language-data-diversity

# Beyond Scale: the Diversity Coefficient as a Data Quality Metric for Natural Language Datasets

This repository provides the official implementation of the Task2Vec Diversity Coefficient for computing natural language data diversity from the following paper:

**Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data**.
Alycia Lee, Brando Miranda, Sanmi Koyejo.
Paper: https://arxiv.org/abs/2306.13840

This repository also contains code for generating GINC datasets and computing the Diversity Coefficient of those datasets (see `ginc/`).

## Getting Started
`diversity/` contains the Task2Vec diversity coefficient computation for natural language data. [**See Quick-start**](https://github.com/alycialee/beyond-scale-language-data-diversity/blob/main/src/diversity/README.md#quick-start) for a tutorial of computing the diversity coefficient for a language dataset.** Run `diversity/runner.sh` to compute Task2Vec embeddings and diversity coefficient for c4, WikiText-103, and The Pile.

When cloning your main repository in the future, you will need to initialize the submodules as well by using:
```bash
cd ~
git clone --recurse-submodules git@github.com:brando90/beyond-scale-language-data-diversity.git
```
If you forget to use --recurse-submodules, you can still initialize the: 
```bash
git clone https://github.com/<user>/beyond-scale-language-data-diversity.git
cd ~/beyond-scale-language-data-diversity
git submodule update --init --recursive
```
Note: to push the changes to submodule cd there and do git cmds there.
In the main repo folder to do git pushes/edits to the main repo code. 

For all experiments including the `ginc` data set:
`ginc/` contains Generative In-Context learning Dataset from [the original GINC repo](https://github.com/p-lambda/incontext-learning). 

Run `ginc/runner_generate.sh` to generate GINC datasets with varying number of HMMs and number of symbols. 
Run `ginc/runner_train.sh` to train GPT-2 Transformers on GINC datasets using wandb.

## Conda Install
Create conda env:
```bash
conda create -n beyond_scale_div_coeff python=3.11 -y
# conda activatexport HOME=/data/
conda activate beyond_scale_div_coeff
pip install -e ~/beyond-scale-language-data-diversity
# conda remove --name beyond_scale_diiv_coeff --all
```

## Venv Install
Create venv python environment:
```bash
deactivate
# mkdir ~/.virtualenvs
ls ~/.virtualenvs
python3.11 -m venv ~/.virtualenvs/beyond_scale_div_coeff
source ~/.virtualenvs/beyond_scale_div_coeff/bin/activate
pip install --upgrade pip
which python
pip install -e ~/beyond-scale-language-data-diversity
```

## Acknowledgements
We acknowledge that code in `ginc/` was sourced from [the original GINC repo](https://github.com/p-lambda/incontext-learning). 
We thank [Rylan Schaeffer](http://rylanschaeffer.github.io/) for his contributions to updating the scripts in `ginc/` for ease of usage.

## Citation

If you found this repo useful, please cite
```
@article{lee2023scale,
      author={Alycia Lee and Brando Miranda and Sanmi Koyejo},
      journal={arXiv preprint arXiv:2306.13840},
      title={Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data}, 
      year={2023},
}
```
