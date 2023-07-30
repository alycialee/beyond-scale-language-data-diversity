# Beyond Scale: the Diversity Coefficient as a Data Quality Metric for Natural Language Datasets

This repository provides the official implementation of the Task2Vec Diversity Coefficient for computing natural language data diversity from the following paper:

**Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data**.
Alycia Lee, Brando Miranda, Sanmi Koyejo.
Paper: https://arxiv.org/abs/2306.13840

This repository also contains code for generating GINC datasets and computing the Diversity Coefficient of those datasets (see `ginc/`).

## Getting Started
`diversity/` contains the Task2Vec diversity coefficient computation for natural language data. **See [#Quick-start](https://github.com/alycialee/beyond-scale-language-data-diversity/tree/main/diversity#quick-start) for a tutorial of computing the diversity coefficient for a language dataset.** Run `diversity/runner.sh` to compute Task2Vec embeddings and diversity coefficient for c4, WikiText-103, and The Pile.

`ginc/` contains Generative In-Context learning Dataset from [the original GINC repo](https://github.com/p-lambda/incontext-learning). 

Run `ginc/runner_generate.sh` to generate GINC datasets with varying number of HMMs and number of symbols. Run `ginc/runner_train.sh` to train GPT-2 Transformers on GINC datasets using wandb.

## Acknowledgements
We acknowledge that code in `ginc/` was sourced from [the original GINC repo](https://github.com/p-lambda/incontext-learning). We thank [Rylan Schaeffer](http://rylanschaeffer.github.io/) for his contributions to updating the scripts in `ginc/` for ease of usage.

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