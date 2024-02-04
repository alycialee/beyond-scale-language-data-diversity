# Summary

**gpt2_train.py** 
-
**Desc**: Script for training a custom architecture GPT-2 model from scratch on a HuggingFace Dataset for a specific number of tokens.

**Tested/Checked?**: Yes (fairly thoroughly).

**Works on SNAP?**: Yes (based on a basic test on Hyperturing2).

**How to run a fair comparison**:

At a high level:
- We use the same architecture, same number of parameters, same number of training steps, same optimizer, same hyperparams (learning rate, etc.), and same 'token pool.'
- Vary the training dataset (by changing the name of the training dataset passed in).
- This allows us to control for essentially all variables except for the diversity of the training data. Therefore, treating diversity as our independent variable, we will know that differences in the trained model (dependent variable) likely stem from differences in their training diversity.

Commands to run:
```
conda activate train_test1

export CUDA_VISIBLE_DEVICES=1,2,3,4  # Select which GPUs to use for training

python gpt2_train.py \
--output_dir path/to/dir \                    # Directory where checkpoints are saved. Logistics
--config_overrides n_embd=x,n_layer=y,n_head=z  # Architecture and number of parameters of each model. Constant
--do_train \
--per_device_train_batch_size b \               # Number of examples seen per training batch. Constant
--max_steps n \                             # Total number of optimization steps, i.e. total number of batches seen. Constant
--save_steps s \                             # How many optimization steps to wait before saving another checkpoint. Constant
--optim adamw_torch \                           # Optimizer to use. Constant
--dataset_name VARIABLE_DATASET \               # Overall dataset name. VARY
--dataset_config_name VARIABLE_SUBDATASET \     # Sub-dataset name, if applicable. VARY
--streaming CacheStream                         # Data loading (i.e.) streaming method. Logistics 

# all non-specified hyperparameters are set to the default for TrainingArguments (see https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#transformers.TrainingArguments). Constant
# e.g. learning rate set to default (5e-05), adam parameters set to default (beta1=0.9, beta2=0.999, epsilon=1e-08)
```

For my experiments, I ran
```
export CUDA_VISIBLE_DEVICES=1,2,3,4

python gpt2_train.py \
--output_dir results_<dataset> \                    
--config_overrides n_embd=512,n_layer=8,n_head=8
--do_train \
--per_device_train_batch_size 16 \
--max_steps 20000 \
--save_steps 2000 \
--optim adamw_torch \
--dataset_name EleutherAI/pile \
--dataset_config_name <dataset> \
--streaming CacheStream
```
for \<dataset\> in \[PubMed, USPTO, PubMed interleaved with USPTO\]

Hence, the important decisions for training were:
- _Using a ~50 M parameter GPT-2 model with embedding size 512, 8 layers, and 8 heads._ I chose this setup because the overall parameter count is similar to the smallest models used in Eleuther's Pythia series of models and appears to provide the minimum viable scale to test differences in ability from diverse pretraining data, and the configuration of embedding size, layers, and heads is also similar to those already tested and used by the Pythia series of models. This is held constant, as it's not our independent variable.
- _Using a batch size of 64 examples per batch._ I chose this setup since this was the largest batch size that trained efficiently given the compute I had access to. Furthermore, this is equivalent to 64*1024 ~= 65.5k tokens per batch, which is within and OOM of GPT-3's token batch sizes (~250k tokens https://discuss.huggingface.co/t/how-to-choose-optimal-batch-size-for-training-llms/23861). This is held constant, as it's not our independent variable.
- _Using 20k optimization steps._ I chose this setup since, by the end of training, each model will have seen 1024\*64\*20000 ~= 1.31B tokens. This is the same OOM as the tokens seen by larger GPT-2 models (~9B tokens for 125M to 1.5B parameter models https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext). This is held constant, as it's not our independent variable.
- _Using the adamw optimizer._ I used this optimizer since it's a standard optimizer used for training LLMs. This is held constant, as it's not our independent variable.
- _Using PubMed, USPTO, PubMed interleaved with USPTO datasets (subsets of the Pile)._ I used the PubMed dataset and USPTO dataset since they have different diversity coefficients (they differ by ~0.01), but are lower diversity than many other datasets tested (i.e. they have diversity coefficient < 0.20). I use the PubMed interleaved with USPTO dataset since it contains the same content as the other two datasets, but is more diverse than each individual dataset (~0.01 more than PubMed, ~0.02 more than USPTO). Therefore, using this mix of datasets allows me to test variation in diversity at relatively low levels (< 0.20), while holding constant the 'pool of tokens' that the data is chosen from, i.e. we test the effect of diversity of the training data directly without the confounding variable of different datasets' overlap and similarity with the training data. E.g. if the driver of performance is merely that PubMed data has more overlap/similarity to the eval data, then we should see the PubMed-trained model perform best (and, in particular, better than the more diverse interleaved model); on the other hand, if the inherent diversity of the training data is the most important factor, then we should see the interleaved model perform best.
- _Setting all other hyperparameters to default._ I chose this setup since the HuggingFace defaults are likely reasonably set to work well across a range of LLM objectives, and, importantly, this choice keeps these non-independent variables controlled/constant across all models.




**cache_datasets.py**
-
**Desc**: Helper script that allows you to run HF dataset caching in the background.

**Tested?**: Yes (assumed, nothing much you can mess up here)

**Works on SNAP?**: Yes (assumed, nothing much you can mess up here)






**gpt2_eval.py**
-
**Desc**: Script for evaluating a HF trainer checkpoint (i.e. a model) on a given HF dataset. Currently configured to calculate the following metrics: cross-entropy loss, perplexity, accuracy. Details on using the script, command templates, etc. can be found in multi-line comments in the script.

**Tested?**: Yes (fairly thoroughly).

**Works on SNAP?**: Yes (did basic test).




Other files
-
Inference pipeline is left unimplemented. Pretty easy to implement if need be, but don't currently see a good reason to. All core scripts needed for training and evaluation are implemented and v1 tested. Feel free to let me something else you'd like to see! [11/10]

