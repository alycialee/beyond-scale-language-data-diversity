# Summary

**gpt2_train.py** 
-
**Desc**: Script for training a custom architecture GPT-2 model from scratch on a HuggingFace Dataset for a specific number of tokens.

**Tested?**: Mostly, about 80% done. 

**Works on SNAP?**: Yes (based on a basic test on Hyperturing2)

**How to run a fair comparison**:
High level:
- Use the same architecture, same number of parameters, same number of training steps (specified in run command), same optimizer, and same hyperparams (learning rate, etc.).
- Vary the training dataset (by changing the name of the training dataset passed in)

Commands to run:
```
conda activate train_test1

export CUDA_VISIBLE_DEVICES=1,2,3,4  # Select which GPUs to use for training

python gpt2_train.py \
--output_dir scrap_results \                    # Directory where checkpoints are saved. Logistical variable
--config_overrides n_embd=x,n_layer=y,n_head=z  # Architecture and number of parameters of each model. Constant
--do_train \
--per_device_train_batch_size 8 \               # Number of examples seen per training batch. Constant
--max_steps 20000 \                             # Total number of optimization steps, i.e. total number of batches seen. Constant
--save_steps 2000 \                             # How many optimization steps to wait before saving another checkpoint. Constant
--optim adamw_torch \                           # Optimizer to use. Constant
--dataset_name VARIABLE_DATASET \               # Overall dataset name. VARY
--dataset_config_name VARIABLE_SUBDATASET \     # Sub-dataset name, if applicable. VARY
--streaming CacheStream                         # Data loading (i.e.) streaming method. Logistical variable 

# all non-specified hyperparameters are set to the default for TrainingArguments (see https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#transformers.TrainingArguments)
# e.g. learning rate set to default (5e-05), adam parameters set to default (beta1=0.9, beta2=0.999, epsilon=1e-08)
```


**cache_datasets.py**
-
**Desc**: Helper script that allows you to run HF dataset caching in the background.

**Tested?**: Yes (assumed, nothing much you can mess up here)

**Works on SNAP?**: Yes (assumed, nothing much you can mess up here)

Other files
-
Inference pipeline and Evaluation pipeline have not yet been tested/refined--should only take a day or two more, currently working on this. Will update this heading with progress [10/25]

