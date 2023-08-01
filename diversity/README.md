# Task2Vec Diversity Coefficient for Measuring Natural Language Data Diversity

The code in this subdirectory computes Task2Vec embeddings and pairwise cosine distance matrix of natural language datasets using GPT-2 as probe network. Currently, streaming the following open-source, HuggingFace datasets is supported:
- [c4](https://huggingface.co/datasets/c4)
- [WikiText-103](https://huggingface.co/datasets/wikitext)
- [The Pile](https://huggingface.co/datasets/the_pile)
    - sub-datasets of The Pile: Pile-CC, Enron Emails, Hacker News, NIH exporter, PubMed, USPTO

## Setup

### Requirements
Please create a conda environment or virtualenv using the information in `div-conda-env.yml`. Here are steps for manual library installation:
```
conda install -c anaconda scikit-learn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge transformers
conda install -c conda-forge datasets
```

## Quick start
To compute a Task2Vec embedding of a language dataset, provide a language model as a probe network and a tokenized dataset:
```python
from task2vec import Task2Vec
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
probe_network = GPT2LMHeadModel.from_pretrained("gpt2")

dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
remove_columns = ["text", "timestamp", "url"]

def preprocess(examples):
    return tokenizer(examples["text"], return_tensors="pt")
    
batch_size = 512
batch = dataset.take(batch_size)
tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)

embedding, loss = Task2Vec(probe_network).embed(tokenized_batch)
```
The [Task2Vec](https://arxiv.org/abs/1902.03545) method computes task embeddings using the diagonal entries of the Fisher Information Matrix. Task2Vec provides two methods for computing the embedding: `montecarlo` and `variational`. In this implementation, we additionally provide an `autoregressive` mode for computing Task2Vec that uses autoregressive cross entropy loss and the `montecarlo` method. The `autoregressive` mode is enabled by default, but to run vanilla Task2Vec using `variatonal` method, you can use:

```python
Task2Vec(probe_network, method='variational').embed(dataset)
```
or to run vanilla Task2Vec using `montecarlo` method, you can use:

```python
Task2Vec(probe_network, method='montecarlo').embed(dataset)
```

Now, let's compute embeddings for many batches sampled randomly from the dataset. Then we can plot the distance matrix between tasks and compute the diversity coefficient:

```python
import task_similarity

num_batches = 100
buffer_size = 500_000
seed = 1111
embeddings = []
for batch_num in range(num_batches):
    print(f'--> {batch_num=}\n')
    shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    batch = shuffled_dataset.take(batch_size)
    tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)

    embeddings.append(Task2Vec(probe_network).embed(tokenized_batch)[0])

distance_matrix = task_similarity.pdist(embeddings, distance='cosine')
task_similarity.plot_distance_matrix(embeddings, ["c4"])
div_coeff, conf_interval = task_similarity.stats_of_distance_matrix(distance_matrix)
```
or with the following function and store your results somewhere
```python
div_coeff, conf_interval = task_similarity.diversity_coefficient(distance_matrix)

from pathlib import Path
output_dir = Path('./').expanduser()
np.save(output_dir / 'distance_matrix.npy', distance_matrix)
results: dict = {'embeddings': [embed for embed in embeddings],
                'distance_matrix': distance_matrix,
                "num_batches": args.num_batches}
np.save(output_dir / 'results.npy', results)
```

## Usage
Run `scripts/runner.sh` to compute 200 Task2Vec embeddings of c4, WikiText-103, and The Pile, each with batch size of 512 using pretrained finetuned network. Alternatively, you can run `main.py` directly using the following python commands:

```
# Compute embeddings for c4
python main.py --task_name c4 --num_tasks 200 --batch_size 512 --buffer_size 500_000 --finetune --pretrained --output_dir output_c4_200tasks_bs512_gpt2_pt_ft --cache_dir cache_dir

# Compute embeddings for WikiText-103
python main.py --task_name wikitext --num_tasks 200 --batch_size 512 --buffer_size 500_000 --finetune --pretrained --output_dir output_wt_200tasks_bs512_gpt2_pt_ft --cache_dir cache_dir

# Compute embeddings for The Pile
python main.py --task_name the_pile --num_tasks 200 --batch_size 512 --buffer_size 500_000 --finetune --pretrained --output_dir output_thepile_200tasks_bs512_gpt2_pt_ft --cache_dir cache_dir
```

Run `scripts/runner_thepile_subdataset.sh` to compute 200 Task2Vec embeddings each for every supported sub-dataset of The Pile, each with batch size of 512 using pretrained finetuned network. Alternatively, you can run `main.py` directly to compute the embeddings for Pile-CC and Enron Emails using the following python commands:

```
# Compute embeddings for Pile-CC and Enron Emails subdatasets of The Pile
python main.py --task_name the_pile_sametaskds --subdataset first --num_tasks 200 --batch_size 512 --buffer_size 500_000 --finetune --pretrained --output_dir output_thepile_first_200tasks_bs512_gpt2_pt_ft --cache_dir cache_dir
```

Run `scripts/runner_ginc.sh` to compute Task2Vec embeddings for generated GINC datasets. Alternatively, you can run `main_ginc.py` directly:

```
python main_ginc.py --batch_size 512 --finetune --pretrained --cache_dir cache_dir --n_hmms=10 --n_symbols=50
```

## Additional Notes
Running `main.py` produces the following outputs:
- `run_args.txt`: args for this run
- `embeddings_[num_tasks]tasks.npy`: embeddings for num_tasks
- `loss_[num_tasks]tasks.npy`: cross entropy loss from finetuning for num_tasks (if finetuning is specified)
- `results.npy`: pairwise cosine distance matrix, embeddings, losses for num_tasks

`task2vec.py`, `task_similarity.py`, and `utils.py` were adapted from [ultimate-aws-cv-task2vec](https://github.com/brando90/ultimate-aws-cv-task2vec) and originally sourced from [aws-cv-task2vec](https://github.com/awslabs/aws-cv-task2vec). Comments `## LLM DIV` demarcate code and methods that were added or modified for computing the diversty coefficient of LM datasets.

`notebooks/plot.ipynb` takes output files from running `main.py` and plots figures included in the arxiv preprint.