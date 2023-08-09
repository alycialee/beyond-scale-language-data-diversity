"""
todo: wish list but not essential. I wish I could have passed the data set as a streaming data set, to avoid tokenizing all the 
data set before doing anything with it. e.g., the data sets might be huge, especially for training/fine-tuning. 
GPT4 suggested this: https://chat.openai.com/share/495de296-71c2-4f5e-83e2-3b22d038e8bc which seems reasonable.
It also would have made all the data set interfaces consistent in training vs computing data set metrics.
"""
from pathlib import Path
import datetime
import json

from torch import nn
import numpy as np

from datasets import load_dataset
from datasets import interleave_datasets

from diversity.task2vec import Task2Vec
import diversity.task_similarity as task_similarity

def get_diversity_coefficient(dataset,
                            map: callable,  # to ease whatever ars you want to batch.map for any data set
                            probe_network: nn.Module,
                            tokenizer = None,
                            batch_size: int = 512,
                            num_batches: int = 600, 
                            seed = 0, 
                            buffer_size: int = 500_000, 
                            distance = 'cosine',
                            verbose: bool = True,
                            debug: bool = False,
                          ) -> dict:
    """
    Compute the diversity coefficient of a dataset using a probe network.
    Return all results in a dictionary since it's often useful to store them to avoid recomputing them.
    If you want the diveristy coefficient and it's confidence interval (ci), use the following:
        div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    """
    if num_batches < 3:
        print(f'Warning: num_batches must be >= 3, but got {num_batches=} otherwise you only get 1 comparison so 1 distance value')
    # - Compute embeddings
    embeddings, losses = [], []
    for batch_num in range(num_batches):
        print(f'--> {batch_num=}\n') if verbose else None
        # - Get batch
        shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        raw_text_batch = dataset.take(batch_size)
        tokenized_batch = map(raw_text_batch)
        # if verbose:
        #     print(f'{type(tokenized_batch)=}')
        #     print(f'{next(iter(tokenized_batch))=}')

        # - Get Task2Vec embedding for batch
        if not debug:
            embedding, loss = Task2Vec(probe_network).embed(tokenized_batch)
        else:
            embedding, loss = Task2Vec(probe_network, classifier_opts={'break_early': True}).embed(tokenized_batch, epochs=1)  # only for debugging
        print(f'{loss=}\n{embedding=}\n') if verbose else None
        
        # - Collect results
        embeddings.append(embedding)
        losses.append(loss)
        
    # - Compute diversity coefficient
    distance_matrix = task_similarity.pdist(embeddings, distance=distance)
    div_coeff, div_coeff_ci = task_similarity.stats_of_distance_matrix(distance_matrix)

    # -- Return results
    results: dict = {'div_coeff': div_coeff, 'div_coeff_ci': div_coeff_ci,
                    'embeddings': embeddings,
                    'distance_matrix': distance_matrix,
                    'losses': losses,
                    "num_batches": num_batches}
    return results

def cross_diversity_coefficient(dataset_target,
                                dataset_source,
                                get_mapped_batch_fn: callable, 
                                probe_network: nn.Module,
                                tokenizer = None,
                                batch_size: int = 512,
                                num_batches: int = 100, 
                                seed = 0, 
                                buffer_size: int = 500_000, 
                                distance = 'cosine',
                          ) -> dict:
    """ 
    Todo: ask Alycia how she did this in the paper. Please compare with her implementation and report which one we prefer.     
    """
    # - Compute embedding of target
    lossses: list[dict] = []
    embeddings: list[dict] = []
    cross_distances = []
    for batch_num in range(num_batches):
        # - Compute embedding of target
        shuffled_dataset = dataset_target.shuffle(buffer_size=buffer_size, seed=seed)
        tokenized_batch = get_mapped_batch_fn(shuffled_dataset)
        embedding_target, loss_target = Task2Vec(probe_network).embed(tokenized_batch)

        # - Compute embedding of source
        shuffled_dataset = dataset_source.shuffle(buffer_size=buffer_size, seed=seed)
        tokenized_batch = get_mapped_batch_fn(shuffled_dataset)
        embedding_source, loss_source = Task2Vec(probe_network).embed(tokenized_batch)

        # - Compute cross distance
        distance_matrix = task_similarity.pdist([embedding_target, embedding_source], distance=distance)
        cross_dist = distance_matrix[0, 1]

        # - Collect results
        losses.append({'loss_target': loss_target, 'loss_source': loss_source})
        embeddings.append({'embedding_target': embedding_target, 'embedding_source': embedding_source})
        cross_distances.append(cross_dist)
    
    # - Compute cross diversity coefficient
    div_coeff, div_coeff_ci = task_similarity.stats_of_distance_matrix(cross_distances)

    # -- Return results
    results: dict = {'div_coeff': div_coeff, 'div_coeff_ci': div_coeff_ci,
                    'embeddings': embeddings,
                    'distance_matrix': distance_matrix,
                    'losses': losses,
                    "num_batches": num_batches}
    return results
    
def get_tokenized_batch(batch):
    return next(iter(batch))

def preprocess(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    
def my_map(batch):
    return batch.map(preprocess, batched=True, remove_columns=[])

def print_examples_from_dataset(dataset, preprocess=preprocess, map=my_map, batch_size=100):
    print('\n---- Examples from dataset ----')
    batch = dataset.take(batch_size)
    # batch = map(batch)
    counts = 0
    for example in list(batch):
        # print()
        print(f'{example.keys()=}')
        if 'url' in example:
            counts += 1
            # print(f'{example=}')
        else:
            print(f'{example=}')
    print(f'{counts=}')

# -- Tests, Examples

def test_get_batch_from_dataset():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    max_seq_length = 128
    batch_size = 512
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token

    # -- Get batch from dataset
    # path, name = 'c4', 'en'
    path, name = "wikitext", 'wikitext-103-v1'
    dataset = load_dataset(path, name, streaming=True, split="train").with_format("torch")
    batch = dataset.take(batch_size)
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=max_seq_length, truncation=True, return_tensors="pt")
    print(f'{batch=}')
    print(f'{next(iter(batch))=}')
    remove_columns = ["text", "timestamp", "url"] if path == "c4" else []
    tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)
    print(f'{tokenized_batch=}')
    print(f'{next(iter(tokenized_batch))=}')
    # Running /lfs/ampere1/0/brando9/beyond-scale-language-data-diversity/src/diversity/div_coeff.py
    # batch=<datasets.iterable_dataset.IterableDataset object at 0x7ff29031fd60>
    # tokenized_batch=<datasets.iterable_dataset.IterableDataset object at 0x7ff2901889d0>
    # Time it took: 0.9143445491790771 seconds

def alycias_original_colab_code():
    """ https://colab.research.google.com/drive/1pL1JmE2LuRg5ClsZ7htFyzEAJ7iMEcys#scrollTo=aRI6a_27Tzd0 """
    # -- Get probe network
    from diversity.task2vec import Task2Vec

    from datasets import load_dataset
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    probe_network = probe_network.to(device)

    # -- Get data set & data batch (as a data set obj)
    dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    remove_columns = ["text", "timestamp", "url"]
    max_seq_length = 128
    batch_size = 512
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=max_seq_length, truncation=True, return_tensors="pt")
    batch = dataset.take(batch_size)
    tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)

    # embedding, loss = Task2Vec(probe_network).embed(tokenized_batch) 
    # embedding, loss = Task2Vec(probe_network).embed(tokenized_batch, epochs=1)

    from diversity.task_similarity import pdist, plot_distance_matrix, \
                                      stats_of_distance_matrix

    num_batches = 3
    if num_batches < 2:
        raise ValueError(f'num_batches must be >= 3, but got {num_batches=} otherwise you only get 1 comparison so 1 distance value')
    buffer_size = 500_000
    seed = 0
    embeddings = []
    for batch_num in range(num_batches):
        print(f'--> {batch_num=}\n')
        shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        batch = shuffled_dataset.take(batch_size)
        tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)

        # embeddings.append(Task2Vec(probe_network).embed(tokenized_batch)[0])
        # embeddings.append(Task2Vec(probe_network).embed(tokenized_batch, epochs=1)[0])  # only for debugging
        embeddings.append(Task2Vec(probe_network, classifier_opts={'break_early': True}).embed(tokenized_batch, epochs=1)[0])  # only for debugging

    print(f'{len(embeddings)=}')
    distance_matrix = pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')
    div_coeff, conf_interval = stats_of_distance_matrix(distance_matrix)
    print(f'Diversity: {(div_coeff, conf_interval)=}')
    plot_distance_matrix(embeddings)

    import numpy as np
    from pathlib import Path

    output_dir = Path('./').expanduser()
    np.save(output_dir / 'distance_matrix.npy', distance_matrix)
    results: dict = {'embeddings': [embed for embed in embeddings],
                    'distance_matrix': distance_matrix,
                    "num_batches": num_batches}
    np.save(output_dir / 'results.npy', results)    

def test_diversity_coefficient():
    # -- Get probe network
    from datasets import load_dataset
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    probe_network = probe_network.to(device)

    # -- Get data set
    dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    remove_columns = ["text", "timestamp", "url"]
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    # batch = dataset.take(batch_size)
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    # tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)

    # -- Compute diversity coefficient
    # results: dict = get_diversity_coefficient(dataset, map, probe_network)
    results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=3)  # only for debugging
    div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    print(f'{div_coeff=} {div_coeff_ci=}')

    # -- Save results or not
    save_results = True
    if save_results:
        today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%H_%M')
        output_dir = Path(f'~/data/div_coeff/{today}').expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'distance_matrix{today}.npy', results['distance_matrix'])
        np.save(output_dir / f'results{today}.npy', results)
        # Save results as a pretty-printed JSON
        results = {key: str(value) for key, value in results.items()}
        with open(output_dir / f'results{today}.json', 'w') as f:
            json.dump(results, f, indent=4)

# -- Experiments

def experiment_compute_diveristy_coeff_singlee_dataset_then_combined_datasets_with_domain_weights():
    """
    Get divs using pt ft, pt (rand, rand ft?) 
    - div c4 
    - div wt = wt-103
    Then with unioned datasets
    - div c4+wt, uniform
    - div c4+wt, data set size proportions (using GBs)
    - div c4+wt, respect doremi
    - div c4+wt, respect the pile
    - div c4+wt, respect gpt3 weights
    then repeat all with pt (no ft)
    """
    # -- Setup wandb
    import wandb
    # - Dryrun
    mode = 'dryrun'
    num_batches = 3

    # - Online (real experiment)
    # mode='online'
    # num_batches = 600
    # path, name = 'c4', 'en'
    # path, name = "wikitext", 'wikitext-103-v1'
    # probabilities = None
    path, name = ['c4', 'wikitext'], ['en', 'wikitext-103-v1']
    # path, name = ['wikitext', 'wikitext'], ['wikitext-103-v1', 'wikitext-103-v1']
    probabilities = [1.0/len(path)] * len(path)
    # probablilities = [0, 1.0]
    # not changing
    batch_size = 512
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    run_name = f'{path} div_coeff_{num_batches=} ({today=} {probabilities=})'
    print(f'{run_name=}')

    # - Init wandb
    wandb.init(mode=mode, project="beyond-scale", name=run_name, save_code=True)
    wandb.config.update({"num_batches": num_batches, "path": path, "name": name, "today": today, 'probabilities': probabilities, 'batch_size': batch_size})
    debug: bool = mode == 'dryrun'
    print(f'{debug=}')
    print(f'{wandb.config=}')

    # -- Get probe network
    from datasets import load_dataset
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    probe_network = probe_network.to(device)

    # -- Get data set
    remove_columns = []
    if isinstance(path, str):
        # dataset = load_dataset(path, name, streaming=True, split="train").with_format("torch")
        # remove_columns = ["text", "timestamp", "url"] if path == 'c4' else []
        raise NotImplementedError
    else:
        datasets = [load_dataset(path, name, streaming=True, split="train").with_format("torch") for path, name in zip(path, name)]
        [print(f'{dataset.description=}') for dataset in datasets]
        dataset = interleave_datasets(datasets, probabilities)
    print(f'{dataset=}')
    # batch = dataset.take(batch_size)
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    # tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)
    print_examples_from_dataset(dataset, batch_size=100)

    # -- Compute diversity coefficient
    results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=num_batches, debug=debug)
    div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    print(f'{div_coeff=} {div_coeff_ci=}')
    wandb.log({'div_coeff': div_coeff, 'div_coeff_ci': div_coeff_ci})

    # -- Save results or not
    save_results = True
    if save_results:
        output_dir = Path(f'~/data/div_coeff/{today}').expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'distance_matrix{today}.npy', results['distance_matrix'])
        np.save(output_dir / f'results{today}.npy', results)
        # Save results as a pretty-printed JSON
        results = {key: str(value) for key, value in results.items()}
        with open(output_dir / f'results{today}.json', 'w') as f:
            json.dump(results, f, indent=4)
        # - wandb save
        base_path = str(output_dir.parent)
        wandb.save(str(output_dir / f'distance_matrix{today}.npy'), base_path=base_path)
        wandb.save(str(output_dir / f'results{today}.npy'), base_path=base_path)
        wandb.save(str(output_dir / f'results{today}.json'), base_path=base_path)
        wandb.save(__file__)
    
    # -- Finish wandb
    wandb.finish()

if __name__ == '__main__':
    print(f'\n\n\n------------------- Running {__file__} -------------------')
    # -- Run tests and time it
    import time
    time_start = time.time()
    # -- Run tests
    # test_get_batch_from_dataset()
    # alycias_original_colab_code()
    # test_diversity_coefficient()
    experiment_compute_diveristy_coeff_singlee_dataset_then_combined_datasets_with_domain_weights()
    # -- End tests, report how long it took
    print(f'Time it took: {time.time() - time_start} seconds \a\n')