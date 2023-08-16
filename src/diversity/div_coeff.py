"""
todo: wish list but not essential. I wish I could have passed the data set as a streaming data set, to avoid tokenizing all the 
data set before doing anything with it. e.g., the data sets might be huge, especially for training/fine-tuning. 
GPT4 suggested this: https://chat.openai.com/share/495de296-71c2-4f5e-83e2-3b22d038e8bc which seems reasonable.
It also would have made all the data set interfaces consistent in training vs computing data set metrics.
"""
import time

from pathlib import Path
import datetime
import json

from torch import nn
import numpy as np
import random

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
                            seed: int = 0, 
                            buffer_size: int = 500_000, 
                            distance = 'cosine',
                            verbose: bool = False,
                            debug: bool = False,
                            shuffle: bool = True,  # False for faster debugging/testing but it won't be shuffled
                          ) -> dict:
    """
    Compute the diversity coefficient of a dataset using a probe network.
    Return all results in a dictionary since it's often useful to store them to avoid recomputing them.
    If you want the diveristy coefficient and it's confidence interval (ci), use the following:
        div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    """
    print(f'{shuffle=}')
    if num_batches < 3:
        print(f'Warning: num_batches must be >= 3, but got {num_batches=} otherwise you only get 1 comparison so 1 distance value')
    # - Compute embeddings
    embeddings, losses = [], []
    for batch_num in range(num_batches):
        print(f'--> {batch_num=}\n')
        # - Get batch
        shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else dataset
        raw_text_batch = shuffled_dataset.take(batch_size)
        # raw_text_batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(range(batch_size))
        tokenized_batch = map(raw_text_batch)
        if verbose:
            print(f'{raw_text_batch=}')
            print(f'{tokenized_batch=}')
            # time_start = time.time()
            # print(f'{next(iter(raw_text_batch))=}')
            # print(f'{next(iter(tokenized_batch))=}')
            # print(f'Time it took: {time.time() - time_start} seconds \a\n')

        # - Get Task2Vec embedding for batch
        if not debug:
            embedding, loss = Task2Vec(probe_network, classifier_opts={'seed': seed}).embed(tokenized_batch)
        else:
            embedding, loss = Task2Vec(probe_network, classifier_opts={'break_early': True, 'seed': seed}).embed(tokenized_batch, epochs=1)  # only for debugging
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
                                map_target: callable, 
                                map_source: callable,
                                probe_network: nn.Module,
                                tokenizer = None,
                                batch_size: int = 512,
                                num_batches: int = 100, 
                                seed = 0, 
                                buffer_size: int = 500_000, 
                                distance = 'cosine',
                                verbose: bool = False,
                                debug: bool = False,
                                shuffle: bool = True,  # False for faster debugging/testing but it won't be shuffled
                          ) -> dict:
    """ """
    # - Compute embedding of target
    lossses: list[dict] = []
    embeddings: list[dict] = []
    cross_distances = []
    for batch_num in range(num_batches):
        # - Get target shuffled data
        shuffled_dataset = dataset_target.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else dataset_target
        raw_text_batch = shuffled_dataset.take(batch_size)
        tokenized_batch = map_target(raw_text_batch)
        
        # - Get Task2Vec embedding for batch
        if not debug:
            embedding_target, loss_target = Task2Vec(probe_network, classifier_opts={'seed': seed}).embed(tokenized_batch)
        else:
            embedding_target, loss_target = Task2Vec(probe_network, classifier_opts={'break_early': True, 'seed': seed}).embed(tokenized_batch, epochs=1)  # only for debugging
        print(f'{loss_target=}\n{embedding_target=}\n') if verbose else None

        # - Get source shuffled data
        shuffled_dataset = dataset_source.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else dataset_source
        raw_text_batch = shuffled_dataset.take(batch_size)
        tokenized_batch = map_target(raw_text_batch)
        
        # - Get Task2Vec embedding for batch
        if not debug:
            embedding_source, loss_source = Task2Vec(probe_network, classifier_opts={'seed': seed}).embed(tokenized_batch)
        else:
            embedding_source, loss_source = Task2Vec(probe_network, classifier_opts={'break_early': True, 'seed': seed}).embed(tokenized_batch, epochs=1)  # only for debugging
        print(f'{loss_source=}\n{embedding_source=}\n') if verbose else None

        # - Append results to save later
        embeddings.append({'embedding_target': embedding_target, 'embedding_source': embedding_source})
        losses.append({'loss_target': loss_target, 'loss_source': loss_source})

    # - Compute cross diversity coefficient
    # todo: embeddings list -> to two seperate lists
    cross_distance_matrix = task_similarity.cross_pdist(embeddings, distance=distance)
    cross_div_coeff, cross_div_coeff_ci = task_similarity.stats_of_distance_matrix(cross_distance_matrix)

    # -- Return results
    results: dict = {'cross_div_coeff': cross_div_coeff, 'cross_div_coeff_ci': cross_div_coeff_ci,
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
        print(f'{example["text"]=}')
    #     print(f'{example.keys()=}')
    #     if 'url' in example:
    #         counts += 1
    #         # print(f'{example=}')
    #     else:
    #         print(f'{example=}')
    # print(f'{counts=}')

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
    print(f'Running function: {test_diversity_coefficient=}')
    batch_size = 512

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
    print(f'{dataset=}')
    batch = dataset.take(batch_size)
    print(f'{next(iter(batch))=}')

    # - Prepare functions to tokenize batch
    time_start = time.time()
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized_batch = map(batch)
    print(f'{next(iter(tokenized_batch))=}')
    print(f'Time it took: {time.time() - time_start} seconds \a\n')

    from torch.utils.data import Dataset, DataLoader
    dataset = tokenized_batch
    print(f'{type(dataset)=}')
    print(f'{dataset.__class__=}')
    print(f'{isinstance(dataset, Dataset)=}')
    for i, d in enumerate(dataset):
        assert isinstance(d, dict)
        # dd = dataset[i]
        # assert isinstance(dd, dict)
    loader_opts = {}
    classifier_opts = {} 
    data_loader = DataLoader(dataset, shuffle=False, batch_size=loader_opts.get('batch_size', 1),
                            num_workers=loader_opts.get('num_workers', 0), drop_last=False)
    print(f'{next(iter(data_loader))=}')

    # -- Compute diversity coefficient
    # results: dict = get_diversity_coefficient(dataset, map, probe_network)
    # results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=3)  # only for debugging
    results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=3, verbose=True, debug=True)  # only for debugging
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

def test_interleaved_data_set_2_data_loader():
    """ https://colab.research.google.com/drive/1QWDhA6Q64qijXYnwIGn63Aq9Eg5qt8tQ#scrollTo=Wjyy6QYimvIm """
    remove_columns = []
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
    from datasets import interleave_datasets

    path, name = ['c4', 'wikitext'], ['en', 'wikitext-103-v1']
    probabilities = [1.0/len(path)] * len(path)
    batch_size = 512
    datasets = [load_dataset(path, name, streaming=True, split="train").with_format("torch") for path, name in zip(path, name)]
    [print(f'{dataset.description=}') for dataset in datasets]
    dataset = interleave_datasets(datasets, probabilities)
    print(f'{dataset=}')
    batch = dataset.take(batch_size)
    column_names = next(iter(batch)).keys()
    print(f'{column_names=}')

    # - Prepare functions to tokenize batch
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    remove_columns = column_names
    print(f'{remove_columns=}')
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized_batch = map(batch)
    print(f'{next(iter(tokenized_batch))=}')

    # -- Get data loader
    from torch.utils.data import DataLoader, Dataset

    data_loader = DataLoader(tokenized_batch, shuffle=False, batch_size=8, num_workers=0, drop_last=False)
    batch = next(iter(data_loader))
    print(f'{batch=}')

    # - test 2
    batch = dataset.take(batch_size)
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=[])
    tokenized_batch = map(batch)
    print(f'{next(iter(tokenized_batch))=}')

    def collate_tokenize(data):
        text_batch = [element["text"] for element in data]
        tokenized = tokenizer(text_batch, padding='longest', max_length=128, truncation=True, return_tensors='pt')
        return tokenized
    data_loader = DataLoader(tokenized_batch, shuffle=False, batch_size=8, num_workers=0, drop_last=False, collate_fn=collate_tokenize)
    batch = next(iter(data_loader))
    print(f'{batch=}')

    print('Done!\a')


def cross_div_test():
    """ """
    batch_size = 512

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
    # path, name = 'c4', 'en'
    dataset_target = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    raw_text_batch = dataset.take(batch_size)
    print(f'{next(iter(raw_text_batch))=}')
    # path, name = "wikitext", 'wikitext-103-v1'
    dataset_source = load_dataset("wikitext", "wikitext-103-v1", streaming=True, split="train").with_format("torch")
    raw_text_batch = dataset.take(batch_size)
    print(f'{next(iter(raw_text_batch))=}')
    column_names = next(iter(raw_text_batch)).keys()
    print(f'{column_names=}')

    # - Prepare functions to tokenize batch
    remove_columns = column_names  # remove all text columns so tensors in default collate_fn don't crash
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized_batch = map(raw_text_batch)
    print(f'{next(iter(tokenized_batch))=}')

    # from torch.utils.data import Dataset, DataLoader
    # dataset = tokenized_batch
    # print(f'{type(dataset)=}')
    # print(f'{dataset.__class__=}')
    # print(f'{isinstance(dataset, Dataset)=}')
    # for i, d in enumerate(dataset):
    #     assert isinstance(d, dict)
    #     # dd = dataset[i]
    #     # assert isinstance(dd, dict)
    # loader_opts = {}
    # classifier_opts = {} 
    # data_loader = DataLoader(dataset, shuffle=False, batch_size=loader_opts.get('batch_size', 1),
    #                         num_workers=loader_opts.get('num_workers', 0), drop_last=False)
    # print(f'{next(iter(data_loader))=}')

    # -- Compute diversity coefficient
    results: dict = cross_diversity_coefficient(dataset_target, dataset_source, map, map, probe_network, num_batches=3, verbose=True, debug=True)  # only for debugging
    cross_div_coeff, cross_div_coeff_ci = results['cross_div_coeff'], results['cross_div_coeff_ci']
    print(f'{cross_div_coeff=} {cross_div_coeff_ci=}')

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

def experiment_compute_diveristy_coeff_single_dataset_then_combined_datasets_with_domain_weights():
    """
    Get divs using pt ft, pt (rand, rand ft?) 
    - div c4 
    - div wt = wt-103
    Then with unioned datasets
    - div c4+wt, uniform [0.5, 0.5]
    - # div c4+wt, data set size proportions (using GBs)
    - div c4+wt, respect doremi
    - div c4+wt, respect the pile
    - div c4+wt, respect gpt3 weights
    then repeat all with pt (no ft)
    """
    import random
    from diversity.data_mixtures import get_uniform_data_mixture_for_c4_wt103, get_doremi_based_data_mixture_for_c4_wt103, get_llama_v1_based_data_mixture_for_c4_wt103
    probabilities = []
    data_mixture_name = None
    streaming = True
    data_files = [None]
    seed = 0
    # -- Setup wandb
    import wandb
    # - Dryrun
    # mode = 'dryrun'; num_batches = 3
    mode = 'dryrun'; num_batches = 3; seed = random.randint(0, 2**32 - 1)

    # - Online (real experiment)
    mode='online'; num_batches = 600; seed = random.randint(0, 2**32 - 1)
    # - c4 wt single
    # path, name = 'c4', 'en'
    # path, name = "wikitext", 'wikitext-103-v1'
    # - c4 wt mix
    path, name, data_files = ['c4', 'wikitext'], ['en', 'wikitext-103-v1'], [None, None]
    probabilities, data_mixture_name = get_uniform_data_mixture_for_c4_wt103()
    # probabilities, data_mixture_name = get_doremi_based_data_mixture_for_c4_wt103()
    # probabilities, data_mixture_name = get_llama_v1_based_data_mixture_for_c4_wt103()
    # probabilities, data_mixture_name = [0.75, 0.25], '[0.75, 0.25]' 
    # probabilities, data_mixture_name = [0.25, 0.75], '[0.25, 0.75]' 
    # - pile, pile cc single 
    # path, name = 'EleutherAI/pile', 'all'
    # path, name = 'conceptofmind/pile_cc', 'sep_ds'
    # - 5 subsets of pile using hf data set viewer (parquet)) 
    # from diversity.pile_subset_urls import urls_hacker_news, urls_nih_exporter, urls_pubmed, urls_uspto
    # path, name, data_files = 'conceptofmind/pile_cc', 'sep_ds', [None]
    # path, name, data_files = 'parquet', 'hacker_news', urls_hacker_news
    # path, name, data_files = 'parquet', 'nih_exporter', urls_nih_exporter
    # path, name, data_files = 'parquet', 'pubmed', urls_pubmed
    # path, name, data_files = 'parquet', 'uspto', urls_uspto
    # - 5 subsets of the pile interleaved
    # from diversity.pile_subset_urls import urls_hacker_news, urls_nih_exporter, urls_pubmed, urls_uspto
    # from diversity.data_mixtures import get_uniform_data_mixture_5subsets_of_pile, get_doremi_data_mixture_5subsets_of_pile, get_llama_v1_data_mixtures_5subsets_of_pile
    # path, name, data_files = ['conceptofmind/pile_cc'] + ['parquet'] * 4, ['sep_ds'] + ['hacker_news', 'nih_exporter', 'pubmed', 'uspto'], [None] + [urls_hacker_news, urls_nih_exporter, urls_pubmed, urls_uspto]
    # probabilities, data_mixture_name = get_uniform_data_mixture_5subsets_of_pile()
    # probabilities, data_mixture_name = get_llama_v1_data_mixtures_5subsets_of_pile(name)
    # probabilities, data_mixture_name = get_doremi_data_mixture_5subsets_of_pile(name)
    # - not changing
    batch_size = 512
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    run_name = f'{path} div_coeff_{num_batches=} ({today=} ({name=}) {data_mixture_name=} {probabilities=})'
    print(f'\n---> {run_name=}\n')

    # - Init wandb
    debug: bool = mode == 'dryrun'
    run = wandb.init(mode=mode, project="beyond-scale", name=run_name, save_code=True)
    wandb.config.update({"num_batches": num_batches, "path": path, "name": name, "today": today, 'probabilities': probabilities, 'batch_size': batch_size, 'debug': debug, 'data_mixture_name': data_mixture_name, 'streaming': streaming, 'data_files': data_files, 'seed': seed})
    # run.notify_on_failure() # https://community.wandb.ai/t/how-do-i-set-the-wandb-alert-programatically-for-my-current-run/4891
    print(f'{debug=}')
    print(f'{wandb.config=}')

    # -- Get probe network
    from datasets import load_dataset 
    from datasets.iterable_dataset import IterableDataset
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    probe_network = probe_network.to(device)

    # -- Get data set
    def my_load_dataset(path, name, data_files=data_files):
        print(f'{path=} {name=} {streaming=} {data_files=}')
        if path == 'json' or path == 'bin' or path == 'csv':
            print(f'{data_files_prefix+name=}')
            return load_dataset(path, data_files=data_files_prefix+name, streaming=streaming, split="train").with_format("torch")
        elif path == 'parquet':
            print(f'{data_files=}')
            return load_dataset(path, data_files=data_files, streaming=streaming, split="train").with_format("torch")
        else:
            return load_dataset(path, name, streaming=streaming, split="train").with_format("torch")
    # - get data set for real now
    if isinstance(path, str):
        dataset = my_load_dataset(path, name, data_files)
    else:
        # -Interleaving datasets
        print('- Interleaving datasets')
        datasets = [my_load_dataset(path, name, data_files).with_format("torch") for path, name, data_files in zip(path, name, data_files)]
        # datasets = [my_load_dataset(path, name).with_format("torch") for path, name in zip(path, name)]
        if any('parquet' == p for p in path) or path == 'parquest':  # idk why I need to do this, I checked very carefully and deleted all columns so interleaved data set matched but when doing this with c4 & wikitext it fails but with the parquet it works https://discuss.huggingface.co/t/why-does-deleting-the-columns-before-giving-it-to-interleave-work-but-sometimes-it-does-not-work/50879
            dataset_descriptions = [dataset.description for dataset in datasets]  # print description if available
            print(f'{dataset_descriptions=}')
            # - make sure all datasets have the same columns to avoid interleave to complain
            all_columns = [col for dataset in datasets for col in dataset.column_names]
            print(f'{all_columns=}')
            columns_to_remove = [col for dataset in datasets for col in dataset.column_names if col != 'text']
            columns_to_remove = list(set(columns_to_remove))  # remove duplicates
            print(f'{columns_to_remove=}')
            datasets = [dataset.remove_columns(columns_to_remove) for dataset in datasets]
            # - interleave
            print(f'{probabilities=}')
            dataset_descriptions = [dataset.description for dataset in datasets]  # print description if available
            print(f'{dataset_descriptions=}')
        dataset = interleave_datasets(datasets, probabilities)
        # dataset = dataset.remove_columns(columns_to_remove)
        print(f'{dataset=}')
        print(f'{dataset.column_names=}')
    print(f'{dataset=}')
    print(f'{type(dataset)=}')
    # datasets.iterable_dataset.IterableDataset
    # datasets.arrow_dataset.Dataset
    # dataset = IterableDataset(dataset) if type(dataset) != IterableDataset else dataset  # to force dataset.take(batch_size) to work in non-streaming mode
    raw_text_batch = dataset.take(batch_size) if streaming else dataset.select(range(batch_size))
    print(f'{raw_text_batch=}')
    print(f'{next(iter(raw_text_batch))=}')
    column_names = next(iter(raw_text_batch)).keys()
    print(f'{column_names=}')

    # - Prepare functions to tokenize batch
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    remove_columns = column_names  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized_batch = map(raw_text_batch)
    print(f'{next(iter(tokenized_batch))=}')

    # -- Compute diversity coefficient
    print(f'-- Compute diversity coefficient')
    print(f'{seed=}, {streaming=}')
    # - Debug run
    # results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=3, seed=seed, debug=True, shuffle=False)  # (quick debug) hardcoded for debugging
    # results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=3, seed=seed, debug=True, shuffle=True)  # (slow debug) hardcoded for debugging
    # results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=3, seed=seed, debug=False, shuffle=False)  # (real) hardcoded for debugging
    # - Real run
    # assert not debug, f'Err: {debug=} for real run'
    results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=num_batches, seed=seed, debug=debug, shuffle=False)
    # results: dict = get_diversity_coefficient(dataset, map, probe_network, num_batches=num_batches, seed=seed, debug=debug, shuffle=True)
    # - Log results
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
    # test_interleaved_data_set_2_data_loader()
    experiment_compute_diveristy_coeff_single_dataset_then_combined_datasets_with_domain_weights()
    # -- End tests, report how long it took in seconds, minutes, hours, days
    print(f'Time it took to run {__file__}: {time.time() - time_start} seconds, {(time.time() - time_start)/60} minutes, {(time.time() - time_start)/60/60} hours, {(time.time() - time_start)/60/60/24} days\a')