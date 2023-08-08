"""
todo: wish list but not essential. I wish I could have passed the data set as a streaming data set, to avoid tokenizing all the 
data set before doing anything with it. e.g., the data sets might be huge, especially for training/fine-tuning. 
GPT4 suggested this: https://chat.openai.com/share/495de296-71c2-4f5e-83e2-3b22d038e8bc which seems reasonable.
It also would have made all the data set interfaces consistent in training vs computing data set metrics.
"""
from pathlib import Path

from torch import nn
import numpy as np

from datasets import load_dataset

from diversity.task2vec import Task2Vec
import diversity.task_similarity as task_similarity

def get_diversity_coefficient(dataset,
                          map_batch: callable,  # to ease whatever ars you want to batch.map for any data set
                          probe_network: nn.Module,
                          tokenizer = None,
                          batch_size: int = 512,
                          num_batches: int = 100, 
                          seed = 0, 
                          buffer_size: int = 500_000, 
                          distance = 'cosine',
                          verbose: bool = True,
                          ) -> dict:
    """
    Compute the diversity coefficient of a dataset using a probe network.
    Return all results in a dictionary since it's often useful to store them to avoid recomputing them.
    If you want the diveristy coefficient and it's confidence interval (ci), use the following:
        div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    """
    # - Compute embeddings
    embeddings = []
    losses = []
    for batch_num in range(num_batches):
        # - Get batch
        shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        raw_text_batch = dataset.take(batch_size)
        tokenized_batch = map_batch(raw_text_batch)
        print(f'{type(tokenized_batch)=}')
        print(f'{next(iter(tokenized_batch))=}')

        # - Get Task2Vec embedding for batch
        batch_embedding, batch_loss = Task2Vec(probe_network).embed(tokenized_batch)
        if verbose:
            print(f'{batch_num=}, {batch_loss=}')
            print(f'{batch_embedding=}')
        
        # - Collect results
        embeddings.append(batch_embedding)
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

# -- Tests, Examples

def test_get_batch_from_dataset():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    max_seq_length = 128
    batch_size = 512
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token

    # -- Get batch from dataset
    dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    remove_columns = ["text", "timestamp", "url"]
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=max_seq_length, truncation=True, return_tensors="pt")
    raw_text_batch = dataset.take(batch_size)
    print(f'{batch=}')
    print(f'{next(iter(batch))=}')
    tokenized_batch = batch.map(preprocess, batched=True, remove_columns=remove_columns)
    print(f'{tokenized_batch=}')
    print(f'{next(iter(tokenized_batch))=}')
    # Running /lfs/ampere1/0/brando9/beyond-scale-language-data-diversity/src/diversity/div_coeff.py
    # batch=<datasets.iterable_dataset.IterableDataset object at 0x7ff29031fd60>
    # tokenized_batch=<datasets.iterable_dataset.IterableDataset object at 0x7ff2901889d0>
    # Time it took: 0.9143445491790771 seconds

def test_diversity_coefficient():
    # -- Get probe network
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'{tokenizer.pad_token=}')
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    print(f'{type(probe_network)}=')
    print(f'{type(tokenizer)}=')

    # -- Get data set
    dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    padding="max_length"; max_seq_length=128; trucation=True; return_tensors="pt"
    def preprocess(examples, padding=padding, max_seq_length=max_seq_length, trucation=trucation, return_tensors=return_tensors):
        return tokenizer(examples["text"], padding=padding, max_length=max_seq_length, trucation=trucation, return_tensors=return_tensors)
    batched=True; remove_columns = ["text", "timestamp", "url"]
    def map_batch(batch, batched=batched, remove_columns=remove_columns):
        return batch.map(preprocess, batched=batched, remove_columns=remove_columns)
    
    # -- Compute diversity coefficient
    results = get_diversity_coefficient(dataset, map_batch, probe_network)
    div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    print(f'{div_coeff=} {div_coeff_ci=}')

    # -- Save results or not
    # save_results = False
    # if save_results:
    #     output_dir = Path('./').expanduser()
    #     np.save(output_dir / 'distance_matrix.npy', results['distance_matrix'])
    #     np.save(output_dir / 'results.npy', results)

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
    pass


if __name__ == '__main__':
    print(f'\nRunning {__file__}')
    # -- Run tests and time it
    import time
    time_start = time.time()
    # -- Run tests
    # test_get_batch_from_dataset()
    test_diversity_coefficient()
    # -- End tests, report how long it took
    print(f'Time it took: {time.time() - time_start} seconds')