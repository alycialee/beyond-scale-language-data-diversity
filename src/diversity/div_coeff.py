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


def diversity_coefficient(dataset,
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
    Compute the diversity coefficient of a dataset using a probe network.
    Return all results in a dictionary since it's often useful to store them to avoid recomputing them.
    If you want the diveristy coefficient and it's confidence interval (ci), use the following:
        div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    """
    # - Compute embeddings
    embeddings = []
    losses = []
    for batch_num in range(num_batches):
        shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        tokenized_batch = get_mapped_batch_fn()
        embedding, loss = Task2Vec(probe_network).embed(tokenized_batch)
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

# -- Tests, Examples

def test_diversity_coefficient():
    save_results = False
    # -- Get probe network
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    print(f'{type(probe_network)}=')
    print(f'{type(tokenizer)}=')

    # -- Get data set
    dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    print(f'{type(dataset)}=') 
    remove_columns = ["text", "timestamp", "url"]
    def preprocess(examples):
        tokenized_examples = tokenizer(examples["text"], return_tensors="pt")
        return tokenized_examples 
    def get_mapped_batch(dataset, preprocess=preprocess, batched=True, remove_columns=remove_columns):
        batch = dataset.map(preprocess, batched=True, remove_columns=remove_columns)
        return batch 
    
    # -- Compute diversity coefficient
    results = diversity_coefficient(dataset, get_mapped_batch, probe_network)
    div_coeff, div_coeff_ci = results['div_coeff'], results['div_coeff_ci']
    print(f'{div_coeff=} {div_coeff_ci=}')
    if save_results:
        output_dir = Path('./').expanduser()
        np.save(output_dir / 'distance_matrix.npy', results['distance_matrix'])
        np.save(output_dir / 'results.npy', results)


if __name__ == '__main__':
    # run tests and time it
    import time
    time_start = time.time()
    # run tests
    test_diversity_coefficient()
    print(f'Time it took: {time.time() - time_start} seconds')