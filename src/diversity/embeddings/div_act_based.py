"""
Problem:
I want to compute the CCA distance between two LLMs. 
The shape of any (intermediate) layer is [B, T, D], where B is batch size, T is sequence length, and D is the dimensionality of the layer.
The problem is that the CCA distance is only defined for two matrices of shape [B, D], where B is batch size and D is the dimensionality of the layer.
What is the right way to compute the CCA distance between two LLMs?

CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

ref: acts debate/conv: https://chat.openai.com/c/9aae0b31-689e-415c-ba40-73a790bb2e0d
ref: general acts code: https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/d50783d2-f958-49d6-a729-2bc6cf28deb7
"""
import json
import pickle
from pathlib import Path
import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import torch
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from functools import partial

import wandb

import sys

from training.utils import raw_dataset_2_lm_data
print(sys.path)
# sys.path.append('/lfs/ampere9/0/brando9/ultimate-anatome/anatome')
# sys.path.append('/afs/cs.stanford.edu/u/brando9/ultimate-anatome/anatome')

from anatome.similarity import pwcca_distance_choose_best_layer_matrix, svcca_distance, linear_cka_distance, orthogonal_procrustes_distance 
# from anatome.similarity import pwcca_distance_choose_best_layer_matrix, svcca_distance, linear_cka_distance, orthogonal_procrustes_distance, temporal_cca  # darn can't remember how I defined temportal_cca

from diversity.task2vec import Task2Vec

metrics = {'svcca': partial(svcca_distance, accept_rate=0.99, backend='svd'),
           'pwcca': partial(pwcca_distance_choose_best_layer_matrix, backend='svd', epsilon=1e-10),
           'lincka': partial(linear_cka_distance, reduce_bias=False),
            "opd": orthogonal_procrustes_distance,
           }

# ref: https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/afec3f76-c259-42da-84a6-3e1e5790507a
class RandomTokenDataset(Dataset):
    """A custom dataset that generates random tokens and an attention map of all ones compatible with GPT-2."""

    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int, percentange_vocab: float, size: int = sys.maxsize):
        """
        Initialize the dataset.

        Args:
            tokenizer (GPT2Tokenizer): Tokenizer for GPT-2 model.
            size (int): Number of samples in the dataset.
            max_length (int): Maximum length of the token sequence.
        """
        self.tokenizer = tokenizer
        self.size = size
        self.max_length = max_length
        self.percentange_vocab = percentange_vocab
        self.vocab_subset_range = int(self.tokenizer.vocab_size * self.percentange_vocab)  # For example, 10% of the total vocab size
        assert self.vocab_subset_range != 0, "The vocabulary subset range is 0!"

    def __len__(self):
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int):
        """
        Generate a random sample along with an attention map.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: A dictionary with keys 'input_ids' and 'attention_mask'. Both contain a list of token ids.
        """
        random_tokens = [random.randint(0, self.vocab_subset_range - 1) for _ in range(self.max_length)]
        attention_mask = [1] * self.max_length  # Attention mask of all ones
        random_tokens = torch.tensor(random_tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {"input_ids": random_tokens, "attention_mask": attention_mask}

    def take(self, num_samples: int):
        """
        Simulate streaming by generating a specified number of random samples.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            list: A list of randomly generated samples.
        """
        return [self.__getitem__(idx) for idx in range(num_samples)]

    def skip(self, num_samples: int):
        """
        Simulate skipping a specified number of samples.

        Args:
            num_samples (int): The number of samples to skip.

        Returns:
            RandomTokenDataset: The same dataset object, allowing for method chaining.
        """
        # self.skip_samples += num_samples
        return self

def print_all_special_tokens():
    special_tokens = tokenizer.all_special_tokens
    print("Special tokens in the GPT-2 tokenizer:")
    for token in special_tokens:
        print(token)

# Function to set all seeds for reproducibility
def set_random_seeds(seed_value=42):
    """
    This function sets the seed for randomness for reproducible results.
    
    Args:
    - seed_value (int): The value of the seed to be used for all libraries.
    """
    random.seed(seed_value)  # Python's built-in random library
    np.random.seed(seed_value)  # NumPy library
    torch.manual_seed(seed_value)  # PyTorch library

    # If you are using CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # If running on the GPU, also set the seed there
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_tokenizer_with_subset_of_vocab(tokenizer: GPT2Tokenizer, percentage_to_keep: float) -> GPT2Tokenizer:
    """ 
    Create a tokenizer with a fraction of the vocabulary. 

    ref: https://chat.openai.com/c/5539083a-55b9-4a31-a0c6-bce5eeb45e1b     
    """
    from copy import deepcopy
    tok = deepcopy(tokenizer)
    assert id(tok) != id(tokenizer), "The tokenizer is not a deep copy!"
    special_tokens = tok.all_special_tokens
    # to make sure there is always a token set no matter what
    tok.unk_token = "the"  # but "the" is hopefully common enough that it doesn't damage the semantics of the sentence too much, however, putting EOS or something else might screw up the semantics of the sentence

    # Calculate the number of tokens to keep
    total_tokens = len(tok)
    tokens_to_keep_count = int(total_tokens * percentage_to_keep)

    # Get all non-special tokens
    vocab = tok.get_vocab()
    all_tokens = list(vocab.keys())
    non_special_tokens = [token for token in all_tokens if token not in special_tokens]
    assert "the" in non_special_tokens, "The token 'the' is not in the non-special tokens!"

    # Randomly sample from non-special tokens
    random_sampled_tokens = random.sample(non_special_tokens, tokens_to_keep_count - len(special_tokens))

    # Combine special tokens with the randomly sampled tokens
    final_tokens_to_keep = set(special_tokens + random_sampled_tokens + ["the"])
    assert "the" in non_special_tokens, "The token 'the' is not in the non-special tokens!"
    assert tok.unk_token == "the", "The token 'the' is not the unknown token!"

    # Update the tokenizer's vocab
    new_vocab = {token: idx for token, idx in vocab.items() if token in final_tokens_to_keep}
    tok.vocab = new_vocab
    tok.ids_to_tokens = {v: k for k, v in vocab.items()}
    return tok


def generate_same_token_sequence(token_value: int, 
                                sequence_length: int = 50, 
                                batch_size: int = 600, 
                                device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> torch.Tensor:
    """
    Generates a batch of token sequences where each token in the sequence is the same.
    Used only to debug that computing CCA works 

    Args:
    - token_value (int): The token value to be repeated in the sequence.
    - sequence_length (int, optional): The length of each token sequence. Defaults to 50.
    - batch_size (int, optional): The number of sequences in the batch. Defaults to 3.
    - device (torch.device): The device to perform computations on. Defaults to GPU if available, else CPU.

    Returns:
    - torch.Tensor: A tensor containing the batch of identical token sequences on the specified device.
    """
    # Create a single sequence of the same token
    single_sequence = [token_value] * sequence_length

    # Create a batch of identical sequences
    batch_sequences = [single_sequence for _ in range(batch_size)]

    # Convert the batch to a PyTorch tensor and move to the specified device
    token_tensor = torch.tensor(batch_sequences, dtype=torch.long).to(device)
    
    return token_tensor

def generate_semi_random_tokens_batch_limited_vocab(tokenizer: GPT2Tokenizer, 
                           sequence_length: int = 50, 
                           batch_size: int = 600, 
                           device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           percentange_vocab: float = 0.1,
                           ) -> torch.Tensor:
    """
    Generates a batch of semi-random token sequences compatible with GPT-2's tokenizer and moves them to the specified device.
    The randomness is reduced by limiting the selection to a subset of the tokenizer's vocabulary.
    --> [B, L]

    Args:
    - tokenizer (GPT2Tokenizer): The tokenizer for GPT-2.
    - sequence_length (int, optional): The length of each random token sequence. Defaults to 50.
    - batch_size (int, optional): The number of sequences in the batch. Defaults to 1.
    - device (torch.device): The device to perform computations on.

    Returns:
    - torch.Tensor: A tensor containing the batch of semi-random token sequences on the specified device.
    """
    # Define a subset range of the tokenizer's vocabulary
    vocab_subset_range = int(tokenizer.vocab_size * percentange_vocab)  # For example, 10% of the total vocab size
    assert vocab_subset_range != 0, "The vocabulary subset range is 0!"

    # Generate batch of token sequences with tokens randomly selected from the subset range
    batch_random_tokens = [[random.randint(0, vocab_subset_range - 1) for _ in range(sequence_length)] for _ in range(batch_size)]
    
    token_tensor = torch.tensor(batch_random_tokens, dtype=torch.long).to(device)
    assert token_tensor.size() == torch.Size([batch_size, sequence_length]), f'Error" {token_tensor.shape=} not equal to {batch_size, sequence_length}.'
    return token_tensor

def generate_random_tokens(tokenizer: GPT2Tokenizer, 
                           sequence_length: int = 50, 
                           batch_size: int = 600, 
                           device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           ) -> torch.Tensor:
    """
    Generates a batch of random token sequences compatible with GPT-2's tokenizer and moves them to the specified device.

    Args:
    - tokenizer (GPT2Tokenizer): The tokenizer for GPT-2.
    - sequence_length (int, optional): The length of each random token sequence. Defaults to 50.
    - batch_size (int, optional): The number of sequences in the batch. Defaults to 1.
    - device (torch.device): The device to perform computations on.

    Returns:
    - torch.Tensor: A tensor containing the batch of random token sequences on the specified device.
    """
    batch_random_tokens = [[random.randint(0, tokenizer.vocab_size - 1) for _ in range(sequence_length)] for _ in range(batch_size)]
    token_tensor = torch.tensor(batch_random_tokens, dtype=torch.long).to(device)
    return token_tensor

# -- Main

def _test0_does_hacky_fraction_tokenizer_work():
    # - have a tokenizer with only the special token "the", check everything is "the"
    text_seq: str = "the cat is nice"
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    new_tokenizer = get_tokenizer_with_subset_of_vocab(tokenizer, 1/tokenizer.vocab_size) 
    # encode to tokens then decode to text
    tokens = new_tokenizer.encode(text_seq)
    llm_seq_txt: str = new_tokenizer.decode(tokens)
    assert llm_seq_txt == "the the the the", f'Error: {llm_seq_txt=} but should be the the the the'
    # have a tokenizer with only the special token "the" and "cat", check the->the anything_else->the and cat->cat
    text_seq: str = "the cat is nice"
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    new_tokenizer = get_tokenizer_with_subset_of_vocab(tokenizer, 1/tokenizer.vocab_size) 
    # encode to tokens then decode to text
    tokens = new_tokenizer.encode(text_seq)
    llm_seq_txt = new_tokenizer.decode(tokens)
    assert llm_seq_txt == "the cat the the", f'Error: {llm_seq_txt=} but should be the cat the the'

def _test_sanity_check_dist_btw_B1_B2_small_same_large_different():
    """
    Debug distance between single pair of dataset/batches X, Y/B1, B2 of tokens works.
    """
    # Determine if CUDA (GPU support) is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 model and tokenizer
    model = GPT2Model.from_pretrained('gpt2')
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_value1: int = tokenizer.encode('the')[0]
    token_value2: int = tokenizer.encode('at')[0]
    print(f'{tokenizer.model_max_length=}')

    # Generate a random sequence of tokens
    # random_tokens1 = generate_random_tokens(tokenizer).to(device)
    # random_tokens2 = generate_random_tokens(tokenizer).to(device)
    random_tokens1 = generate_same_token_sequence(token_value1).to(device)
    random_tokens2 = generate_same_token_sequence(token_value2).to(device)
    random_tokens1 = generate_semi_random_tokens_batch_limited_vocab(tokenizer).to(device)
    random_tokens2 = generate_semi_random_tokens_batch_limited_vocab(tokenizer).to(device)
    assert random_tokens1.sum().item() != random_tokens2.sum().item(), "Two random sequences of tokens are the same!"
    print(f'{random_tokens1.shape=}')
    print(f'{random_tokens2.shape=}')
    print(f'{random_tokens1.sum()=}')
    print(f'{random_tokens2.sum()=}')

    # Compute the activations from the model
    activations1 = model(random_tokens1)
    activations2 = model(random_tokens2)
    # Extract the activations tensor
    activations1 = activations1.last_hidden_state
    activations2 = activations2.last_hidden_state
    # Reshape the activations tensor to the shape [B, T*D] BAD
    # activations1 = activations1.view(activations1.size(0), -1)
    # activations2 = activations2.view(activations2.size(0), -1)
    # Reshape the activations tensor to the shape [B*T, D]  # BETTER
    activations1 = activations1.view(-1, activations1.size(-1))
    activations2 = activations2.view(-1, activations2.size(-1))

    # Print the shape of the activations tensor
    print(f"Shape of activations tensor: {activations1.shape}")
    print(f"Shape of activations tensor: {activations2.shape}")
    print(f'{activations1.sum()=}')
    print(f'{activations2.sum()=}')

    dist: torch.Tensor = svcca_distance(activations1, activations2, accept_rate=0.99, backend='svd')
    # dist: torch.Tensor = pwcca_distance_choose_best_layer_matrix(activations, activations, backend='svd', epsilon=1e-10)
    # dist, dists = temporal_cca(activations1, activations2)
    print(f'Dist btw single pair of data sets/batches should be large (different data sets): {dist=}')
    
    dist: torch.Tensor = svcca_distance(activations1, activations1, accept_rate=0.99, backend='svd')
    # dist: torch.Tensor = pwcca_distance_choose_best_layer_matrix(activations, activations, backend='svd', epsilon=1e-10)
    # dist, dists = temporal_cca(activations1, activations2)
    print(f'Dist btw single pair of data sets/batches should be small becuase (same data sets): {dist=}')

def main2_percent_vs_avg_dist():
    """
    Main function to plot the relationship between percentage of vocabulary used in token generation
    and the average CCA distance between two sets of activations from a GPT-2 model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 model and tokenizer
    model = GPT2Model.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    percentages = np.linspace(0.05, 1.0, 30)  # Range of percentages from 0.1 to 1.0
    avg_distances = []

    with torch.no_grad():
        for i, percentage in tqdm(enumerate(percentages)):
            print(f'{i=} percentage = {percentage}')
            torch.cuda.empty_cache()
            # Generate token sequences with the given percentage of the vocabulary
            random_tokens1 = generate_semi_random_tokens_batch_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
            random_tokens2 = generate_semi_random_tokens_batch_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
            torch.cuda.empty_cache()
            # Compute the activations from the model
            activations1 = model(random_tokens1)[0]
            activations2 = model(random_tokens2)[0]

            # Compute the activations
            # activations1 = activations1.view(random_tokens1.size(0), -1)
            # activations2 = activations2.view(random_tokens2.size(0), -1)
            # Reshape the activations tensor to the shape [B*T, D]
            activations1 = activations1.view(-1, activations1.size(-1))
            activations2 = activations2.view(-1, activations2.size(-1))
            torch.cuda.empty_cache()
            print(f'{activations1.shape=} {activations2.shape=}')

            # Calculate CCA distance
            # dist = svcca_distance(activations1, activations2)
            # dist = linear_cka_distance(activations1, activations2)
            dist = orthogonal_procrustes_distance(activations1, activations2)
            torch.cuda.empty_cache()
            div = dist.mean().item()
            print(f'{div=}')
            avg_distances.append(div)
            torch.cuda.empty_cache()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, avg_distances, marker='o')
    plt.xlabel('Percentage of Vocabulary Used')
    plt.ylabel('Average CCA Distance')
    plt.title('Average CCA Distance vs. Vocabulary Usage Percentage')
    plt.grid(True)
    plt.show()
    # save plot as .png file to ~/beyond-scale-language-data-diversity
    plt.savefig(os.path.expanduser('~/beyond-scale-language-data-diversity/avg_cca_dist_vs_vocab_usage.png'))

# def main3_percent_vs_avg_dist_with_cis():
#     """
#     Main function to plot the relationship between percentage of vocabulary used in token generation
#     and the average CCA distance between two sets of activations from a GPT-2 model,
#     including 95% confidence intervals.

#     Note:
#         you can improve current code & speed up by:
#         1. computing a single set of activations for each batch from a data set, so list O(num_batches)
#         2. then for each pair of batches compute their distance and store it in a list O(num_batches^2 - num_batches) [minus diagonal same batch]
#         3. then compute the average and std of the distances of this O(num_batches^2 - num_batches) list
#     """
#     # set random seed
#     seed = 0
#     set_random_seeds(seed)

#     # Load the GPT-2 model and tokenizer
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = GPT2Model.from_pretrained('gpt2').to(device)
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     print(f'{tokenizer.vocab_size=}')

#     num_batches:int = 30
#     metric: str = 'svcca'
#     metric: str = 'pwcca'
#     metric: str = 'lincka'
#     # metric: str = 'opd'
#     metric: str = 'Task2Vec'
#     # metric: str = 'token_dist_entropy'
#     start=1.0/tokenizer.vocab_size
#     stop=1.0
#     num_percentages=30
#     percentages = list(np.linspace(start, stop, num_percentages))  # Range of percentages from 0.05 to 1.0
#     # percentages = np.linspace(1.0/tokenizer.vocab_size, 0.02, 60)  # Range of percentages from 0.05 to 1.0
#     # percentages = np.linspace(1.0/tokenizer.vocab_size, 0.001, 60)  # Range of percentages from 0.05 to 1.0
#     print(f'{percentages=}')
#     print(f'x-axis (vocab) linspace range: {start=} {stop=} {num_percentages=} {metric=} {num_batches=}')
#     avg_dists_per_data_set = []  # [avg(dists1), avg(dists1, ...] = [div1, div2, ...]
#     std_per_data_set = []  # [std(dists1), std(dists2), ...]
#     ci_per_data_set = []  # [ci(dist1)), ci(dists2), ...]
#     dist_func = metrics[metric]
#     embeddings = []
#     losses = []
#     # for each percentage vocab ~ for each data set with different diversity
#     for i, percentage in tqdm(enumerate(percentages)):
#         print(f'{i=} percentage = {percentage}')
#         # given a specific percentage vocab/data set diversity, compute average distance between batches
#         dist_current_data_set = []
#         current_embedding_pair = []
#         current_loss_pair = []
#         for batch_idx in range(num_batches):
#             print(f'{batch_idx=}')
#             torch.cuda.empty_cache()
#             # D1, D2 ~ p(Di | taui),
#             # raw batch [B, L]
#             tokens1 = generate_semi_random_tokens_batch_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
#             tokens2 = generate_semi_random_tokens_batch_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
#             if metric != 'Task2Vec':
#                 # act batch [B, L, D]
#                 with torch.no_grad():
#                     activations1 = model(tokens1).last_hidden_state
#                     activations2 = model(tokens2).last_hidden_state
#                     print(f'{activations1.shape=} {activations2.shape=}')

#                     activations1 = activations1.view(-1, activations1.size(-1))
#                     activations2 = activations2.view(-1, activations2.size(-1))
#                     print(f'{activations1.shape=} {activations2.shape=}')
#                     print(f'{activations1.shape[0]/activations1.shape[1]=} (curse low div suggest at least 10 i.e., B/D >= 10)')

#                     dist = dist_func(activations1, activations2)
#             else:
#                 # package [B, L] pytorch data set object into mini data sets

#                 # Task2Vec
#                 embedding1, loss1 = Task2Vec(model, classifier_opts={'seed': seed}).embed(batch)
#                 embedding2, loss2 = Task2Vec(model, classifier_opts={'seed': seed}).embed(batch)
#                 current_embedding_pair.append((embedding1, embedding2))
#                 current_loss_pair.append((loss1, loss2))
#                 from diversity.task_similarity import _DISTANCES
#                 distance_fn = _DISTANCES['cosine']
#                 dist = distance_fn(embedding1, embedding2)
#             dist = float(dist.view(-1).cpu().numpy())
#             print(f'{dist=}')
#             dist_current_data_set.append(dist)
#         # compute avg, std, ci for current data set
#         avg_dist = np.mean(dist_current_data_set)
#         std_dist = np.std(dist_current_data_set)
#         n_samples = len(dist_current_data_set)
#         ci = 1.96 * (std_dist / np.sqrt(n_samples))
#         div = avg_dist
#         print(f'Data set {percentage=}: avg_dist={div=} +- {ci} ')
#         print(f'Data set {percentage=}: N[dist | {avg_dist=} {std_dist=}]')
#         # TODO: compute distance to a standard normal distribution
#         avg_dists_per_data_set.append(avg_dist)
#         std_per_data_set.append(std_dist)
#         ci_per_data_set.append(ci)
#         # for current data set pair store the embeddings of the data set and the losses
#         embeddings.append(current_embedding_pair)
#         losses.append(current_loss_pair)
#     # Plotting the results with 95% CI
#     print(f'{percentages=}')
#     print(f'{avg_dists_per_data_set=}')
#     print(f'{ci_per_data_set=}')
#     print(f'{std_per_data_set=}')
#     plt.figure(figsize=(10, 6))
#     # plt.plot(percentages, avg_distances, marker='o')
#     plt.errorbar(percentages, avg_dists_per_data_set, yerr=ci_per_data_set, fmt='-o', ecolor='lightgray', capsize=5)
#     plt.xlabel('Percentage of Vocabulary Used')
#     plt.ylabel(f'Average {metric} Distance')
#     # plt.title('Average CCA Distance vs. Vocabulary Usage Percentage with 95% CI')
#     plt.title(f'Average {metric} Distance vs. Vocabulary Usage Percentage')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(os.path.expanduser(f'~/beyond-scale-language-data-diversity/avg_{metric}_dist_vs_vocab_usage_with_ci_start_{start:.2f}_stop_{stop:.2f}_num_{num}_num_batches_{num_batches}.png'))
#     # save 4 lists to json
#     import json
#     with open(os.path.expanduser(f'~/beyond-scale-language-data-diversity/avg_{metric}_dist_vs_vocab_usage_with_ci_start_{start:.2f}_stop_{stop:.2f}_num_{num}_num_batches_{num_batches}.json'), 'w') as f:
#         data = {'percentages': percentages, 'avg_dists_per_data_set': avg_dists_per_data_set, 'ci_per_data_set': ci_per_data_set, 'std_per_data_set': std_per_data_set}
#         print(f'{data=}')
#         json.dump(data, f)
#         # json.dump({'percentages': percentages, 'avg_dists_per_data_set': avg_dists_per_data_set, 'ci_per_data_set': ci_per_data_set, 'std_per_data_set': std_per_data_set}, f)
#     print(f'x-axis (vocab) linspace range: {start=} {stop=} {num_percentages=} {metric=} {num_batches=}')

def main4_real_hf_percent_vocab_vs_avg_dist_with_cis():
    epochs_task2_vec = None
    # - Dryrun
    mode = 'dryrun'; seed = 0
    mode = 'online'; seed = 0

    # set random seed
    seed = 0
    set_random_seeds(seed)
    
    # metric
    metric: str = 'svcca'
    metric: str = 'pwcca'
    metric: str = 'lincka'
    # metric: str = 'opd'
    metric: str = 'Task2Vec'
    # epochs_task2_vec: int = 5
    epochs_task2_vec: int = 0
    # metric: str = 'token_dist_entropy'
    print(f'--> {metric=} {epochs_task2_vec=}')
    
    # Load the GPT-2 model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2').to(device)
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device) if metric == 'Task2Vec' else model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Run main hps
    # debug hps
    start=0.9
    num_batches:int = 2
    batch_size = 2
    num_percentages=2
    block_size = 2
    
    # real hps
    num_batches:int = 30
    num_batches:int = 300
    batch_size = 32
    num_percentages=100
    block_size = 240
    ## block_size = tokenizer.model_max_length
    print(f'--> {num_batches=} {batch_size=} {num_percentages=} {block_size=}')
    print(f'--> tot_iterations={num_batches*num_percentages=}')
    predicted_safety_margin = batch_size * block_size / config.n_embd
    print(f'--> {predicted_safety_margin=}')
    
    # Get hf data set
    # path, name, split = "c4", "en", "train"
    # dataset = load_dataset(path=path, name=name, split=split, streaming=True)
    path, name, split = "Random Uniform", 'Random Uniform', 'Random Uniform'

    start=2.0/tokenizer.vocab_size
    # stop=1.0
    stop=0.4
    # stop=0.5
    percentages = list(np.linspace(start, stop, num_percentages))  # Range of percentages from 0.05 to 1.0
    print(f'{percentages=}')
    print(f'x-axis (vocab) linspace range: {start=} {stop=} {num_percentages=} {metric=} {num_batches=}')
    avg_dists_per_data_set = []  # [avg(dists1), avg(dists1, ...] = [div1, div2, ...]
    std_per_data_set = []  # [std(dists1), std(dists2), ...]
    ci_per_data_set = []  # [ci(dist1)), ci(dists2), ...]
    dist_func = metrics.get(metric, None)
    embeddings = []
    losses = []
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    current_tmux_session = os.environ.get("TMUX", "").split(",")[-1]
    run_name = f"beyond-scale: {today} {metric} {start=:.2f} {stop=:.2f} {num_percentages=} {num_batches=} {block_size=} {batch_size=} {name=} {path=} {split=} {seed=} {device=} {CUDA_VISIBLE_DEVICES=} {current_tmux_session=} {epochs_task2_vec=}"
    run = wandb.init(mode=mode, project="beyond-scale", name=run_name, save_code=True)
    wandb.config.update({'metric': metric, 'start': start, 'stop': stop, 'num_percentages': num_percentages, 'num_batches': num_batches, 'block_size': block_size, 'batch_size': batch_size, 'name': name, 'path': path, 'split': split, 'seed': seed, 'device': device, 'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES, 'current_tmux_session': current_tmux_session, 'epochs_task2_vec': epochs_task2_vec})
    print(f'{run.url=}')
    # for each percentage vocab ~ for each data set with different diversity
    for i, percentage in tqdm(enumerate(percentages), total=len(percentages)):
        print(f'{i=} percentage = {percentage}')
        if path == 'c4':
            # TODO: fix idk if this is why it's not working https://stackoverflow.com/questions/77917717/how-does-one-create-a-hf-tokenizer-with-only-a-fraction-of-the-vocabulary-but-wi
            # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # tokenizer = get_tokenizer_with_subset_of_vocab(tokenizer, percentage)
            print(f'{tokenizer.vocab_size=}')
            lm_dataset = raw_dataset_2_lm_data(dataset, tokenizer, block_size)  # can't use for random tokens because it expects a text field but random tokens are just a tensor of randints
        elif path == 'Random Uniform':
            lm_dataset = RandomTokenDataset(tokenizer, max_length=block_size, percentange_vocab=percentage)
        else:
            raise ValueError(f'Err: {name=}')
        dataloader = iter(DataLoader(lm_dataset, batch_size=batch_size))
        # given a specific percentage vocab/data set diversity, compute average distance between batches
        dist_current_data_set = []
        current_embedding_pair = []
        current_loss_pair = []
        for batch_idx in range(num_batches):
            # print(f'{batch_idx=}')
            torch.cuda.empty_cache()
            device = next(model.parameters()).device
            if metric != 'Task2Vec':
                # D1, D2 ~ p(Di | taui),
                # raw batch [B, L]
                tokens1: dict = next(dataloader)
                tokens2: dict = next(dataloader)
                input_ids1, attention_mask1 = tokens1['input_ids'].to(device), tokens1['attention_mask'].to(device)
                input_ids2, attention_mask2 = tokens2['input_ids'].to(device), tokens2['attention_mask'].to(device)
                # assert input_ids1.sum().item() != input_ids2.sum().item() and percentage != start, "Batch of sequences of tokens are the same!"
                assert input_ids1.sum().item() != input_ids2.sum().item(), "Batch of sequences of tokens are the same!"
                # act batch [B, L, D]
                with torch.no_grad():
                    activations1 = model(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state
                    activations2 = model(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state
                    # print(f'{activations1.shape=} \n{activations2.shape=}')

                    activations1 = activations1.view(-1, activations1.size(-1))
                    activations2 = activations2.view(-1, activations2.size(-1))
                    # print(f'{activations1.shape=} \n{activations2.shape=}')
                    # print(f'--> {activations1.shape[0]/activations1.shape[1]=} {predicted_safety_margin=} (curse low div suggest at least 10 i.e., B/D >= 10)')
                    # print(f'--> {activations2.shape[0]/activations2.shape[1]=} {predicted_safety_margin=} (curse low div suggest at least 10 i.e., B/D >= 10)')

                    dist = dist_func(activations1, activations2)
                    dist: float = float(dist.view(-1).cpu().numpy())
            else:
                # Task2Vec
                # ds = dataset.shuffle(buffer_size=500_000, seed=seed) 
                ds = lm_dataset
                batch1 = ds.take(batch_size)
                batch2 = ds.skip(batch_size).take(batch_size)
                # assert list(batch1)[0]['text'] != list(batch2)[0]['text'], f'Err: Batch of seq of tokens are the same! {batch1["text"]=} {batch2["text"]=}'
                # assert list(batch1)[0]['input_ids'].sum() != list(batch2)[0]['input_ids'].sum(), f'DErr: Batch of seq of tokens are the same! {batch1["input_ids"]=} {batch2["input_ids"]=}'
                embedding1, loss1 = Task2Vec(model, classifier_opts={'seed': seed}).embed(batch1, epochs=epochs_task2_vec)
                embedding2, loss2 = Task2Vec(model, classifier_opts={'seed': seed}).embed(batch2, epochs=epochs_task2_vec)
                current_embedding_pair.append((embedding1, embedding2))
                current_loss_pair.append((loss1, loss2))
                from diversity.task_similarity import _DISTANCES
                distance_fn = _DISTANCES['cosine']
                dist: float = float(distance_fn(embedding1, embedding2))
            dist: float = float(dist)
            # print(f'{dist=}')
            dist_current_data_set.append(dist)
        # compute avg, std, ci for current data set
        avg_dist = np.mean(dist_current_data_set)
        std_dist = np.std(dist_current_data_set)
        n_samples = len(dist_current_data_set)
        ci = 1.96 * (std_dist / np.sqrt(n_samples))
        div = avg_dist
        print(f'Data set {percentage=}: avg_dist={div=} +- {ci}')
        print(f'Data set {percentage=}: N[dist | {avg_dist=} {std_dist=}]')
        if mode == 'online':
            wandb.log({'avg_dist': avg_dist, 'std_dist': std_dist, 'ci': ci, 'percentage': percentage})
        # TODO: compute distance to a standard normal distribution
        avg_dists_per_data_set.append(avg_dist)
        std_per_data_set.append(std_dist)
        ci_per_data_set.append(ci)
        # for current data set pair store the embeddings of the data set and the losses
        embeddings.append(current_embedding_pair)
        losses.append(current_loss_pair)
    # Plotting the results with 95% CI
    print(f'{percentages=}')
    print(f'{avg_dists_per_data_set=}')
    print(f'{ci_per_data_set=}')
    print(f'{std_per_data_set=}')
    plt.figure(figsize=(10, 6))
    # plt.plot(percentages, avg_distances, marker='o')
    plt.errorbar(percentages, avg_dists_per_data_set, yerr=ci_per_data_set, ecolor='gray', fmt='-o', capsize=5)
    plt.xlabel('Percentage of Vocabulary Used')
    plt.ylabel(f'Average {metric} Distance (Diversity)')
    # plt.title('Average CCA Distance vs. Vocabulary Usage Percentage with 95% CI')
    plt.title(f'Average {metric} Distance (Diversity) vs. Vocabulary Usage Percentage')
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.expanduser(f'~/beyond-scale-language-data-diversity/avg_{metric}_dist_vs_vocab_usage_with_ci_start_{start:.2f}_stop_{stop:.2f}_num_percentages_{num_percentages}_num_batches_{num_batches}_{path}.png'))
    if mode == 'online':
        print(f'{run.url=}')
        # save figure in wandb asap before a potential error in saving file occurs e.g., disk quota
        wandb.log({f"Diversity of {path} as Vocabulary Varies": wandb.Image(plt)})
        run.finish()
    # save 4 lists to json
    path = path.replace(' ', '_')
    output_dir = Path(f'~/data/beyond_scale/div_acts_vs_task2vec_vs_tokens/').expanduser() 
    output_dir.mkdir(parents=True, exist_ok=True)
    filename: str = f'avg_{metric}_dist_vs_vocab_usage_with_ci_start_{start:.2f}_stop_{stop:.2f}_num_percentages_{num_percentages}_num_batches_{num_batches}_{path}'
    with open(output_dir / f'{filename}.json', 'w') as f:
        data = {'percentages': percentages, 'avg_dists_per_data_set': avg_dists_per_data_set, 'ci_per_data_set': ci_per_data_set, 'std_per_data_set': std_per_data_set, 'path': path, 'start': start, 'stop': stop}
        print(f'{data=}')
        json.dump(data, f)
    if metric == 'Task2Vec':
        # pickle embeddings and losses & data dict
        with open(output_dir / f'{filename}_embeddings_losses.pkl', 'w') as f:
            pickle.dump({'embeddings': embeddings, 'losses': losses, 'data': data}, f)
    print(f'x-axis (vocab) linspace range: {start=} {stop=} {num_percentages=} {metric=} {num_batches=}') 

if __name__ == '__main__':
    import time
    start = time.time()
    # _test_sanity_check_dist_btw_B1_B2_small_same_large_different()
    # _test0_does_hacky_fraction_tokenizer_work()
    # main2_percent_vs_avg_dist()
    # main3_percent_vs_avg_dist_with_cis()
    main4_real_hf_percent_vocab_vs_avg_dist_with_cis()
    # print secs, mins, hours elapste one line
    print(f'Done!\a Time elapsed: {(time.time() - start):.2f}secs {((time.time() - start)/60):.2f}mins {((time.time() - start)/60/60):.2f}hours\a\a')