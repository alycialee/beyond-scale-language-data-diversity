"""
Original descriptions from paper:

2.3. Recipe for Establishing if a Diversity Coefficient is
High via the Conceptual Lower and Upper Bounds
To establish if a diversity coefficient div(D) of a dataset D
is high (or low), we use two conceptually well-motivated
reference values. We call them the lower and upper bounds
of the diversity coefficient. There, we explain the conceptually motivated lower and upper bounds of the diversity
coefficient. Consider a dataset constructed by sampling with
most of the probability mass concentrated on some arbitrary
token. This is a good candidate for a dataset with minimum
diversity. On the other extreme, a dataset constructed by
sampling any token uniformly at random given a fixed vocabulary (in our case, the GPT-2 tokenizer vocabulary) is a
good candidate to create a dataset with maximum diversity.
Therefore, we measure a conceptual lower bound on a
dataset with a vocabulary size of 2: <eos> token and a
randomly selected non-special token from the GPT-2 tokenizer vocabulary. The <eos> token was assigned a probability weight of 1/{GPT-2 vocab size}. The non-special
token was assigned the remaining weight. Similarly, a high
or maximum diversity dataset would consist of random sequences of all possible tokens, with no underlying order
to semantics, formatting, etc. The upper bound of the diversity coefficient was therefore measured on a synthetic
dataset with an equal probability of occurrence assigned to
all tokens in the GPT-2 tokenizer vocabulary.

refs: 
  - colab: https://colab.research.google.com/drive/1YHMSnvevy23FJ80hGXcPpD0l0_LlXfxY#scrollTo=sjgFdf-ls8rp
  - https://claude.ai/chat/f53bcb39-2c54-4e02-a831-87c6cf7f0d80, https://claude.ai/chat/7bc1c10d-ee24-4add-b8a8-6047408e0c5b
  - https://chat.openai.com/c/56296331-190f-4572-868c-12d510d19c69
  - https://github.com/brando90/beyond-scale-language-data-diversity/blob/main/src/diversity/_lower_upper_div_bounds.py
"""
buffer_size: int = 500_000

import random
import torch
import sys  
from datasets import Dataset

# Generate a single sample/sequence

def gen_lb_seq(tokenizer, max_length: int = 128):
  """ 
  Algorithm to generate lower bound div data set:
    - at each step generate special token or eos, eos with small prob, 1/|V| eos, other 1 - 1/|V| the none eos token
    - once eos is generate, use pad token in tokenizer to generate seq up to length max_length
  Alycia
    - So once the eos token was predicted (with small probability 1/vocabulary size) then the rest of tokens would be padding tokens up to the max seq length

  return: type ~ seq/sample ~ list/torch of lb token ids (eos with small prob, 1/|V| eos, other 1 - 1/|V| the none eos token, then padded to max len)
  """
  # ~ return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
  # Get EOS token 
  eos_token_id = tokenizer.eos_token_id

  # Generate a token id that is not -- EOS (since we want to have lb seq to be either the non token id until eos is sampled, then padding to max length
  # Get vocabulary size
  vocab_size = len(tokenizer)
  # Generate random int between 0 and vocab size
  random_token_id = random.randint(0, vocab_size - 1)
  # If random ID is EOS, regenerate until it is not eos
  while random_token_id == eos_token_id:
    random_token_id = random.randint(0, vocab_size - 1)

  # Gen single lb sample/seq - generate random_token_id (none eos) or eos with corresponding probs 1/|V| eos, other 1 - 1/|V| the none eos token
  p_eos = 1/vocab_size  # Probability of EOS (small prob for eos)
  # Generate sequence
  input_ids = []
  for i in range(max_length-1):
    # First token is always non-EOS
    if i == 0:
      input_ids.append(random_token_id)
    # Later tokens are EOS with prob p_eos
    else:
      # generate a random number if it's < p_eos then generate p_eos
      if random.random() < p_eos:
        input_ids.append(eos_token_id)
        break
      else:
        input_ids.append(random_token_id)

  # Get final lb seq - Pad sequence to max length
  num_pads = max_length - len(input_ids)
  input_ids.extend([tokenizer.pad_token_id] * num_pads)
  # return {"input_ids": input_ids}
  input_ids = torch.tensor(input_ids)
  assert not isinstance(input_ids, list)
  return input_ids

def gen_ub_seq(tokenizer, max_length: int = 128):
  """
  Algorithm to generate lower bound div data set:
    - generate either eos or any eos token token with equal prob
    - once eos is generate, pad the remaining seq to max length with eos's pad token
  """
  # ~ return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
  # Get EOS token 
  eos_token_id = tokenizer.eos_token_id

  # Generate a token id that is not -- EOS (since we want to have lb seq to be either the non token id until eos is sampled, then padding to max length
  # Get vocabulary size
  vocab_size = len(tokenizer)
  # Generate random int between 0 and vocab size
  random_token_id = random.randint(0, vocab_size - 1)
  # If random ID is EOS, regenerate until it is not eos
  while random_token_id == eos_token_id:
    random_token_id = random.randint(0, vocab_size - 1)

  # Generate ub seq - sequence sample any token uniformly, once eos is gen, pad
  input_ids = []
  for i in range(max_length-1):
    # First token is always non-EOS
    if i == 0:
      input_ids.append(random_token_id)
    # Sample any token ID from 0 to vocab_size-1  # this is the uniform sampling 0.5, 0.5 any token
    token_id = random.randint(0, vocab_size-1)
    if token_id == eos_token_id:
      # If EOS, end sequence generation
      input_ids.append(token_id)
      break
    else:
      input_ids.append(token_id)  

  # Get final ub seq - Pad sequence to max length
  num_pads = max_length - len(input_ids)
  input_ids.extend([tokenizer.pad_token_id] * num_pads)
  # return torch.tensor(input_ids)
  # return {"input_ids": input_ids}
  input_ids = torch.tensor(input_ids)
  assert not isinstance(input_ids, list)
  return input_ids

def get_lb_ds(tokenizer, num_sequences: int = 307200, max_length: int = 128):
  samples: list[list] = [gen_lb_seq(tokenizer) for i in range(num_sequences)]  # generate sequenes/samples for lb/ub data set, list of sequences/samples
  dataset = Dataset.from_dict({"input_ids": samples})  # converts the only elements in the sequences into rows in the hf dataset with column name input_ids
  if verbose:
    print(f'{dataset=}')
    print(f'{type(dataset)=}')
  return dataset

def get_ub_ds(tokenizer, num_sequences: int = 307200, max_length: int = 128, verbose: bool = False):
  samples: list[list] = [gen_ub_seq(tokenizer) for i in range(num_sequences)]  # generate sequenes/samples for lb/ub data set, list of sequences/samples
  dataset = Dataset.from_dict({"input_ids": samples})  # converts the only elements in the sequences into rows in the hf dataset with column name input_ids
  if verbose:
    print(f'{dataset=}')
    print(f'{type(dataset)=}')
  return dataset

# -- Test, examples, etc.

def test_lb_ds_looping_with_div_coeff_map_code():
  import torch
  from datasets import Dataset
  from transformers import AutoTokenizer
  batch_size = 512
  num_batches = 600
  streaming = False
  seed = 0
  shuffle = False

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or other tokenizer
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

  # Generate dataset with num_batches * batch_size = 1000 samples/sequences
  num_sequences = num_batches * batch_size  # total number of samples/sequences for the lb/ub data set
  
  samples: list[list] = [gen_lb_seq(tokenizer) for i in range(num_sequences)]  # generate sequenes/samples for lb/ub data set, list of sequences/samples
  dataset = Dataset.from_dict({"input_ids": samples})  # converts the only elements in the sequences into rows in the hf dataset with column name input_ids
  print(f'{dataset=}')
  print(f'{type(dataset)=}')
  ## print(f"{dataset['input_ids']=}")  # memory issues in colab

  # once we have the datasets as a table with columns as the samples/seqs and rows as the exact sample, sample a batch samples (a batch of size batch_size)
  # --
  for batch_sum in range(num_batches):
    shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else dataset
    # sample batch of samples/rows from data set
    batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(random.sample(list(range(len(shuffled_dataset))), batch_size))
    # raw_text_batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(random.sample(batch_size, batch_size))
    # tokenized_batch = map(raw_text_batch) will this being the identity work?
    tokenized_batch = map(lambda x: x, batch) #  will this being the identity work?

  samples: list[list] = [gen_ub_seq(tokenizer) for i in range(num_sequences)]  # generate sequenes/samples for lb/ub data set, list of sequences/samples
  dataset = Dataset.from_dict({"input_ids": samples})  # converts the only elements in the sequences into rows in the hf dataset with column name input_ids
  print(f'{dataset=}')
  print(f'{type(dataset)=}')
  ## print(f"{dataset['input_ids']=}")  # memory issues in colab

  # once we have the datasets as a table with columns as the samples/seqs and rows as the exact sample, sample a batch samples (a batch of size batch_size)
  # --
  for batch_sum in range(num_batches):
    shuffled_dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else dataset
    # sample batch of samples/rows from data set
    batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(random.sample(list(range(len(shuffled_dataset))), batch_size))
    # raw_text_batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(random.sample(batch_size, batch_size))
    # tokenized_batch = map(raw_text_batch) will this being the identity work?
    # tokenized_batch = map(lambda x: x, batch) #  will this being the identity work?
    # batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(random.sample(list(range(len(shuffled_dataset))), batch_size))
    batch = map(batch)

if __name__ == '__main__':
  test_lb_ds_looping_with_div_coeff_map_code()
  print('Success!\a')
