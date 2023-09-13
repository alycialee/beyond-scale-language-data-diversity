"""
Original descriptions from paper:

2.3. Recipe for Establishing if a Diversity Coefficient is
High via the Conceptual Lower and Upper Bounds
To establish if a diversity coefficient ˆdiv(D) of a dataset D
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
  - https://claude.ai/chat/f53bcb39-2c54-4e02-a831-87c6cf7f0d80
  - https://chat.openai.com/c/56296331-190f-4572-868c-12d510d19c69
"""

from transformers import GPT2Tokenizer
from datasets import Dataset
from typing import List, Dict

def create_lower_bound_dataset(num_batches: int = 200, 
                               batch_size: int = 512,
                               tokenizer: GPT2Tokenizer = None) -> List[Dict[str, List[int]]]:
    """
    Create a synthetic dataset for computing the lower bound of the diversity coefficient.
    
    This dataset concentrates most of the probability mass on an arbitrary token that is not a special token (especially not EOS).
    This leads to minimal diversity (since a sequence is the repetition of a single token).

    Original paper explanation:
      Consider a dataset constructed by sampling with
      most of the probability mass concentrated on some arbitrary
      token. This is a good candidate for a dataset with minimum
      diversity.
      Therefore, we measure a conceptual lower bound on a
      dataset with a vocabulary size of 2: <eos> token and a
      randomly selected non-special token from the GPT-2 tokenizer vocabulary. The <eos> token was assigned a probability weight of 1/{GPT-2 vocab size}. The non-special
      token was assigned the remaining weight
    
    Args:
        num_batches: Number of batches to create.
        batch_size: Number of samples per batch.
        tokenizer: Tokenizer instance for token-ID conversions.
        
    Returns:
        A list of dictionaries, where each dictionary has a key 'input_ids' mapped to a list of token IDs.
    """
    # Tokenize <eos> token
    eos_token_id = tokenizer.convert_tokens_to_ids(['<eos>'])[0]
    
    # Determine the size of the GPT-2 tokenizer vocabulary.
    vocab_size = len(tokenizer)
    # Initially, sample a random token ID from the entire GPT-2 vocabulary.
    non_special_token_id = np.random.choice(range(vocab_size))
    # Check if the selected token ID corresponds to a special token.
    # If it does, continue sampling new token IDs until we find a non-special token.
    while tokenizer.convert_ids_to_tokens([non_special_token_id])[0].startswith('<'):
        # Resample a new token ID because the previous one was a special token.
        non_special_token_id = np.random.choice(range(vocab_size))
    # At this point, non_special_token_id contains the ID of a randomly selected non-special token.

    # Assign probability weights 
    p_eos = 1 / vocab_size  
    p_other = 1 - p_eos
    
    # Generate batches with a concentration on the <eos> token and the randomly selected non-special token
    all_batches = []
    for _ in range(num_batches):
        batch = {"input_ids": [eos_token_id if np.random.rand() < p_eos else non_special_token_id for _ in range(batch_size)]}
        all_batches.append(batch)
        
    return all_batches

def create_upper_bound_dataset(num_batches: int = 200, batch_size: int = 512) -> List[Dict[str, List[int]]]:
    """
    Create a synthetic dataset for computing the upper bound of the diversity coefficient.
    
    This dataset samples tokens uniformly at random from the GPT-2 vocabulary, leading to maximum diversity.

    Original paper explanation:
      On the other extreme, a dataset constructed by
      sampling any token uniformly at random given a fixed vocabulary (in our case, the GPT-2 tokenizer vocabulary) is a
      good candidate to create a dataset with maximum diversity.
      Similarly, a high or maximum diversity dataset would consist of random sequences of all possible tokens, with no underlying order
      to semantics, formatting, etc. The upper bound of the diversity coefficient was therefore measured on a synthetic
      dataset with an equal probability of occurrence assigned to
      all tokens in the GPT-2 tokenizer vocabulary.
    
    Args:
        num_batches: Number of batches to create.
        batch_size: Number of samples per batch.
        
    Returns:
        A list of dictionaries, where each dictionary has a key 'input_ids' mapped to a list of token IDs.
    """
    
    vocab_size = len(tokenizer)
    
    # Generate batches with uniform sampling across the GPT-2 vocabulary
    all_batches = []
    for _ in range(num_batches):
        batch = {"input_ids": [np.random.choice(vocab_size) for _ in range(batch_size)]}
        all_batches.append(batch)
        
    return all_batches

# Create synthetic datasets for both bounds
lower_bound_data = create_lower_bound_dataset(tokenizer=tokenizer)
upper_bound_data = create_upper_bound_dataset()

# Convert synthetic datasets into HuggingFace Datasets
lower_bound_dataset = Dataset.from_dict({"input_ids": [item["input_ids"] for item in lower_bound_data]})
upper_bound_dataset = Dataset.from_dict({"input_ids": [item["input_ids"] for item in upper_bound_data]})

# Compute the diversity coefficients for both datasets
lower_bound_diversity = get_diversity_coefficient(lower_bound_dataset)
upper_bound_diversity = get_diversity_coefficient(upper_bound_dataset)

# Display the results
print(f"Lower Bound Diversity Coefficient: {lower_bound_diversity}")
print(f"Upper Bound Diversity Coefficient: {upper_bound_diversity}")
