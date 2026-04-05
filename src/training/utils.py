"""
todo:
    - finish passing the HF block_size tokenization code here so its modular
    - add function to our train code train.py
    - print the sequence length of the data once we include this code
    - create a unit test here to test block size
    - use the re-init code smart ally & brando wrote
"""
from itertools import chain
import math
import random
from typing import Optional, Any

import torch

import datasets
from datasets import load_dataset, interleave_datasets

from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig
from transformers.testing_utils import CaptureLogger
from transformers import GPT2Tokenizer

def cuda_debug():
    import torch

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    # Get the CUDA version used by PyTorch
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")

    # Get the number of CUDA devices (GPUs)
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_cuda_devices}")

# # For each CUDA device, print its name and capabilities
# for i in range(num_cuda_devices):
#     print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
#     print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")


def do_quick_matrix_multiply():
    """
python -c "import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'));"
    """
    print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'))

def get_actual_data_batch(data_set_or_batch):
    """ Returns the actual  data batch from the HF dataset obj e.g., dataset, batch etc. """
    data_batch = next(iter(data_set_or_batch))
    return data_batch

def get_vocab_size_and_ln(tokenizer: GPT2Tokenizer) -> tuple[int, float]:
    """
    Calculate the vocabulary size and its natural logarithm for a given tokenizer.

    Note:
        Sanity check -- is loss random? lnV = -ln(1/V) = -ln(1/50257) = 10.82 since CE = avg_i v_i * ln(1/p_i) but only one token is right so vi = 1 for some i so CE = ln(1/p_i)

    Args:
    tokenizer (GPT2Tokenizer): A tokenizer from the Hugging Face library.

    Returns:
    tuple[int, float]: A tuple containing the vocabulary size and its natural logarithm.
    """
    vocab_size = len(tokenizer)  # Get the size of the tokenizer's vocabulary
    ln_vocab_size = math.log(vocab_size)  # Calculate the natural logarithm of the vocabulary size
    return vocab_size, ln_vocab_size

def num_tokens(max_steps: int, batch_size: int, L: int, num_batches: int) -> int:
    """
    All sequences are of length L, due to our block size code. 
    num_batch = when using distributed training. 
            num_tokens_trained = max_steps * batch_size * L * num_batches

    how long do I have to train     
    """
    num_tokens_trained = max_steps * batch_size * L * num_batches
    return num_tokens_trained

def get_freest_gpu():
    # Get the index of the GPU with the most free memory
    devices = list(range(torch.cuda.device_count()))
    free_memory = [torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) for device in devices]
    freest_device = devices[free_memory.index(max(free_memory))]
    return freest_device

# Use for IterableDatasetDict objects (i.e. streaming=T, split is unspecified (each key in dict is name of split))
def view_exs_iterable_dataset_dict(dataset, num_exs=10, split='train'):
  dataset_split = dataset[split]
  for ex in dataset_split:
    print(ex)
    print('example details: keys', ex.keys(), ', text char length', len(ex['text']), '\n---')
    num_exs -= 1
    if num_exs == 0:
      break


# Use for IterableDataset objects (i.e. streaming=T, split=specified)
def view_exs_iterable_dataset(dataset_split, num_exs=10):
  for ex in dataset_split:
    print(ex)
    print('example details: keys', ex.keys(), ', text char length', len(ex['text']), '\n---')
    num_exs -= 1
    if num_exs == 0:
      break

def get_num_steps():
    # dataset_size: int = int(1.5e12)  # TODO, doesn't seem easy to solve. Either count all the sequennces/rows or have the meta data have this. Or make this number huge. 
    # dataset_size: int = train_dataset.num_rows
    # dataset_size: int = len(train_dataset)
    # TODO dataset.info['split']['train']['num_examples']
    # dataset_size = sum(len(dataset) for dataset in datasets)  # TODO: works on with streaming = False?
    # dataset_size = sum(dataset.cardinality() for dataset in datasets)
    pass

def raw_dataset_2_lm_data(raw_dataset, 
                          tokenizer, 
                          block_size: int, 
                          desired_dataset_column: str = 'text',
                          method_to_remove_columns: str = 'keys',
                          debug: bool = False, 
                          streaming: bool = True,
                          batch_size: int = 2,
                          fromat: str = 'torch',
                          ):
    """ Get lm data set but note it uses the grou_texts function which concatenates all tests into a single sequence according to block size (some seq length e.g., max seq length)."""
    raw_dataset = raw_dataset.with_format(fromat)
    remove_columns = get_column_names(raw_dataset, method_to_remove_columns)  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    # - Get tokenized train data set
    # Note: Setting `batched=True` in the `dataset.map` function of Hugging Face's datasets library processes the data in batches rather than one item at a time, significantly speeding up the tokenization and preprocessing steps.
    tokenize_function = lambda examples: tokenizer(examples[desired_dataset_column])
    tokenized_train_datasets = raw_dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    _group_texts = lambda examples : group_texts(examples, block_size)
    # - Get actual data set for lm training (in this case each seq is of length block_size, no need to worry about pad = eos since we are filling each sequence)
    lm_dataset = tokenized_train_datasets.map(_group_texts, batched=True)
    if debug:
        batch = get_data_from_hf_dataset(lm_dataset, streaming=streaming, batch_size=batch_size)
        print(f'{len(next(iter(batch))["input_ids"])=}')
        assert all(len(data_dict['input_ids']) == block_size for data_dict in iter(batch)), f'Error, some seq in batch are not of length {block_size}'
    return lm_dataset

def get_size_of_seq_len(dataset_or_batch, verbose: bool = True, streaming: bool = True, batch_size: int = 2) -> int:
    """Print size of a sequence length in a batch. Give a hf data set obj (batches are data set objs sometimes)."""
    batch = get_data_from_hf_dataset(dataset_or_batch, streaming=streaming, batch_size=batch_size)
    size_seq_len = len(next(iter(batch))["input_ids"])
    if verbose:
        print(f'{size_seq_len=}')
        print(f'{len(next(iter(batch))["input_ids"])=}')
    return size_seq_len

def get_column_names(dataset, 
                    #   split: str = 'train',
                      method: str = 'keys', 
                      streaming: bool = True,
                      ):
    if method == 'features':
        # column_names = list(dataset[spit].features)
        column_names = list(dataset.features)
    elif method == 'keys':
        batch = get_data_from_hf_dataset(dataset, streaming=streaming, batch_size=1)
        column_names = next(iter(batch)).keys()
        # column_names = next(iter(dataset)).keys()
    else:
        raise ValueError(f"method {method} not supported")
    return column_names

def get_data_from_hf_dataset(dataset, 
                             streaming: bool = True, 
                             batch_size: int = 4, 
                            #  shuffle: bool= False, # shuffle is better but slower afaik
                            #  seed: int = 0, 
                            #  buffer_size: int = 500_000,
                             ):
    """ Gets data from a HF dataset, it's usually an iterator object e.g., some ds.map(fn, batched=True, remove_columns=remove_columns) has been applied. 
    Handles both streaming and non-streaming datasets, take for streaming and select for non-streaming.
    """
    # sample_data = dataset.select(range(batch_size)) if not isinstance(dataset, datasets.iterable_dataset.IterableDataset) else dataset.take(batch_size)
    batch = dataset.take(batch_size) if streaming else dataset.select(random.sample(list(range(len(dataset))), batch_size))
    return batch

def _tokenize_function(examples, tokenizer, tok_logger, text_column_name: str):
    """
    
    To use do:
    tokenizer = ...obtained from your model... 
    tokenize_function = lambda examples: tokenize_function(examples, tokenizer=tokenizer) 
    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
    """
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output

def tokenize_function(examples, tokenizer, text_column_name: str):
    """ 
    creates a tokenize function that can be used in HF's map function and you specify which text column to tokenize.
    
    Assumes batched=True so examples is many row/data points.
    """
    return tokenizer(examples["text_column_name"])

def preprocess(examples, tokenizer, max_length: int = 1024):
    return tokenizer(examples["text"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    # return tokenizer(examples["text"], padding="max_length", max_length=model.config.context_length, truncation=True, return_tensors="pt")

def group_texts(examples, # if batched=True it's a dict of input_ids, attention_mask, labels of len(examples['input_ids']) = 1000 
                block_size: int,  # 4096, 1024
                ):
    """
    tokenizer = ...obtained from your model... 
    tokenize_function = lambda examples: tokenize_function(examples, tokenizer=tokenizer) 
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names)

    if used as above then examples is
    examples = {'input_ids': [[...], [...], ...], 'attention_mask': [[...], [...], ...], 'labels': [[...], [...], ...]]]}
    examples.keys() = dict_keys(['input_ids', 'attention_mask'])
    type(examples) = <class 'dict'>
    type(examples['input_ids']) = <class 'list'>
    len(examples['input_ids']) = 1000  # if batched=True

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map    
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size  # rounds down
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def group_texts_v2(examples, # if batched=True it's a dict of input_ids, attention_mask, labels of len(examples['input_ids']) = 1000 
                block_size: int,  # 4096, 1024
                ):
    """
    tokenizer = ...obtained from your model... 
    tokenize_function = lambda examples: tokenize_function(examples, tokenizer=tokenizer) 
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names)
    _group_texts = lambda examples : group_texts_v2(examples, block_size)
    lm_train_dataset = tokenized_train_datasets.map(_group_texts, batched=True)

    if used as above then examples is
    examples = {'input_ids': [[...], [...], ...], 'attention_mask': [[...], [...], ...], 'labels': [[...], [...], ...]]]}
    examples.keys() = dict_keys(['input_ids', 'attention_mask'])
    type(examples) = <class 'dict'>
    type(examples['input_ids']) = <class 'list'>
    len(examples['input_ids']) = 1000  # if batched=True

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder for each of those groups of 1,000 texts. 
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map    
    """
    # Concatenate all texts for each key in the examples e.g., it creates one concatenated list of all input_ids, one for all attention_mask, etc.
    # for column_name in examples.keys():
    #     # chain makes an iterator that returns elements from each iterator in order, basically concatenates iterators 
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # # get sequences of length block_size, then add eos token to end of each sequence and mask the rest of the sequence
    # result = {}
    # for k, t in concatenated_examples.items():
    #     # Initialize a list for each key (really key="text" is the one we care about) in the result
    #     result[k] = []
    #     total_length = len(t)  # Assuming t is a list or has a length
    #     for i in range(0, total_length, block_size):
    #         # Append the sublist of t from i to i + block_size
    #         seq = t[i : i + block_size]

    #         result[k].append(t[i : i + block_size])
    result["labels"] = result["input_ids"].copy()
    return result

def collate_fn_train_only_first_eos_token_mask_everything_after_it(data: list[dict[str, str]], 
                                                                   tokenizer: PreTrainedTokenizer, 
                                                                   max_length: int=1024,  # GPT2 default, likely worth you change it! This default might cause bugs.
                                                                   ) -> dict[str, torch.Tensor]:
    """ Train only on first occurence of eos. The remaining eos are masked out.

    Sometimes the model might not have a padding token. Sometimes people set the padding token to be the eos token.
    But sometimes this seems to lead to the model to predict eos token to much. 
    So instead of actually using the pad token that was set to the eos token, we instead mask out all excesive eos tokens that act as pads 
    and leave the first eos token at the end to be predicted -- since that is the only one that semantically means end of sequence 
    and therby by not training on random eos at the end by masking it not unncesserily shift/amplify the distribution of eos. 
    
    ref: https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954/13?u=brando 
    ref: https://chat.openai.com/share/02d16770-a1f3-4bf4-8fc2-464286daa8a1
    ref: https://claude.ai/chat/80565d1f-ece3-4fad-87df-364ce57aec15 on when to call .clone()
    ref: https://stackoverflow.com/questions/76633368/how-does-one-set-the-pad-token-correctly-not-to-eos-during-fine-tuning-to-avoi
    """
    # we are training full context length for llama so remove code bellow, if it tries to pad hopefully it throws an error
    # -- Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # -- Extract sequences
    # sequences: list[str] = [example.get("text", "") or "" for example in data]
    sequences: list[str] = []
    for idx, example in enumerate(data):
        # Retrieve the value for "text" from the dictionary or default to an empty string if not present or falsy. ref: https://chat.openai.com/share/bead51fe-2acf-4f05-b8f7-b849134bbfd4
        text: str = example.get("text", "") or ""
        sequences.append(text)
    # -- Tokenize the sequences
    tokenized_data = tokenizer(sequences, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    tokenized_data["labels"] = tokenized_data["input_ids"].clone()  # labels is hardcoded in HF so put it!
    # -- Set the mask value for the first eos_token in each sequence to 1 and remaining to -100
    eos_token_id = tokenizer.eos_token_id
    for idx, input_ids in enumerate(tokenized_data["input_ids"]):
        # Find all occurrences of eos_token
        eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.nelement() > 0:  # Check if eos_token is present
            first_eos_position = eos_positions[0]
            tokenized_data["attention_mask"][idx, first_eos_position] = 1  # Set the mask value to 1
            
            # Assert that the label for the first occurrence of eos_token is eos_token_id
            assert tokenized_data["labels"][idx, first_eos_position] == eos_token_id, "The label for the first eos_token is incorrect!"
            
            # For all subsequent occurrences of eos_token, set their labels to -100
            for subsequent_eos_position in eos_positions[1:]:
                tokenized_data["labels"][idx, subsequent_eos_position] = -100
                assert tokenized_data["labels"][idx, subsequent_eos_position] == -100, "The label for the subsequent_eos_position incorrect! Should be -100."
    return tokenized_data

# -- eval code

def compute_metrics(eval_preds):
    """ todo document clearly, from SS's code. """
    import evaluate
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

def eval_hf(trainer: Trainer, path: str, name: str, split: str, max_eval_samples: Any = 'Unknown_Eval_Max_Samples',):
    metrics = trainer.evaluate()
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    path = path.replace('/', '_')  # needed only when saving results
    print(f'Eval metrics {path} {name} {split} {max_eval_samples}: {metrics=}')
    trainer.log_metrics(f"eval_{path}_{name}_{split}_{max_eval_samples}", metrics)  # display metrics
    trainer.save_metrics(f"eval_{path}_{name}_{split}_{max_eval_samples}", metrics)
    return metrics

def eval_hf_with_subsample(path, name, split, model, tokenizer, block_size, output_dir, 
                           max_eval_samples: int = 1024,
                           streaming: bool = True, 
                           verbose: bool = True,
                           print_str: Optional[str] = None,
                           ):
    eval_dataset = load_dataset(path, name, streaming=streaming, split=split).with_format("torch") 
    eval_dataset2 = raw_dataset_2_lm_data(eval_dataset, tokenizer, block_size)
    if max_eval_samples is None:
        eval_batch2 = eval_dataset2 
    else:
        eval_batch2 = eval_dataset2.take(max_eval_samples)
    print(f'Saving eval results at: {output_dir=}') # The output directory where the model predictions and checkpoints will be written.
    eval_args = TrainingArguments(output_dir=output_dir, fp16=False, bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8)
    trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_batch2)
    metrics = eval_hf(trainer, path, name, split,)
    if verbose:
        print(f'----> {path=}, {name=}, {split=}, {metrics=}, {max_eval_samples=}')
    if print_str is not None:
        print(print_str)
    return metrics

# -- unit tests -- #

def _test_all_batches_are_size_block_size():
    print('-- starting unit test')
    batch_size = 4
    # get gpt2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenize_function = lambda examples: tokenizer(examples["text"])
    # load c4 data set hf in streaming mode 
    from datasets import load_dataset
    streaming = True
    # raw_datasets = load_dataset("c4", "en", streaming=streaming, split="train")
    # raw_datasets = load_dataset('UDACA/PileSubsets', streaming=streaming).with_format('torch')   # this defaults to the subset 'all'
    # raw_datasets = load_dataset('UDACA/PileSubsets', 'pubmed', split='train', streaming=streaming).with_format('torch')
    raw_datasets = load_dataset('UDACA/PileSubsets', 'uspto', split='train', streaming=streaming).with_format('torch')
    batch = get_data_from_hf_dataset(raw_datasets, streaming=streaming, batch_size=batch_size) 
    # print(f'{batch=}')
    # print(f'{next(iter(batch))=}')
    # print(f'{next(iter(batch)).keys()}')
    # print()
    remove_columns = get_column_names(raw_datasets)  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader

    # how does it know which column to tokenize? gpt4 says default is text or your tokenized function can specify it, see my lambda fun above
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,  # Setting `batched=True` in the `dataset.map` function of Hugging Face's datasets library processes the data in batches rather than one item at a time, significantly speeding up the tokenization and preprocessing steps.
        remove_columns=remove_columns,
    )
    batch = get_data_from_hf_dataset(tokenized_datasets, streaming=streaming, batch_size=batch_size)
    # print(f'{batch=}')
    # print(f'{next(iter(batch))=}')
    # print(f'{next(iter(batch)).keys()}')
    # print()

    _group_texts = lambda examples : group_texts(examples, block_size=tokenizer.model_max_length)
    lm_datasets = tokenized_datasets.map(
        _group_texts,
        batched=True,  # Setting `batched=True` in the `dataset.map` function of Hugging Face's datasets library processes the data in batches rather than one item at a time, significantly speeding up the tokenization and preprocessing steps.
    )
    batch = get_data_from_hf_dataset(lm_datasets, streaming=streaming, batch_size=batch_size)
    # print(f'{batch=}')
    # print(f'{next(iter(batch))=}')
    # print(f'{next(iter(batch)).keys()}')
    # print()

    # - Make sure all seq are of length block_size
    batch = get_data_from_hf_dataset(lm_datasets, streaming=streaming, batch_size=batch_size)
    for data_dict in iter(batch):
        seq = data_dict['input_ids']
        print(len(seq))
    print('Success!')

def _test_train_dataset_setup_for_main_code():
    import os
    batch_size = 2
    streaming = True
    # path, name, data_files, split = ['c4'], ['en'], [None], ['train']
    # path, name, data_files, split = ['c4', 'c4'], ['en', 'en'], [None, None], ['train', 'validation']
    # path, name, data_files, split = ['csv'], [None], [os.path.expanduser('~/data/maf_data/maf_textbooks_csv_v1/train.csv')], ['train']
    # path, name, data_files, split = ['suolyer/pile_pile-cc'] + ['parquet'] * 4, [None] + ['hacker_news', 'nih_exporter', 'pubmed', 'uspto'], [None] + [urls_hacker_news, urls_nih_exporter, urls_pubmed, urls_uspto], ['validation'] + ['train'] * 4
    # path, name, data_files, split = ['UDACA/PileSubsets'], ['uspto'], [None], ['train']
    # path, name, data_files, split = ['UDACA/PileSubsets'], ['pubmed'], [None], ['train']
    path, name, data_files, split = ['UDACA/PileSubsets', 'UDACA/PileSubsets'], ['uspto', 'pubmed'], [None, None], ['train', 'train']

    # -- Get tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side="right", use_fast=False, trust_remote_code=True, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', padding_side="right", use_fast=False, trust_remote_code=True, use_auth_token=True)
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenize_function = lambda examples: tokenizer(examples["text"])
    # torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32  # if >= 8 ==> brain float 16 available or set to True if you always want fp32 
    # model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True, torch_dtype=torch_dtype, use_auth_token=True)

    # -- Get raw train data set
    raw_train_datasets = [load_dataset(p, n, data_files=data_file, streaming=streaming, split=split).with_format("torch") for p, n, data_file, split in zip(path, name, data_files, split)]
    probabilities = [1.0/len(raw_train_datasets) for _ in raw_train_datasets]  
    raw_train_datasets = interleave_datasets(raw_train_datasets, probabilities)
    # raw_train_datasets = load_dataset(path[0], name[0], data_files=data_files[0], streaming=streaming, split=split[0]).with_format("torch")
    # raw_train_datasets = load_dataset('UDACA/PileSubsets', 'uspto', split='train', streaming=streaming).with_format('torch')
    batch = get_data_from_hf_dataset(raw_train_datasets, streaming=streaming, batch_size=batch_size) 
    print(f'{batch=}')
    print(f'{next(iter(batch))=}')
    print(f'{next(iter(batch)).keys()}')
    print()
    remove_columns = get_column_names(raw_train_datasets)  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    
    # - Get tokenized train data set
    # Note: Setting `batched=True` in the `dataset.map` function of Hugging Face's datasets library processes the data in batches rather than one item at a time, significantly speeding up the tokenization and preprocessing steps.
    tokenized_train_datasets = raw_train_datasets.map(tokenize_function, batched=True, remove_columns=remove_columns)
    # block_size: int = tokenizer.model_max_length
    block_size: int = 4096
    assert block_size != 1000000000000000019884624838656, f'Error, block_size is {block_size} which is the default value. This is likely because you are using a tokenizer that does not have a model_max_length attribute. Please set block_size to a value that makes sense for your model.'
    _group_texts = lambda examples : group_texts_v2(examples, block_size)
    batch = get_data_from_hf_dataset(tokenized_train_datasets, streaming=streaming, batch_size=batch_size) 
    print(f'{batch=}')
    print(f'{next(iter(batch))=}')
    print(f'{next(iter(batch)).keys()}')
    
    # - Get data set for lm training (in this case each seq is of length block_size, no need to worry about pad = eos since we are filling each sequence)    lm_train_dataset = tokenized_train_datasets
    lm_train_dataset = tokenized_train_datasets.map(_group_texts, batched=True)
    batch = get_data_from_hf_dataset(lm_train_dataset, streaming=streaming, batch_size=batch_size)
    print(f'{batch=}')
    # - get an example for debugging
    # batch = iter(batch)
    # example = next(batch)
    # print(f'{example=}')
    # print(f'{next(iter(batch))=}')
    print(f'{next(iter(batch)).keys()}')
    
    # - Make sure all seq are of length block_size
    batch = get_data_from_hf_dataset(lm_train_dataset, streaming=streaming, batch_size=batch_size)
    for data_dict in iter(batch):
        seq = data_dict['input_ids']
        print(len(seq))
    print('Success!')

def _test_expt_planning():
    # -- 2.5B tokens
    num_tokens_desired: int = int(2.5e9)
    batch_size = 32
    num_batches = 1
    L = 4096
    # num_tokens_trained = max_steps * batch_size * L * num_batches
    max_steps = num_tokens_desired / (batch_size * L * num_batches)
    print(f'{max_steps=}')
    # 19_073
    # 281:07:43 --> 11 days ...

    # -- 5.5M tokens
    num_tokens_desired: int = int(5.5e6)
    max_steps = num_tokens_desired / (batch_size * L * num_batches)
    print(f'{max_steps=}')
    # 42

def _test_utils_padding_and_eos():
    # GPT2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    block_size = 1024
    print(f'{tokenizer.model_max_length=}')
    print(f'{block_size=}')
    raw_dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    lm_dataset = raw_dataset_2_lm_data(raw_dataset, tokenizer, block_size=block_size)
    # take a batch of size 2 and print it
    batch = get_data_from_hf_dataset(lm_dataset, streaming=True, batch_size=2) 
    print(f'{batch=}')
    data_batch = next(iter(batch))
    # todo: test that when length changes attention mask labels etc make sense
    # todo: do we need to put eos & padding and make sure label = -1? 
    print()

def _test_log_trainer():
    # gpt2 model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to("cuda")
    raw_dataset = load_dataset("c4", "en", streaming=True, split="train").with_format("torch")
    eval_dataset = raw_dataset.take(2)
    name = 'c4_fake'
    eval_args = TrainingArguments(output_dir='.') 
    metrics = {'eval_loss': 0.1, 'eval_runtime': 0.1, 
               'eval_samples_per_second': 0.1, 'eval_steps_per_second': 0.1, 'perplexity': 0.1, 'name': name}
    # trainer.save_metrics(f"eval_{name}", metrics)
    trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_dataset)
    trainer.log_metrics(f"eval_{name}", metrics)  # display metrics

if __name__ == "__main__":
    from time import time
    start_time = time()
    # _test_all_batches_are_size_block_size()
    # _test_train_dataset_setup_for_main_code()
    # _test_expt_planning()
    # _test_utils_padding_and_eos()
    _test_log_trainer()
    print(f"Done!\a Total time: {time() - start_time} seconds, or {(time() - start_time)/60} minutes. or {(time() - start_time)/60/60} hours.\a")
