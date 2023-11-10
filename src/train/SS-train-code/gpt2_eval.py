import logging
import math
import os
import sys
import pprint
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from pprint import pprint as pp
import tqdm
import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import datasets
import evaluate
import torch
from datasets import load_dataset, interleave_datasets
from evaluate import evaluator

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    pipeline
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.31.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

'''
Important arguments in ModelArguments: 
- 'model_dir': path/to/checkpoint/folder / specifies path to checkpoint folder of the model you want to evaluate
'''
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Folder from which to load model checkpoint anc config for evaluation"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default='gpt2',     # Default set to GPT-2 architecture
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default='n_embd=64,n_layer=3,n_head=2',  # Default set to 3.4M param model
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='gpt2', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}    # Default set to GPT-2 pretrained tokenizer
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,       # Note: Not sure if/how this affects performance. My guess is it's a small performance cost for medium speed up
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

'''
Important arguments in DataTrainingArguments: 
- 'dataset_name': e.g. 'EleutherAI/pile' / specifies HF dataset address of eval data
- 'dataset_config_name': e.g. 'pubmed', 'pubmed+uspto' / specifies subset of above dataset. Using '+' between subsets interleaves the datasets given a mix (default equal-parts mixture)
- 'max_eval_samples': same idea as above
- 'data_mix': e.g. '.3,.3,.4', '0.75,0.25' / specifies the data mix for each subset of the overall interleaved dataset. Should not be provided for non-interleaved datasets.
- 'streaming': e.g. 'CacheStream', 'True', 'False' / specifies how the dataset is loaded. 'CacheStream' is fastest, 'True' is easiest for testing, 'False' is not yet implemented/tested for this script. 
- 'split': e.g. 'train', 'test' / specifies which split of the dataset is used for evaluation. By default, this is "validation."
'''
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default='EleutherAI/pile', metadata={"help": "The name of the dataset to use (via the datasets library)."}  # Default set to The Pile on HF hub
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split: Optional[str] = field(
        default='validation', metadata={"help": "The split name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: Optional[str] = field(default='True', metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    total_train_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "The total number of examples to be seen per batch during training. Overrides per_device_train_batch_size."},
    )


    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    """
    ARGS SETUP
    """
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.overwrite_output_dir is True:
        print('NOTE: Script WILL overwrite output directory', training_args.output_dir)
    else:
        print('NOTE: Script will NOT overwrite output directory', training_args.output_dir)

    print('NOTE: Total number of eval samples set to', data_args.max_eval_samples)

    print('> > > Args parsed correctly. This is what I\'ve got:', training_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    """
    MODEL SETUP
    """
    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": "main",
        "use_auth_token": None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    config = CONFIG_MAPPING[model_args.model_type]()
    config = config.from_json_file(model_args.model_dir+'/config.json')
    print(config)

    model = AutoModelForCausalLM.from_config(config)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print('model param count:', n_params)

    model.load_state_dict(
        torch.load(model_args.model_dir + '/pytorch_model.bin', map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    model.eval()

    """
    DATA SETUP
    """
    raw_datasets = None
    if data_args.dataset_name is not None:
        # Assumes you have already cached the datasets you want to use. If you haven't, this will automatically cache the dataset during the cached_dataset = load_dataset() call
        if data_args.streaming == 'CacheStream':
            # Logic for single datasets from HF hub, e.g. just Pubmed
            if data_args.dataset_config_name is None or '+' not in data_args.dataset_config_name:
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=True,
                )
                cached_dataset = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=False,
                )

                raw_datasets["validation"] = cached_dataset[data_args.dataset_split].to_iterable_dataset()

                print('> > > Loaded *single* dataset from HF hub. Details: overall name:', data_args.dataset_name,
                      'sub-dataset:', data_args.dataset_config_name, 'streaming:', data_args.streaming, 'split:', data_args.dataset_split)

            # Logic for interleaved datasets, e.g. Pubmed + USPTO with mix [0.4, 0.6]
            else:
                sub_dataset_names = data_args.dataset_config_name.split('+')
                sub_datasets = []

                for sub_dataset_name in sub_dataset_names:
                    cached_dataset = load_dataset(
                        data_args.dataset_name,
                        sub_dataset_name,
                        cache_dir=model_args.cache_dir,
                        use_auth_token=True if model_args.use_auth_token else None,
                        streaming=False,
                    )

                    sub_datasets.append(cached_dataset[data_args.dataset_split].to_iterable_dataset())

                # Datasets interleaved equally unless size is specified
                if data_args.data_mix is None:
                    data_mix = [1.0 / len(sub_datasets) for i in range(len(sub_datasets))]
                else:
                    data_mix = data_args.data_mix.split(',')
                    data_mix = [float(elem) for elem in data_mix]

                raw_dataset = interleave_datasets(sub_datasets, probabilities=data_mix)

                # Put data into the format of IterableDatasetDict (to make Trainer calls work smoothly later in the script)
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    sub_dataset_names[0],
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=True,
                )

                raw_datasets["validation"] = raw_dataset

                print('> > > Loaded *interleaved dataset* from HF hub. Details: overall name:', data_args.dataset_name,
                      'sub-datasets:', sub_dataset_names, 'interleave probabilities:', data_mix, 'streaming:',
                      data_args.streaming, 'split:', data_args.dataset_split)
        # Assumes no caching; streams dataset directly from HF server. Can be slower/more unreliable than streaming from cache, but varies by dataset.
        elif data_args.streaming == 'True':
            if data_args.dataset_config_name is None or '+' not in data_args.dataset_config_name:
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=True,
                )

                raw_datasets["validation"] = raw_datasets[data_args.dataset_split]

                print('> > > Loaded *single* dataset from HF hub. Details: overall name:', data_args.dataset_name,
                      'sub-dataset:', data_args.dataset_config_name, 'streaming:', data_args.streaming, 'split:', data_args.dataset_split)

            # Logic for interleaved datasets, e.g. Pubmed + USPTO with mix [0.4, 0.6]
            else:
                sub_dataset_names = data_args.dataset_config_name.split('+')
                sub_datasets = []

                for sub_dataset_name in sub_dataset_names:
                    cached_dataset = load_dataset(
                        data_args.dataset_name,
                        sub_dataset_name,
                        cache_dir=model_args.cache_dir,
                        use_auth_token=True if model_args.use_auth_token else None,
                        streaming=True,
                    )

                    sub_datasets.append(cached_dataset[data_args.dataset_split])

                # Datasets interleaved equally unless size is specified
                if data_args.data_mix is None:
                    data_mix = [1.0 / len(sub_datasets) for i in range(len(sub_datasets))]
                else:
                    data_mix = data_args.data_mix.split(',')
                    data_mix = [float(elem) for elem in data_mix]

                if sum(data_mix) != 1.0:
                    raise ValueError(
                        "The data mixture (data_mix) you provided is invalid. Sum of elements must be 1.0, but yours was",
                        sum(data_mix))
                raw_dataset = interleave_datasets(sub_datasets, probabilities=data_mix)

                # Put data into the format of IterableDatasetDict (to make Trainer calls work smoothly later in the script)
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    sub_dataset_names[0],
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=True,
                )

                raw_datasets["validation"] = raw_dataset

                print('> > > Loaded *interleaved dataset* from HF hub. Details: overall name:', data_args.dataset_name,
                      'sub-datasets:', sub_dataset_names, 'interleave probabilities:', data_mix, 'streaming:',
                      data_args.streaming, 'split:', data_args.dataset_split)
        elif data_args.streaming == 'False':
            raise ValueError('streaming=False not implemented/tested.')
    else:
        raise ValueError('Please specify the name of a HuggingFace dataset to be used for evaluation')

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    print('> > > Set proper embedding size')

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    print('> > > Gotten column names')

    # Since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # Used to tokenize input (eval) dataset
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
        print('> > > Token mapped streaming dataset')

    # TODO: Check if compute flops assessment changes with token length of example (copy code from Trainer obj sourcecode and test it in different cases)
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # TODO: check the raw lengths of the first n examples, and then the concatenated lengths of the first n examples, to see if both over and under is properly addressed
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
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
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
        print('> > > Text grouped streaming dataset')

    """
    EVALUATION
    """
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets["validation"]
    max_eval_samples = data_args.max_eval_samples
    eval_dataset = eval_dataset.take(max_eval_samples)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    print('> > > Did do_eval setup')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    print('> > > Instantiated trainer')

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        print('> > > Beginning evaluation...')
        # Note: running .evaluate() doesn't currently have a progress bar, so output will be 'blank' until evaluation is finished.
        metrics = trainer.evaluate()                # Source code: https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/trainer.py#L2974

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


'''
conda activate train_test1
python src_div_emergence_icl/experiment_model_training/eval_test_trainer.py --output_dir ./scrap_results2 --overwrite_output_dir True --do_eval True --optim adamw_torch --config_overrides n_embd=512,n_layer=8,n_head=8


python gpt2_eval.py \
--output_dir eval-scrap1 \
--model_dir ../checkpoint-20000-pubmed-50M \
--do_eval \
--max_eval_samples 10 \
--save_steps 2000 \
--optim adamw_torch \
--dataset_config_name pubmed \
--dataset_split train \
--streaming CacheStream


python gpt2_eval.py \
--output_dir eval-scrap1 \
--model_dir ../checkpoint-20000-pubmed-50M \
--do_eval \
--max_eval_samples 10 \
--save_steps 2000 \
--optim adamw_torch \
--dataset_name wikitext \
--dataset_config_name wikitext-103-v1 \
--streaming True
'''
if __name__ == "__main__":
    main()
