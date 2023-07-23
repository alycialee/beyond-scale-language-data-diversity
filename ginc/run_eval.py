# Adapted from https://github.com/p-lambda/incontext-learning/blob/main/run_clm.py
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from typing import List
import wandb

from datasets import load_dataset
import torch

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
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    small_model: bool = field(
        default=True,
        metadata={"help": "Whether to use small model"},
    )
    custom_embedding_size: int = field(
        default=768,
        metadata={"help": "embedding size in the custom `small` model"},
    )
    custom_num_layers: int = field(
        default=12,
        metadata={"help": "number of layers in the custom `small` model"},
    )
    custom_num_heads: int = field(
        default=12,
        metadata={"help": "number of heads in the custom `small` model"},
    )
    custom_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use custom tokenizer"},
    )
    eval_incontext: bool = field(
        default=False,
        metadata={"help": "Whether to eval in context prompts"},
    )
    eval_incontext_results: str = field(
        default='in_context_results.tsv',
        metadata={"help": "in context results tsv name"},
    )
    generate_cache: bool = field(
        default=False,
        metadata={"help": "Only generate cache"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    n_train_samples: int = field(
        default=1000,
        metadata={
            "help": ""
        },
    )
    n_hmms: int = field(
        default=10,
        metadata={
            "help": ""
        },
    )
    n_slots: int = field(
        default=10,
        metadata={
            "help": ""
        },
    )
    n_symbols: int = field(
        default=50,
        metadata={
            "help": ""
        },
    )
    n_values: int = field(
        default=10,
        metadata={
            "help": ""
        },
    )
    dataset_seed: int = field(
        default=0,
        metadata={
            "help": ""
        },
    )
    start_temp: float = field(
        default=10.,
        metadata={
            "help": "The temperature used for the transition matrix"
        },
    )
    transition_temp: float = field(
        default=0.1,
        metadata={
            "help": "The temperature used for the transition matrix"
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    value_identity_coeff: float = field(
        default=0.9,
        metadata={
            "help": "The temperature used for the transition matrix"
        },
    )


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def eval_prompts(model, data_dir, tokenizer, output_dir, results_name='in_context_results.tsv',
                 prompt_lengths: List[int] = None, before_train=False):

    if prompt_lengths is None:
        prompt_lengths = [3, 5, 8, 10]

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    paths = [
        [
            [
                data_dir / f'id_prompts_randomsample_{prompt_length}.json',
                data_dir / f'ood_prompts_randomsample_{prompt_length}.json'
            ]
        ]
        for prompt_length in prompt_lengths
    ]
    paths = list(itertools.chain.from_iterable(paths))

    if before_train:
        results_names = [Path(results_name).stem + f'_beforetrain_randomsample_{prompt_length}.tsv'
                     for prompt_length in prompt_lengths]
    else:
        results_names = [Path(results_name).stem + f'_aftertrain_randomsample_{prompt_length}.tsv'
                     for prompt_length in prompt_lengths]

    model = model.cuda()
    model.eval()

    accuracies = []
    dfs_list = []
    with torch.no_grad():
        for path_ls, curr_results_name, prompt_length in zip(paths, results_names, prompt_lengths):
            results = []
            for path, in_or_out_of_distribution in zip(path_ls, ['in', 'out']):
                if not path.exists():
                    continue
                df = pd.read_json(path, lines=True)

                num_examples_list = df['n_examples'].unique()
                for num_examples in num_examples_list:
                    df_n = df[df['n_examples'] == num_examples]

                    ex_len = len(tokenizer(df_n.iloc[0]['text'])['input_ids'])
                    if ex_len > 1024:
                        continue

                    batch_size = (4 * 1024) // ex_len
                    num_batches = len(df_n) // batch_size
                    if len(df_n) % batch_size != 0:
                        num_batches += 1

                    preds = []
                    labels = []
                    format_matches = []
                    for i in range(num_batches):
                        batch = df_n['text'].iloc[i*batch_size: (i+1)*batch_size]
                        batch_labels = df_n['label'].iloc[i*batch_size: (i+1)*batch_size]
                        batch = torch.tensor([tokenizer(b)['input_ids'] for b in batch.tolist()]).cuda()
                        length = batch.shape[1]
                        out = model.generate(input_ids=batch,
                                       max_length=length + 1,
                                       temperature=0.0,
                                       do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
                        out = out.detach().cpu().numpy()
                        pred = out[:, length]

                        preds += list(pred)
                        labels += [tokenizer(bl)['input_ids'][0] for bl in batch_labels.tolist()]

                    preds = np.asarray(preds)
                    labels = np.asarray(labels)
                    acc = np.mean(preds == labels)
                    accuracies.extend((preds == labels).tolist())
                    print(f"PATH: {path}, NUM EXAMPLES: {num_examples}, ACC: {acc}")
                    results.append({'num_examples': num_examples, 'acc': acc,
                                    'prompt_length': prompt_length,
                                    'in_or_out_of_distribution': in_or_out_of_distribution,
                                    'path': path})
            df = pd.DataFrame(results)
            df.to_csv(output_dir / curr_results_name, sep='\t')
            # print(df)
        #     dfs_list.append(df)

        # dfs = pd.concat(dfs_list)

        # acc_by_num_examples_by_prompt_length = {}
        # for row, row in dfs.iterrows():
        #     acc_by_num_examples_by_prompt_length[
        #         f'eval_acc/prmptlen={row["prompt_length"]}_numex={row["num_examples"]}'] = row['acc']

        # wandb.log(acc_by_num_examples_by_prompt_length)


def main():
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

    # What are we doing here? To explain, GINC's code was written to accept the pre-formed
    # path to be passed in. However, to work with W&B's sweeps, I couldn't figure out how
    # to pre-form the path. So we make the path here and overwrite the appropriate values.
    disk_path = './'
    ginc_datasets_dir_path = os.path.join(disk_path, 'ginc-output-repro/data')
    ginc_runs_dir_path = os.path.join(disk_path, 'ginc-output-repro/train')
    os.makedirs(ginc_runs_dir_path, exist_ok=True)
    if data_args.n_train_samples == 1000:
        dataset_name = 'GINC_trans{}_start{}_nsymbols{}_nvalues{}_nslots{}_vic{}_nhmms{}_seed{}'.format( # _nsamples{}
            data_args.transition_temp,
            data_args.start_temp,
            data_args.n_symbols,
            data_args.n_values,
            data_args.n_slots,
            data_args.value_identity_coeff,
            #data_args.n_train_samples,
            data_args.n_hmms,
            data_args.dataset_seed
        )
    else:
        dataset_name = 'GINC_trans{}_start{}_nsymbols{}_nvalues{}_nslots{}_vic{}_nsamples{}_nhmms{}_seed{}'.format(
            data_args.transition_temp,
            data_args.start_temp,
            data_args.n_symbols,
            data_args.n_values,
            data_args.n_slots,
            data_args.value_identity_coeff,
            data_args.n_train_samples,
            data_args.n_hmms,
            data_args.dataset_seed
        )
    ginc_dataset_dir_path = os.path.join(ginc_datasets_dir_path, dataset_name)
    assert os.path.isdir(ginc_dataset_dir_path), f'Dataset directory {ginc_dataset_dir_path} does not exist.'
    model_args.tokenizer_name = os.path.join(ginc_dataset_dir_path, 'tokenizer.json')
    data_args.train_file = os.path.join(ginc_dataset_dir_path, 'train.json')
    data_args.validation_file = os.path.join(ginc_dataset_dir_path, 'val.json')
    # run_dir_path = os.path.join(ginc_runs_dir_path, '{}_nlayers={}_dembed={}_trainseed={}'.format(
    #     dataset_name,
    #     model_args.custom_num_layers,
    #     model_args.custom_embedding_size,
    #     training_args.seed))
    # os.makedirs(run_dir_path, exist_ok=True)
    # print("DATA DIR:", ginc_dataset_dir_path)
    assert model_args.model_name_or_path != ""
    ind = model_args.model_name_or_path.index("/checkpoint")
    training_args.output_dir = model_args.model_name_or_path[:ind]
    # print("OUTPUT DIR:", training_args.output_dir)
    # exit(0)
    # Join the args to log to W&B.
    # args = {**model_args.__dict__, **data_args.__dict__, **training_args.__dict__}
    # run = wandb.init(project='ginc-icl',
    #                  name=run_dir_path,
    #                  config=args)

    # Detecting last checkpoint.
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.info(f"Dataset dir: {ginc_dataset_dir_path}")
    logger.info(f"Output dir: {training_args.output_dir}")
    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if is_main_process(training_args.local_rank):
    #     transformers.utils.logging.set_verbosity_info()
    #     transformers.utils.logging.enable_default_handler()
    #     transformers.utils.logging.enable_explicit_format()
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if model_args.eval_incontext:
    #     pass
    # if data_args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    #     if "validation" not in datasets.keys():
    #         datasets["validation"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[:{data_args.validation_split_percentage}%]",
    #         )
    #         datasets["train"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[{data_args.validation_split_percentage}%:]",
    #         )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #     extension = (
    #         data_args.train_file.split(".")[-1]
    #         if data_args.train_file is not None
    #         else data_args.validation_file.split(".")[-1]
    #     )
    #     if extension == "txt":
    #         extension = "text"
    #     datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        print("getting model from model name or path")
        print(model_args.model_name_or_path)
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.custom_tokenizer:
        from transformers import PreTrainedTokenizerFast
        eot = '[endoftext]'
        tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=model_args.tokenizer_name,
                bos_token=eot,
                eos_token=eot,
                unk_token=eot)

    elif model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.small_model:
        config.vocab_size = tokenizer.vocab_size
        config.n_embd = model_args.custom_embedding_size
        config.n_layer = model_args.custom_num_layers
        config.n_head = model_args.custom_num_heads
        logger.info('Running with small model')
    else:
        logger.info('Running with large model')

    # logger.info('Config: ', config)

    if model_args.model_name_or_path:
        logger.info('Training model from pretrained')
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if model_args.eval_incontext:
        logger.info("*** Evaluate In-Context 1 ***")
        model.eval()
        eval_prompts(model,
                     Path(data_args.train_file).parent,
                     tokenizer,
                     results_name=dataset_name + "_" + model_args.eval_incontext_results,
                     output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
    print("Finished run eval!")
