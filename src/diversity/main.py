from copy import deepcopy
from pathlib import Path
import os
import argparse
import json
import time
import numpy as np
import torch
import math

from task2vec import Task2Vec
import task_similarity

from datasets import load_dataset
from transformers import AutoConfig, GPT2Tokenizer, GPT2LMHeadModel

# map from The Pile subdataset name (from HuggingFace) to data load type
thepile_loadtype_dict = {'conceptofmind/pile_cc': 'sep_ds',
 'enron_emails': 'field',
 'hacker_news': 'field',
 'nih_exporter': 'field',
 'pubmed': 'field',
 'uspto': 'field'}

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_tasks", default=None, type=int, required=True,
                        help="The number of tasks to sample from data and compute diversity for.")
    parser.add_argument("--finetune", default=False, action='store_true',
                        help="Whether to run finetuning on probe network.")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Whether or not to use a pretrained probe network.")
    
    ## Other parameters
    parser.add_argument("--subdataset", default=None, type=str,
                        help="Specify what subset of the sub-datasets of The Pile will be used to train \
                            (i.e. first = first two, mid = middle two, last = last two).\
                            This arg should only be used when task_name == the_pile")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--break_early", default=False,
                        help="Break after 1 iteration.")
    parser.add_argument("--buffer_size", default=10_000, type=int,
                        help="Buffer size for streamed data.")
    parser.add_argument("--batch_size", default=512, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'run_args.txt'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=2))
        f.close()
    
    # Load dataset
    ds_dict = {}
    if args.task_name == "c4":
        print("DATASET: C4")
        ds_dict["c4"] = load_dataset(args.task_name, "en", streaming=True, split="train").with_format("torch")
        remove_columns = ["text", "timestamp", "url"]
    elif args.task_name == "wikitext":
        print("DATASET: WIKITEXT")
        ds_dict["WikiText-103"] = load_dataset("wikitext", 'wikitext-103-v1', streaming=True, split="train").with_format("torch")
        remove_columns = ["text"]
    elif args.task_name == "the_pile":
        print("DATASET: THE PILE")
        ds_dict["The Pile"] = load_dataset("the_pile", streaming=True, split="train").with_format("torch")
        remove_columns = ["text", "meta"]
    elif args.task_name == "the_pile_sametaskds":
        if args.subdataset == "first":
            tp_datasets = list(thepile_loadtype_dict.keys())[0:2]
        elif args.subdataset == "mid":
            tp_datasets = list(thepile_loadtype_dict.keys())[2:4]
        elif args.subdataset == "last":
            tp_datasets = list(thepile_loadtype_dict.keys())[4:]
        print("THE PILE DATASET: ", tp_datasets)
        
        for tp in tp_datasets:
            load_type = thepile_loadtype_dict[tp]
            if load_type == 'sep_ds':
                ds = load_dataset(tp, streaming=True, split="train").with_format("torch")
            elif load_type == 'field':
                ds = load_dataset("the_pile", tp, streaming=True, split="train").with_format("torch")
            elif load_type == "json":
                ds = load_dataset(
                    "json",
                    data_files="https://the-eye.eu/public/AI/pile_preliminary_components/" + tp,
                    split='train',
                    streaming=True,
                ).with_format("torch")
            else:
                print("COULD NOT LOAD DATA FOR ", tp)
                exit(0)
                
            if tp == 'conceptofmind/pile_cc':
                ds_dict['pile_cc'] = ds
            else:
                ds_dict[tp] = ds
        
        remove_columns = ['text', 'meta']
    

    # Load GPT-2 model and tokenizer (pretrained or randomly initialized)
    if args.pretrained:
        print("USING PRETRAINED MODEL")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir if args.cache_dir else None)
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        print("USING RANDOM MODEL")
        config = AutoConfig.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir if args.cache_dir else None)
        model = GPT2LMHeadModel(config)

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id == 50256

    # Tokenize examples
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=args.max_seq_length, truncation=True, return_tensors="pt")
    
    def process_and_filter(batch):
        """This function removes empty examples."""
        result = {"text": []}
        for text in batch["text"]:
            if len(text)> 0:
                result["text"].append(text)
        return result
    
    if args.task_name == "wikitext":
        print("PROCESS AND FILTER")
        ds = ds.map(process_and_filter, batched=True)
        
    # Compute Task2Vec embeddings
    embeddings, losses = [], []
    for key, ds in ds_dict.items():
        print("CURRENT DATASET: ", key)
        for task_num in range(args.num_tasks):
            print(f'--> {task_num=}\n')
            seed = args.seed + task_num
            classifier_opts = {'break_early': args.break_early, "finetune": args.finetune, "seed": seed, "epochs": args.epochs, 
                "task_batch_size": args.batch_size}
            
            shuffled_dataset = ds.shuffle(buffer_size=args.buffer_size, seed=seed)
            task_dataset = shuffled_dataset.take(args.batch_size)
            tokenized_task_dataset = task_dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)
            
            probe_network = model
            start = time.time()
            embedding, loss = Task2Vec(deepcopy(probe_network), classifier_opts=classifier_opts).embed(tokenized_task_dataset)
            end = time.time()
            print("TIME TO COMPUTE TASK2VEC:", end - start)
            print(f'{embedding.hessian.shape=}')
            embeddings.append(embedding)
            if loss is not None:
                losses.append(loss)
            
            # Save embeddings and loss in output_dir, and overwrite existing embeddings and loss checkpt files.
            num_tasks_processed = task_num + 1
            np.save(os.path.join(args.output_dir, key + '_embeddings_' + str(num_tasks_processed) + 'tasks.npy'), embeddings)
            if loss is not None:
                np.save(os.path.join(args.output_dir, key + '_loss_' + str(num_tasks_processed) + 'tasks.npy'), losses)
            
            # Remove previous checkpt files.
            last_num_tasks = num_tasks_processed - 1
            if last_num_tasks > 0:
                last_file = os.path.join(args.output_dir, key + '_embeddings_' + str(last_num_tasks) + 'tasks.npy')
                if os.path.isfile(last_file):
                    os.remove(last_file)
                if loss is not None:
                    last_file = os.path.join(args.output_dir, key + '_loss_' + str(last_num_tasks) + 'tasks.npy')
                    if os.path.isfile(last_file):
                        os.remove(last_file)

    # Compute pairwise cosine distance matrix between Task2Vec embeddings
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')
    np.save(os.path.join(args.output_dir, 'distance_matrix.npy'), distance_matrix)

    results: dict = {'embeddings': [embed for embed in embeddings],
                     'distance_matrix': distance_matrix,
                     'losses': [loss for loss in losses],
                     "num_tasks": args.num_tasks}
    np.save(os.path.join(args.output_dir, 'results.npy'), results)
    
if __name__ == '__main__':
    main()