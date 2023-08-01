from copy import deepcopy
from pathlib import Path
import os
import argparse
import json
import numpy as np
import torch
import math

from task2vec import Task2Vec
import task_similarity

from datasets import load_dataset
from transformers import AutoConfig, GPT2LMHeadModel, PreTrainedTokenizerFast

def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters 
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_tasks", default=None, type=int,
                        help="The number of tasks to sample from data and compute diversity for.")
    parser.add_argument("--finetune", default=False, action='store_true',
                        help="Whether to run finetuning on probe network.")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Whether or not to use a pretrained probe network.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--break_early", default=False,
                        help="Break after 1 iteration.")
    parser.add_argument("--buffer_size", default=10_000, type=int,
                        help="Buffer size for streamed data.")
    parser.add_argument("--batch_size", default=128, type=int,
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

    ## GINC parameters
    parser.add_argument('--n_hmms', type=int, required=True)
    parser.add_argument('--n_slots', type=int, default=10)
    parser.add_argument('--n_symbols', type=int, required=True)
    parser.add_argument('--n_values', type=int, default=10)
    parser.add_argument('--dataset_seed', type=int, default=1111)
    parser.add_argument('--transition_temp', type=float, default=0.1)
    parser.add_argument('--start_temp', type=float, default=10.0)
    parser.add_argument('--value_identity_coeff', type=float, default=0.9)
    parser.add_argument('--n_examples', type=int, default=1000)
    parser.add_argument('--num_tokens_per_doc', type=int, default=10240)
    parser.add_argument("--ginc_data_dir", default=None, type=str, required=True,
                        help="The directory that stores GINC datasets.")

    args = parser.parse_args()
    
    # num documents * num tokens per doc / max seq length
    num_lm_dataset_ex = args.n_examples * args.num_tokens_per_doc/args.max_seq_length
    output_dir_name = 'ginc_nhmms{}_nsymbols{}_nvalues{}_nslots{}_{}tasks_bs{}_gpt2_maxseqlen{}_nsamples{}_seed{}'.format(
        args.n_hmms,
        args.n_symbols,
        args.n_values,
        args.n_slots,
        int(math.ceil(num_lm_dataset_ex/args.batch_size)),
        args.batch_size,
        args.max_seq_length,
        args.seed
    )
    
    if args.pretrained:
        output_dir_name += "_pt"
    else:
        output_dir_name += "_rand"
    if args.finetune:
        output_dir_name += "_ft"
    
    args.output_dir += output_dir_name

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'run_args.txt'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=2))
        f.close()
    
    # Load GINC data
    dataset_name = 'GINC_trans{}_start{}_nsymbols{}_nvalues{}_nslots{}_vic{}_nsamples{}_nhmms{}_seed{}'.format(
            args.transition_temp,
            args.start_temp,
            args.n_symbols,
            args.n_values,
            args.n_slots,
            args.value_identity_coeff,
            args.n_examples,
            args.n_hmms,
            args.dataset_seed
        )
    ginc_dataset_dir_path = os.path.join(args.ginc_data_dir, dataset_name)
    assert os.path.isdir(ginc_dataset_dir_path), f'Dataset directory {ginc_dataset_dir_path} does not exist.'
    print(f"GETTING DATA FROM {ginc_dataset_dir_path}")
    tokenizer_name = os.path.join(ginc_dataset_dir_path, 'tokenizer.json')
    train_file = os.path.join(ginc_dataset_dir_path, 'train.json')

    data_files = {}
    assert train_file is not None, f'Train file {train_file} does not exist.'
    data_files["train"] = train_file
    extension = (
        train_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    ds = load_dataset(extension, data_files=data_files, split="train")

    eot = '[endoftext]'
    tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_name,
            bos_token=eot,
            eos_token=eot,
            unk_token=eot)

    if args.pretrained:
        print("USING PRETRAINED MODEL")
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=args.cache_dir if args.cache_dir else None)
        model.config.vocab_size = tokenizer.vocab_size
    else:
        print("USING RANDOM MODEL")
        config = AutoConfig.from_pretrained('gpt2')
        config.vocab_size = tokenizer.vocab_size
        model = GPT2LMHeadModel(config)

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    column_names = ds.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
    )
    
    block_size = args.max_seq_length
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: torch.from_numpy(np.array([t[i : i + block_size] for i in range(0, total_length, block_size)]))
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True,
    ).with_format("torch")
        
    num_tasks = math.ceil(len(lm_datasets) / args.batch_size)
    print("NUM_TASKS:", num_tasks)
    embeddings, losses = [], []
    for task_num in range(num_tasks):
        print(f'--> {task_num=}\n')
        seed = args.seed + task_num
        classifier_opts = {'break_early': args.break_early, "finetune": args.finetune, "seed": seed, "epochs": args.epochs, "task_batch_size": args.batch_size}

        end_index = task_num * args.batch_size + args.batch_size
        if end_index > len(lm_datasets):
            end_index = len(lm_datasets)
        tokenized_task_dataset = lm_datasets.select(range(task_num * args.batch_size, end_index))

        probe_network = model
        embedding, loss = Task2Vec(deepcopy(probe_network), classifier_opts=classifier_opts).embed(tokenized_task_dataset)
        print(f'{embedding.hessian.shape=}')
        embeddings.append(embedding)
        if loss is not None:
            print("LOSS HERE: ", loss)
            losses.append(loss)

        num_tasks_processed = task_num + 1
        np.save(os.path.join(args.output_dir, 'embeddings_' + str(num_tasks_processed) + 'tasks.npy'), embeddings)
        if loss is not None:
            np.save(os.path.join(args.output_dir, 'loss_' + str(num_tasks_processed) + 'tasks.npy'), losses)
        last_num_tasks = num_tasks_processed - 1
        if last_num_tasks > 0:
            last_file = os.path.join(args.output_dir, 'embeddings_' + str(last_num_tasks) + 'tasks.npy')
            if os.path.isfile(last_file):
                os.remove(last_file)
            if loss is not None:
                last_file = os.path.join(args.output_dir, 'loss_' + str(last_num_tasks) + 'tasks.npy')
                if os.path.isfile(last_file):
                    os.remove(last_file)

    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')
    np.save(os.path.join(args.output_dir, 'distance_matrix.npy'), distance_matrix)

    results: dict = {'embeddings': [embed for embed in embeddings],
                     'distance_matrix': distance_matrix,
                     'losses': [loss for loss in losses],
                     "num_tasks": num_tasks}
    np.save(os.path.join(args.output_dir, 'results.npy'), results)
    
if __name__ == '__main__':
    main()