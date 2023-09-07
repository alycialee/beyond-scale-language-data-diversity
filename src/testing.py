import time
import datetime
import sys

from diversity.task2vec import Task2Vec
from diversity import task_similarity
from diversity.div_coeff import cross_diversity_coefficient

from pathlib import Path

import torch
import torch.nn as nn

from alignment.align import alignment_with_diversity_coefficient

from datasets import load_dataset, interleave_datasets
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def preprocess(examples):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")


def map_func(batch):
    column_names = next(iter(batch)).keys()
    print(f'Deleting these columns: {column_names=}')
    remove_columns = column_names

    return batch.map(preprocess, batched=True, remove_columns=remove_columns)


def main():
    arg1 = int(sys.argv[1]) if len(sys.argv) > 1 else None
    arg2 = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if arg1 is None and arg2 is None:
        arg1 = 0
        arg2 = 1

    # pubmedT = load_dataset('EleutherAI/pile', 'pubmed', streaming=True, split='train').with_format('torch')
    # USPTOT = load_dataset('EleutherAI/pile', 'USPTO', streaming=True, split='train').with_format('torch')
    # comboT = interleave_datasets([pubmedT, USPTOT], probabilities=[0.4568, 0.5431])
    pile = load_dataset('monology/pile', streaming=True, split='train').with_format('torch')
    wikitext = load_dataset('wikitext', 'wikitext-103-v1', streaming=True, split='validation').with_format('torch')
    opensubs = load_dataset('suolyer/pile_opensubtitles', streaming=True, split='validation').with_format('torch')
    openwebtext2 = load_dataset('suolyer/pile_openwebtext2', streaming=True, split='validation').with_format('torch')
    nihexporter = load_dataset('suolyer/pile_nih-exporter', streaming=True, split='validation').with_format('torch')
    hnews = load_dataset('suolyer/pile_hackernews', streaming=True, split='validation').with_format('torch')
    tinystories = load_dataset('roneneldan/TinyStories', streaming=True, split='validation').with_format('torch')
    pubmedV = load_dataset('suolyer/pile_pubmed-abstracts', streaming=True, split='validation').with_format('torch')
    usptoV = load_dataset('suolyer/pile_uspto', streaming=True, split='validation').with_format('torch')

    # dataset_list = [pubmedT, USPTOT, comboT, pile, wikitext, opensubs, openwebtext2, nihexporter, hnews, tinystories, pubmedV, usptoV]
    # names_list = ["pubmedT", "USPTOT", "comboT", "pile", "wikitext", "opensubs", "openwebtext2", "nihexporter", "hnews", "tinystories", "pubmedV", "usptoV"]
    dataset_list = [pile, wikitext, opensubs, openwebtext2, nihexporter, hnews, tinystories, pubmedV, usptoV]
    names_list = ["pile", "wikitext", "opensubs", "openwebtext2", "nihexporter", "hnews", "tinystories", "pubmedV", "usptoV"]

    dataset1 = dataset_list[arg1]
    dataset2 = dataset_list[arg2]

    print('These are the arguments:', arg1, "/", arg2, '\nAnd these are the corresponding datasets:', names_list[arg1], "/", names_list[arg2])

    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    probe_network = probe_network.to(device)

    output = alignment_with_diversity_coefficient(dataset1,
                                                  dataset2,
                                                  map_func,
                                                  map_func,
                                                  probe_network,
                                                  batch_size=3,
                                                  num_batches=3,
                                                  buffer_size=1000,
                                                  shuffle=False,
                                                  verbose=True)
    for key in output:
        print(key, output[key], '\n\n')

    print('\ncross div coeff', output['cross_div_coeff'])
    print('\ncross div coeff ci (really stdev)', output['cross_div_coeff_ci'])
    print('\ndistance matrix mean', output['distance_matrix'])
    print('\nnum batches', output['num_batches'])
    print('\ncross_align', output['cross_align'])



if __name__ == "__main__":
    main()
