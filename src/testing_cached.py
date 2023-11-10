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
    arg1 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    arg2 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    arg3 = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    pubmedT = load_dataset('SudharsanSundar/PileSubsets', 'pubmed', split='train').with_format('torch')
    pubmedT = pubmedT.to_iterable_dataset()
    USPTOT = load_dataset('SudharsanSundar/PileSubsets', 'uspto', split='train').with_format('torch')
    USPTOT = USPTOT.to_iterable_dataset()
    comboT = interleave_datasets([pubmedT, USPTOT], probabilities=[0.4568, 0.5431])
    pile = load_dataset('monology/pile', split='train').with_format('torch')
    pile = pile.to_iterable_dataset()
    wikitext = load_dataset('wikitext', 'wikitext-103-v1', split='validation').with_format('torch')
    wikitext = wikitext.to_iterable_dataset()
    opensubs = load_dataset('suolyer/pile_opensubtitles', split='validation').with_format('torch')
    opensubs = opensubs.to_iterable_dataset()
    openwebtext2 = load_dataset('suolyer/pile_openwebtext2', split='validation').with_format('torch')
    openwebtext2 = openwebtext2.to_iterable_dataset()
    nihexporter = load_dataset('suolyer/pile_nih-exporter', split='validation').with_format('torch')
    nihexporter = nihexporter.to_iterable_dataset()
    hnews = load_dataset('suolyer/pile_hackernews', split='validation').with_format('torch')
    hnews = hnews.to_iterable_dataset()
    tinystories = load_dataset('roneneldan/TinyStories', split='validation').with_format('torch')
    tinystories = tinystories.to_iterable_dataset()
    pubmedV = load_dataset('suolyer/pile_pubmed-abstracts', split='validation').with_format('torch')
    pubmedV = pubmedV.to_iterable_dataset()
    usptoV = load_dataset('suolyer/pile_uspto', split='validation').with_format('torch')
    usptoV = usptoV.to_iterable_dataset()

    dataset_list = [pubmedT, USPTOT, comboT, pile, wikitext, opensubs, openwebtext2, nihexporter, hnews, tinystories, pubmedV, usptoV]
    names_list = ["pubmedT", "USPTOT", "comboT", "pile", "wikitext", "opensubs", "openwebtext2", "nihexporter", "hnews", "tinystories", "pubmedV", "usptoV"]

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
                                                  batch_size=512,
                                                  num_batches=arg3,
                                                  buffer_size=500000,
                                                  shuffle=True,
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
