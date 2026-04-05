"""
FL := Formal Language
NL := Natural Language

TODO: add instruction following data & slim pijamas
TODO: hf ds splitting https://stackoverflow.com/questions/78497069/how-to-split-a-hugging-face-dataset-in-streaming-mode-without-loading-it-into-me

dataset.train_test_split(test_size=0.1)
"""
from tqdm import tqdm
import os
from pathlib import Path
import jsonlines

from datasets import load_dataset, concatenate_datasets, interleave_datasets

from utils import raw_dataset_2_lm_data_per_row_mask_excess_eos

# -- FL -> Lean4 TODO: CoqGym, Algebraic Stack, Python, Mirabelle
...

# -- AF data sets TODO: add MMA, ai4m, etc.
...

# -- FL data sets

def get_proofnet_fl_ds(
        path_2_ds: str = 'hoskinson-center/proofnet',
        split='validation',
        verbose: bool = False,
):
    """
    Get proofnet data set that returns all possible formal data. 

    It's ok to include it in both train and val, because, trai fl will do fl->nl so the nl here is fine it's for AIf.
    We can also fit fl which is fine because it can be seen as data we accumulated anyway.
    Since we aren't going to filter data based on the ground truth fl* just generated fl^, then our algorithms 
    shouldn't always choose the pf data anyway (+ the val fl filter data should have fl data from other sources too).
    We are also choosing fl data for AF so we are filtering to get both:
        - fl
        - fl, nl
        obtained through: 
        fl^, nl* <- AIf
        fl^ <- filter(fl^)
        fl^, nl* <- from previous filter step
    val: 185
    test: 186
    tot: 371
    """
    dss = []
    ds = load_dataset(path_2_ds, split)
    all_columns = ds.column_names
    # return all possible formal data
    # only formal statement
    return_only_formal_statement = lambda example: {'text': example['formal_statement']}
    dataset_only_formal_statement = ds.map(return_only_formal_statement, remove_columns=all_columns)
    dss.append(dataset_only_formal_statement)
    # no formal proof statement (sadly)
    ...
    # formal statement with src header
    fl_stmt_plus_src_header = lambda example: {'text': example['src_header'] + '\n\n' + example['formal_statement']}
    dataset_formal_statement_plus_src_header = ds.map(fl_stmt_plus_src_header, remove_columns=all_columns)
    dss.append(dataset_formal_statement_plus_src_header)
    # Interleave
    probabilities = [1.0/len(dss) for _ in dss]  
    print(f'{probabilities=}') if verbose else None
    final_dataset = interleave_datasets(dss, probabilities)
    return final_dataset

def get_minif2f_lean4_fl_data(
        path_2_ds: str = 'hoskinson-center/proofnet',
        split: str = 'validation',
):
    """
    val: 244
    test: 244
    tot: 488
    """
    dss = []
    ds = load_dataset(path_2_ds, split)
    all_columns = ds.column_names
    # return all possible formal data
    # only formal statement
    return_only_formal_statement = lambda example: {'text': example['formal_statement']}
    dataset_only_formal_statement = ds.map(return_only_formal_statement, remove_columns=all_columns)
    dss.append(dataset_only_formal_statement)
    # no formal proof statement (sadly)
    ...
    # formal statement with src header
    fl_stmt_plus_src_header = lambda example: {'text': example['src_header'] + '\n\n' + example['formal_statement']}
    formal_stmt_plus_src_header = ds.map(fl_stmt_plus_src_header, remove_columns=all_columns)
    dss.append(formal_stmt_plus_src_header)
    # Interleave
    probabilities = [1.0/len(dss) for _ in dss]
    final_dataset = interleave_datasets(dss, probabilities)
    return final_dataset

def get_lean_dojo_fl_data(
        path_2_ds: str = 'tasksource/leandojo',  # lean4! see first row https://github.com/leanprover-community/mathlib4
        split: str = 'train',
):
    """
    note: double new line to indicate seperate goals the tactic output, and single line when concatnating ps, tac, ps' 
    train: 87.8K
    val: 2k
    test: 2k
    """
    dss = []
    ds = load_dataset(path_2_ds, split)
    all_columns = ds.column_names
    # return all possible formal data
    # only formal statement
    def return_concat_all_state_before(example):
        traced_tactic: list[dict[str, str]] = example['traced_tactic']
        state_before: str = ''
        # append all before proof states if the tactic returns multiple goals to close  # TODO: is this the best way to append before proof states? For a single goal, it's fine but what to do for multiple goals? best would be to get a seperate extraction for each index...so we'd need several data set for that given the current approach, improve later, for now ignore.
        for before_proof_state, tactic, after_proof_state in traced_tactic:
            state_before += before_proof_state + '\n\n'
        return {'text': state_before}
    ds_state_before = ds.map(return_concat_all_state_before, remove_columns=all_columns)
    dss.append(ds_state_before)
    def return_concat_all_state_after(example):
        traced_tactic: list[dict[str, str]] = example['traced_tactic']
        state_after: str = ''
        # append all after proof states if the tactic returns multiple goals to close  # TODO: is this the best way to append after proof states? For a single goal, it's fine but what to do for multiple goals? best would be to get a seperate extraction for each index...so we'd need several data set for that given the current approach, improve later, for now ignore.
        for before_proof_state, tactic, after_proof_state in traced_tactic:
            state_after += after_proof_state + '\n\n'
        return {'text': state_after}
    ds_state_after = ds.map(return_concat_all_state_after, remove_columns=all_columns) 
    dss.append(ds_state_after)
    # TODO we could have also added ps + ps' without inference step but seems weird
    def return_state_before_after_tactic_state_after(example):
        traced_tactic: list[dict[str, str]] = example['traced_tactic']
        state_before_tactic_state_after: str = ''
        for before_proof_state, tactic, after_proof_state in traced_tactic:
            state_before_tactic_state_after += before_proof_state + '\n' + tactic + '\n' + after_proof_state + '\n\n'
        return {'text': state_before_tactic_state_after}
    ds_state_before_after_tactic_state_after = ds.map(return_state_before_after_tactic_state_after)
    dss.append(ds_state_before_after_tactic_state_after)
    def return_tactic_only(example):
        traced_tactic: list[dict[str, str]] = example['traced_tactic']
        tactic_only: str = ''
        for before_proof_state, tactic, after_proof_state in traced_tactic:
            tactic_only += tactic + '\n\n'
        return {'text': tactic_only}
    # Interleave
    probabilities = [1.0/len(dss) for _ in dss]
    final_dataset = interleave_datasets(dss, probabilities)
    return final_dataset

def algebraic_stack_lean_fl_data(
        path_2_ds: str = 'EleutherAI/proof-pile-2',  # lean4! see first row
        split: str = 'train',
):
    """ TODO: https://huggingface.co/datasets/EleutherAI/proof-pile-2/tree/main/algebraic-stack """
    ...

def get_ntp_fl_data(
        path_2_ds: str = 'l3lab/ntp-mathlib',  # lean4! see first row
        split: str = 'train',
        verbose: bool = False,
):
    """
    Version 1 of:
        https://huggingface.co/datasets/l3lab/ntp-mathlib

    TODO: seperate functions
        https://huggingface.co/datasets/l3lab/ntp-mathlib-instruct-st
        https://huggingface.co/datasets/l3lab/ntp-mathlib-instruct-context
    """
    dss = []
    ds = load_dataset(path_2_ds, split)
    all_columns = ds.column_names
    # return all possible formal language data
    # decl
    decl_as_text = lambda example: {'text': example['decl']}
    decl_dataset = ds.map(decl_as_text, remove_columns=all_columns)
    dss.append(decl_dataset)
    # declUpToTactic
    decl_up_to_tactic_as_text = lambda example: {'text': example['declUpToTactic']}
    decl_up_to_tactic_dataset = ds.map(decl_up_to_tactic_as_text, remove_columns=all_columns)
    dss.append(decl_up_to_tactic_dataset)
    # declupToTactic + nextTactic
    decl_up_to_tactic_next_tactic_as_text = lambda example: {'text': example['declUpToTactic'] + '\n' + example['nextTactic']}
    decl_up_to_tactic_next_tactic_dataset = ds.map(decl_up_to_tactic_next_tactic_as_text, remove_columns=all_columns)
    dss.append(decl_up_to_tactic_next_tactic_dataset)
    # (clean) srcUpToTactic TODO: we need to clean this up with a regex but unsure which regex to use
    def clean_src_up_to_tactic(example):
        import re
        cleaned_text = re.sub(r'(?s)/- ?.*?-/(\s)+', '', example['srcUpToTactic'])
        return {'text': cleaned_text}
    src_up_to_tactic_dataset = ds.map(clean_src_up_to_tactic, remove_columns=all_columns)
    dss.append(src_up_to_tactic_dataset)
    # (clean) srcUpToTactic + nextTactic
    def clean_src_up_to_tactic_next_tactic(example):
        import re
        cleaned_text = re.sub(r'(?s)/- ?.*?-/(\s)+', '', example['srcUpToTactic'])
        return {'text': cleaned_text + '\n' + example['nextTactic']}
    src_up_to_tactic_next_tactic_dataset = ds.map(clean_src_up_to_tactic_next_tactic, remove_columns=all_columns)
    dss.append(src_up_to_tactic_next_tactic_dataset)
    # (clean) srcUpToTactic + state + nextTactic
    def clean_src_up_to_tactic_state_next_tactic(example):
        import re
        cleaned_text = re.sub(r'(?s)/- ?.*?-/(\s)+', '', example['srcUpToTactic'])
        return {'text': cleaned_text + '\n' + example['state'] + '\n' + example['nextTactic']}
    src_up_to_tactic_state_next_tactic_dataset = ds.map(clean_src_up_to_tactic_state_next_tactic, remove_columns=all_columns)
    dss.append(src_up_to_tactic_state_next_tactic_dataset)
    # file_tag TODO decide later
    ...
    # nextTactic
    next_tactic_as_text = lambda example: {'text': example['nextTactic']}
    next_tactic_dataset = ds.map(next_tactic_as_text, remove_columns=all_columns)
    dss.append(next_tactic_dataset)
    # state 
    state_as_text = lambda example: {'text': example['state']}
    state_dataset = ds.map(state_as_text, remove_columns=all_columns)
    dss.append(state_dataset)
    # state + nextTactic
    state_next_tactic_as_text = lambda example: {'text': example['state'] + '\n' + example['nextTactic']}
    state_next_tactic_dataset = ds.map(state_next_tactic_as_text, remove_columns=all_columns)
    dss.append(state_next_tactic_dataset)
    # decl + state + nextTactic TODO: should we add text to say something like "The following is a declaration {decl}, state {st}, and next tactic {next_tac}"?
    decl_state_next_tactic_as_text = lambda example: {'text': example['decl'] + '\n' + example['state'] + '\n' + example['nextTactic']}
    decl_state_next_tactic_dataset = ds.map(decl_state_next_tactic_as_text, remove_columns=all_columns)
    dss.append(decl_state_next_tactic_dataset)
    # declUpToTactic + state + nextTactic TODO: should we add text to say something like "The following is a declaration up to tactic {decl}, state {st}, and next tactic {next_tac}"?
    decl_up_to_tactic_state_next_tactic_as_text = lambda example: {'text': example['declUpToTactic'] + '\n' + example['state'] + '\n' + example['nextTactic']}
    decl_up_to_tactic_state_next_tactic_dataset = ds.map(decl_up_to_tactic_state_next_tactic_as_text, remove_columns=all_columns)
    dss.append(decl_up_to_tactic_state_next_tactic_dataset)
    # Interleave
    probabilities = [1.0/len(dss) for _ in dss]
    print(f'{probabilities=}') if verbose else None
    final_dataset = interleave_datasets(dss, probabilities)
    return final_dataset

# -- NL data sets

def get_hf_math_ds(
        path_2_ds: str = '~/gold-ai-olympiad/data/MATH/train.jsonl', 
        # path_2_ds: str = '~/gold-ai-olympiad/data/debug/mathematica_debug.jsonl', 
        split: str = 'train',
        ):
    path_2_ds = os.path.expanduser(path_2_ds)
    return get_hf_mathematica_ds(path_2_ds, split)

def get_hf_khan_ds(
        path_2_ds: str = '~/gold-ai-olympiad/data/amps/khan/train.jsonl', 
        # path_2_ds: str = '~/gold-ai-olympiad/data/debug/khan_debug.jsonl', 
        split: str = 'train',
        ):
    path_2_ds = os.path.expanduser(path_2_ds)
    dataset = load_dataset('json', data_files=[path_2_ds], split=split)
    all_columns = dataset.column_names
    
    problem_as_text = lambda example: {'text': example['problem']}
    problem_dataset = dataset.map(problem_as_text, remove_columns=all_columns)
    
    hints_as_text = lambda example: {'text': ' '.join(example['hints'])}
    hints_dataset = dataset.map(hints_as_text, remove_columns=all_columns)
    
    problem_hints_as_text = lambda example: {'text': example['problem'] + ' ' + ' '.join(example['hints'])}
    combined_dataset = dataset.map(problem_hints_as_text, remove_columns=all_columns)
    
    final_dataset = concatenate_datasets([problem_dataset, hints_dataset, combined_dataset])
    return final_dataset

def get_hf_mathematica_ds(
        path_2_ds: str = '~/gold-ai-olympiad/data/amps/mathematica/train.jsonl', 
        # path_2_khan_ds: str = '~/gold-ai-olympiad/data/debug/mathematica_debug.jsonl', 
        split: str = 'train',
        ):
    path_2_ds = os.path.expanduser(path_2_ds)
    dataset = load_dataset('json', data_files=[path_2_ds], split=split)
    all_columns = dataset.column_names
    
    problem_as_text = lambda example: {'text': example['problem']}
    problem_dataset = dataset.map(problem_as_text, remove_columns=all_columns)
    
    hints_as_text = lambda example: {'text': ' '.join(example['solution'])}
    hints_dataset = dataset.map(hints_as_text, remove_columns=all_columns)
    
    problem_hints_as_text = lambda example: {'text': example['problem'] + ' ' + ' '.join(example['solution'])}
    combined_dataset = dataset.map(problem_hints_as_text, remove_columns=all_columns)
    
    final_dataset = concatenate_datasets([problem_dataset, hints_dataset, combined_dataset])
    return final_dataset

def get_hf_olypiad_bench_maths_tp_ds(
        path_2_ds: list[str] = [
            '~/gold-ai-olympiad/data/OlympiadBench_Dataset/data_math/TP_MM_maths_en_COMP.json',
            '~/gold-ai-olympiad/data/OlympiadBench_Dataset/data_math/TP_TO_maths_en_MATH.json',
        ],
        split: str = 'train',
        verbose: bool = False,
    ):
    """ NL Maths Theorem Proving (TP) datasets. """
    path_2_dss = [os.path.expanduser(p) for p in path_2_ds]
    dataset = load_dataset('json', data_files=path_2_dss, split=split)
    all_columns = dataset.column_names
    dss = []
    # return all possible natural language data
    # problems
    problem_as_text = lambda example: {'text': example['question']}
    problem_dataset = dataset.map(problem_as_text, remove_columns=all_columns)
    dss.append(problem_dataset)
    # solutions
    soln_as_text = lambda example: {'text': example['solution'][0]}  # TODO what to do with multiple solutions?
    soln_dataset = dataset.map(soln_as_text, remove_columns=all_columns)
    dss.append(soln_dataset)
    # problems + solutions  # TODO what to do with multiple solutions?
    prob_soln_as_text = lambda example: {'text': f"Problem: {example['question']}\nSolution:{example['solution'][0]}"}
    # def prob_solns_as_text(example):
    #     return {'text': f"Problem: {example['question']}\nSolution:{example['solution]}"}
    prob_soln_dataset = dataset.map(prob_soln_as_text, remove_columns=all_columns)
    dss.append(prob_soln_dataset)
    # Interleave
    probabilities = [1.0/len(dss) for _ in dss]
    print(f'{probabilities=}') if verbose else None
    final_dataset = interleave_datasets(dss, probabilities)
    return final_dataset

def get_hf_olypiad_bench_physics_tp_ds(
        path_2_ds: list[str] = [
            '~/gold-ai-olympiad/data/OlympiadBench_Dataset/data_physics/TP_MM_physics_en_COMP.json',
            '~/gold-ai-olympiad/data/OlympiadBench_Dataset/data_physics/TP_TO_physics_en_MATH.json',
        ],
        split: str = 'train',
        verbose: bool = False,
    ):
    """ NL Physics Theorem Proving (TP) datasets. """
    path_2_dss = [os.path.expanduser(p) for p in path_2_ds]
    dataset = load_dataset('json', data_files=path_2_dss, split=split)
    all_columns = dataset.column_names
    dss = []
    # return all possible natural language data
    # questions
    question_as_text = lambda example: {'text': example['question']}
    question_dataset = dataset.map(question_as_text, remove_columns=all_columns)
    dss.append(question_dataset)
    # context 
    context_as_text = lambda example: {'text': example['context']}
    context_dataset = dataset.map(context_as_text, remove_columns=all_columns)
    dss.append(context_dataset)
    # context + question
    context_question_as_text = lambda example: {'text': f"Context: {example['context']}\nQuestion: {example['question']}"}
    context_question_dataset = dataset.map(context_question_as_text, remove_columns=all_columns)
    dss.append(context_question_dataset)
    # question + 1st solution
    question_soln_as_text = lambda example: {'text': f"Question: {example['question']}\nSolution: {example['solution'][0]}"}
    question_soln_dataset = dataset.map(question_soln_as_text, remove_columns=all_columns)
    dss.append(question_soln_dataset)
    # context + question + 1st solution
    context_question_soln_as_text = lambda example: {'text': f"Context: {example['context']}\nQuestion: {example['question']}\nSolution: {example['solution'][0]}"}
    context_question_soln_dataset = dataset.map(context_question_soln_as_text, remove_columns=all_columns)
    dss.append(context_question_soln_dataset)
    # Interleave
    probabilities = [1.0/len(dss) for _ in dss]
    print(f'{probabilities=}') if verbose else None
    final_dataset = interleave_datasets(dss, probabilities)
    return final_dataset        

# -- Get all datasets as a single dataset
# vX to indicate version X and track experiments improvements

def get_all_fl_datasets_as_single_ds_v1(
        tokenizer,
        split: str = 'train',
        verbose: bool = False,
        max_length: int = 1024,
):
    """ Get all FL datasets as a single dataset """
    train_datasets = []
    # Proofnet, Minif2f, LeanDojo
    ds_proofnet = get_proofnet_fl_ds(split=split)
    train_datasets.append(ds_proofnet)
    ds_minif2f = get_minif2f_lean4_fl_data(split=split)
    train_datasets.append(ds_minif2f)
    ds_lean_dojo = get_lean_dojo_fl_data(split=split)
    train_datasets.append(ds_lean_dojo)
    # Interleave
    probabilities = [1.0/len(train_datasets) for _ in train_datasets]  
    print(f'{probabilities=}') if verbose else None
    raw_train_datasets = interleave_datasets(train_datasets, probabilities)
    # Preprocess data
    lm_train_dataset = raw_dataset_2_lm_data_per_row_mask_excess_eos(raw_train_datasets, tokenizer, max_length)
    return lm_train_dataset

def get_all_nl_datasets_as_single_ds_v1(
        tokenizer,
        split: str = 'train',
        verbose: bool = False,
        max_length: int = 1024,
):  
    train_datasets = []
    # MATH, Khan, Mathematica
    ds_math = get_hf_math_ds(split=split)
    train_datasets.append(ds_math)
    ds_khan = get_hf_khan_ds(split=split)
    train_datasets.append(ds_khan)
    ds_mathematica = get_hf_mathematica_ds(split=split)
    train_datasets.append(ds_mathematica)
    # Olympiad Bench Maths TP, Olympiad Bench Physics TP
    ds_olypiad_bench_maths_tp = get_hf_olypiad_bench_maths_tp_ds(split=split)
    train_datasets.append(ds_olypiad_bench_maths_tp)
    ds_olypiad_bench_physics_tp = get_hf_olypiad_bench_physics_tp_ds(split=split)
    train_datasets.append(ds_olypiad_bench_physics_tp)
    # TODO Putnam TP
    # Interleave
    probabilities = [1.0/len(train_datasets) for _ in train_datasets]  
    print(f'{probabilities=}') if verbose else None
    raw_train_datasets = interleave_datasets(train_datasets, probabilities)
    # Preprocess data
    lm_train_dataset = raw_dataset_2_lm_data_per_row_mask_excess_eos(raw_train_datasets, tokenizer, max_length)
    return lm_train_dataset

# -- Main

def main(
        verbose: bool = True,
):
    """"""
    # - Get raw train data set

    # lm_train_dataset = raw_dataset_2_lm_data(raw_train_datasets, tokenizer, block_siz

    # -- Get data sets
    
if __name__ == '__main__':
    import fire
    import time
    start = time.time()
    # fire.Fire(main)
    main()
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")