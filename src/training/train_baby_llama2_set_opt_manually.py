"""
https://huggingface.co/docs/transformers/v4.29.0/perf_train_gpu_one

Inspiration:
- ref: SO accelerate + trainer: https://stackoverflow.com/questions/76675018/how-does-one-use-accelerate-with-the-hugging-face-hf-trainer
- ref: The unreasonable effectiveness of few-shot learning for machine translation https://arxiv.org/abs/2302.01398
- ref: colab: https://colab.research.google.com/drive/1io951Ex17-6OUaogCo7OiR-eXga_oUOH?usp=sharing
- ref: SO on collate: https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999

- qlora https://github.com/artidoro/qlora/blob/main/scripts/finetune_llama2_guanaco_7b.sh, 
- https://github.com/artidoro/qlora/blob/main/qlora.py

export CUDA_VISIBLE_DEVICES=6
"""
from pathlib import Path
from typing import Callable
import datasets
from datasets import load_dataset, interleave_datasets
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
import math

import sys
from training.reinit_and_smaller_llama2 import get_deafult_smallest_baby_llama2_v1_36m_0p036b, get_weight_norms, reinitialize_weights_gpt_neox_20B_inspired_4_llama2
sys.path = [''] + sys.path
from training.utils import eval_hf, get_column_names, get_data_from_hf_dataset, group_texts, raw_dataset_2_lm_data
from training.optim_utils import get_paged_adamw_32bit_manual

# -- Experiments 

def train():
    """
    I decided to make the string data close to context length of llama2 7B 4096 tokens.
    So if any string is shorter, the tokenize will padd it according to Claude.
    
    """
    # feel free to move the import statements if you want, sometimes I like everything in one place so I can easily copy-paste it into a script
    import datetime
    from pathlib import Path
    import datasets
    from datasets import load_dataset, interleave_datasets
    import torch
    import transformers
    from transformers import PreTrainedTokenizer
    from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
    import random
    import math
    import os
    torch.cuda.empty_cache()
    # buffer_size = 500_000  # can't remember what this was for and doesn't seem to be anywhere
    probabilities = []
    data_mixture_name = None
    streaming = True
    data_files = [None]
    seed = 0
    split = 'train'
    max_length = 1024  # gpt2 context length
    shuffle = False
    report_to = 'none'  # safest default
    # CHUNK_SIZE = 16_896  # approximately trying to fill the llama2 context length of 4096
    batch_size = 2
    gradient_accumulation_steps = 2
    num_epochs = 1
    num_tokens_trained = None
    num_batches=1
    optim='paged_adamw_32bit'
    learning_rate=1e-5
    warmup_ratio=0.01
    weight_decay=0.01
    lr_scheduler_type='constant_with_warmup'
    lr_scheduler_kwargs={}

    # -- Setup wandb
    import wandb
    # - Dryrun
    mode = 'dryrun'; seed = 0; report_to = 'none'

    # - Online (real experiment)
    mode = 'online'; seed = 0; report_to = 'wandb'

    # - train data sets
    # path, name, data_files, split = ['c4'], ['en'], [None], ['train']
    # path, name, data_files, split = ['UDACA/PileSubsets'], ['uspto'], [None], ['train']
    path, name, data_files, split = ['UDACA/PileSubsets'], ['pubmed'], [None], ['train']
    # path, name, data_files, split = ['UDACA/PileSubsets', 'UDACA/PileSubsets'], ['uspto', 'pubmed'], [None, None], ['train', 'train']
    # - models
    # pretrained_model_name_or_path = 'gpt2'
    # pretrained_model_name_or_path = 'meta-llama/Llama-2-7b-hf'
    # pretrained_model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'
    # pretrained_model_name_or_path = 'meta-llama/Llama-2-13b-hf'
    # pretrained_model_name_or_path = 'meta-llama/Llama-2-70b-hf'
    # pretrained_model_name_or_path = 'mistralai/Mistral-7B-v0.1'
    pretrained_model_name_or_path = 'baby_llama2_v1'
    # - important training details or it wont run, mem issues maybe
    max_steps = 300 # <- CHANGE THIS  11 days with baby llama2 v1 36m 1, 32
    # max_steps = 19_073 # <- CHANGE THIS  11 days with baby llama2 v1 36m 1, 32
    # max_steps = 866 # <- CHANGE THIS 12hs with with baby llama2 v1 36m 1, 32
    # max_steps = 1_761 # <- CHANGE THIS 12hs with with baby llama2 v1 36m 5, 6 0.2168M tokens
    # max_steps = 306_000 # <- CHANGE THIS 12hs with with baby llama2 v1 36m 1, 32 35.1 tokens
    max_length = 4096
    num_batches=1
    # single gpu
    # batch_size, gradient_accumulation_steps = 1, 32  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    # batch_size, gradient_accumulation_steps = 6, 5  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    # batch_size, gradient_accumulation_steps = 5, 6  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    # batch_size, gradient_accumulation_steps = 4, 6  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    batch_size, gradient_accumulation_steps = 4, 8  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    learning_rate=1e-4
    learning_rate=1e-5
    # learning_rate=5e-4
    # learning_rate=1e-6
    # optim='adamw'
    optim='paged_adamw_32bit'
    # optim = 'adafactor'
    weight_decay=0.1
    warmup_ratio=0.01
    # lr_scheduler_type='cosine'
    # lr_scheduler_type='constant_with_warmup'
    lr_scheduler_type='cosine_with_warmup'
    # lr_scheduler_kwargs={},  # ref: https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/optimizer_schedules#transformers.SchedulerType 
    # -- multiple gpus 3 4096 context len
    # batch_size, gradient_accumulation_steps = 4, 8  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    # gradient_checkpointing = False
    gradient_checkpointing = True
    print(f'{batch_size=} {gradient_accumulation_steps=} {gradient_checkpointing=} {num_epochs=}')
    # -- wandb
    num_tokens_trained = max_steps * batch_size * max_length * num_batches 
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    run_name = f'beyond scale: {path} ({today=} ({name=}) {data_mixture_name=} {probabilities=} {pretrained_model_name_or_path=} {data_files=} {max_steps=} {batch_size=} {num_tokens_trained=} {gradient_accumulation_steps=} {optim=} {learning_rate=} {max_length=} {weight_decay=} {warmup_ratio=})'
    print(f'\n---> {run_name=}\n')

    # - Init wandb
    debug: bool = mode == 'dryrun'  # BOOL, debug?
    run = wandb.init(mode=mode, project="beyond-scale", name=run_name, save_code=True)
    # wandb.config.update({"num_batches": num_batches, "path": path, "name": name, "today": today, 'probabilities': probabilities, 'batch_size': batch_size, 'debug': debug, 'data_mixture_name': data_mixture_name, 'streaming': streaming, 'data_files': data_files, 'seed': seed, 'pretrained_model_name_or_path': pretrained_model_name_or_path})
    wandb.config.update({"path": path, "name": name, "today": today, 'probabilities': probabilities, 'batch_size': batch_size, 'debug': debug, 'data_mixture_name': data_mixture_name, 'streaming': streaming, 'data_files': data_files, 'seed': seed, 'pretrained_model_name_or_path': pretrained_model_name_or_path, 'num_epochs': num_epochs, 'gradient_accumulation_steps': gradient_accumulation_steps})
    # run.notify_on_failure() # https://community.wandb.ai/t/how-do-i-set-the-wandb-alert-programatically-for-my-current-run/4891
    print(f'{debug=}')
    print(f'{wandb.config=}')

    # -- Load model and tokenizer  
    print(f'{pretrained_model_name_or_path=}')
    if pretrained_model_name_or_path == 'gpt2':
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        print(f'{tokenizer.eos_token=}')
        print(f'{ tokenizer.eos_token_id=}')
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        block_size: int = tokenizer.model_max_length
    elif 'Llama-2' in pretrained_model_name_or_path or 'Mistral' in pretrained_model_name_or_path:
        # - llama2
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
        # bf16 or fp32
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        # get model
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            # quantization_config=quantization_config,
            # device_map=device_map,  # device_map = None  https://github.com/huggingface/trl/blob/01c4a35928f41ba25b1d0032a085519b8065c843/examples/scripts/sft_trainer.py#L82
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_auth_token=True,
        )
        # https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L347C13-L347C13
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            # cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=False, # Fast tokenizer giving issues.
            # tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
            # tokenizer_type='llama',
            trust_remote_code=True,
            use_auth_token=True,
            # token=token,  # load from cat keys/brandos_hf_token.txt if you want to load it in python and not run huggingface-cli login
        )
        # - Ensure padding token is set TODO: how does this not screw up the fine-tuning? e.g., now model doesn't learn to predict eos since it's padded our by mask, ref: https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f'{tokenizer.pad_token=}')
        print(f'{tokenizer.eos_token=}')
        print(f'{ tokenizer.eos_token_id=}')
        # get context length for setting max length for training
        if hasattr(model.config, "context_length"):
            print("Context length:", model.config.context_length)
            max_length = model.config.context_length
        else:
            # CHUNK_SIZE = 16_896  # approximately trying to fill the llama2 context length of 4096
            max_length = 4096
        block_size: int = 4096
        print(f'{max_length=}')
    elif 'baby_llama2_v1' in pretrained_model_name_or_path:
        model = get_deafult_smallest_baby_llama2_v1_36m_0p036b()
        reinitialize_weights_gpt_neox_20B_inspired_4_llama2(model, L=max_length)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side="right", use_fast=False, trust_remote_code=True, use_auth_token=True)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 else torch.float32 # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        model = model.to(torch_dtype)
        block_size: int = max_length
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    print(f"Total weight norm: {get_weight_norms(model)=}")
    print(f'{torch.cuda.device_count()=} (makes sure GPUs are visible and accesible to Pytorch.)')
    print(f'Model is currently on: {next(iter(model.parameters())).device=}')
    print(f'Model is currently on: {next(iter(model.parameters())).dtype=}')
    
    # --- Load datasets
    # -- Get train data set
    # - Load interleaved combined datasets
    # train_datasets = [load_dataset(path, name, streaming=True, split="train").with_format("torch") for path, name in zip(path, name)]
    train_datasets = [load_dataset(path, name, data_files=data_file, streaming=streaming, split=split).with_format("torch") for path, name, data_file, split in zip(path, name, data_files, split)]
    probabilities = [1.0/len(train_datasets) for _ in train_datasets]  
    # - Get raw train data set
    raw_train_datasets = interleave_datasets(train_datasets, probabilities)
    remove_columns = get_column_names(raw_train_datasets)  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    # - Get tokenized train data set
    # Note: Setting `batched=True` in the `dataset.map` function of Hugging Face's datasets library processes the data in batches rather than one item at a time, significantly speeding up the tokenization and preprocessing steps.
    tokenize_function = lambda examples: tokenizer(examples["text"])
    tokenized_train_datasets = raw_train_datasets.map(tokenize_function, batched=True, remove_columns=remove_columns)
    _group_texts = lambda examples : group_texts(examples, block_size)
    # - Get actual data set for lm training (in this case each seq is of length block_size, no need to worry about pad = eos since we are filling each sequence)
    lm_train_dataset = tokenized_train_datasets.map(_group_texts, batched=True)
    batch = get_data_from_hf_dataset(lm_train_dataset, streaming=streaming, batch_size=batch_size)
    print(f'{len(next(iter(batch))["input_ids"])=}')
    assert all(len(data_dict['input_ids']) == block_size for data_dict in iter(batch)), f'Error, some seq in batch are not of length {block_size}'
    train_dataset = lm_train_dataset

    # -- max steps manually decided depending on how many tokens we want to train on
    per_device_train_batch_size = batch_size
    print(f'{per_device_train_batch_size=}')

    print(f'{num_epochs=} {max_steps=}')

    # -- Training arguments and trainer instantiation ref: https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments
    output_dir = Path(f'~/data/results_{today}/').expanduser() if not debug else Path(f'~/data/results/').expanduser()
    # output_dir = '.'
    # print(f'{debug=} {output_dir=} \n {report_to=}')
    training_args = TrainingArguments(
        output_dir=output_dir,  # The output directory where the model predictions and checkpoints will be written.
        # output_dir='.',  # The output directory where the model predictions and checkpoints will be written.
        # num_train_epochs = num_train_epochs, 
        max_steps=max_steps,  # TODO: hard to fix, see above
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing = gradient_checkpointing,  # TODO depending on hardware set to true?
        optim=optim,
        warmup_steps=int(max_steps*warmup_ratio),  # TODO: once real training starts we can select this number for llama v2, what does llama v2 do to make it stable while v1 didn't?
        warmup_ratio=warmup_ratio,  # copying alpaca for now, number of steps for a linear warmup, TODO once real training starts change? 
        # weight_decay=0.01,  # TODO once real training change?
        weight_decay=weight_decay,  # TODO once real training change?
        learning_rate = learning_rate,  # TODO once real training change? anything larger than -3 I've had terrible experiences with
        max_grad_norm=1.0, # TODO once real training change?
        # lr_scheduler_type=lr_scheduler_type,  # TODO once real training change? using what I've seen most in vision 
        # lr_scheduler_kwargs=lr_scheduler_kwargs,  # ref: https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/optimizer_schedules#transformers.SchedulerType 
        logging_dir=Path('~/data/maf/logs').expanduser(),
        # save_steps=4000,  # alpaca does 2000, other defaults were 500
        save_steps=max_steps//3,  # alpaca does 2000, other defaults were 500
        # save_steps=1,  # alpaca does 2000, other defaults were 500
        # logging_steps=250,
        # logging_steps=50,  
        logging_first_step=True,
        # logging_steps=3,
        logging_steps=1,
        remove_unused_columns=False,  # TODO don't get why https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999 , https://claude.ai/chat/475a4638-cee3-4ce0-af64-c8b8d1dc0d90
        report_to=report_to,  # change to wandb!
        fp16=False,  # never ever set to True
        bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    )
    print(f'{pretrained_model_name_or_path=}\n{optim=}\n{learning_rate=}')

    # -- Get Optimizer & Scheduler
    # - Get Optimizer
    if optim == 'paged_adamw_32bit':
        assert training_args.optim == optim, f'Error, training_args.optim={training_args.optim} != optim={optim}'
        _trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        optimizer = get_paged_adamw_32bit_manual(_trainer)
    elif optim == 'adamw':
        assert training_args.optim == optim, f'Error, training_args.optim={training_args.optim} != optim={optim}'
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print(f'{optim=} {training_args.optim=}')
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f'{optimizer=}')
    # - Get Scheduler
    print(f'{lr_scheduler_type=}')
    if lr_scheduler_type == 'cosine_with_warmup':
        # trainer also has a get_scheduler func but opted not to use it since it required lr_scheduler_kwargs and I prefered to create the scheduler in one place where I can see the kwargs being set then reinit trainer with opt, sch. But if trianer had side effects my approach might not work. 
        num_warmup_steps = int(max_steps*warmup_ratio)
        num_training_steps = max_steps
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_steps,
        )
        print(f'{num_warmup_steps=} {num_training_steps=}')
    else:
        lr_scheduler = None
    print(f'{lr_scheduler=}')

    # -- Init Trainer
    trainer = Trainer(
        model=model,
        args=training_args,  
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
    )

    # -- Train
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        print(f"CUDA_VISIBLE_DEVICES = {cuda_visible_devices}")
    trainer.train()
    trainer.save_model(output_dir=output_dir)  # TODO is this really needed? https://discuss.huggingface.co/t/do-we-need-to-explicity-save-the-model-if-the-save-steps-is-not-a-multiple-of-the-num-steps-with-hf/56745

    # -- Evaluation, NOTE: we are evaluating at the end not during training
    # - Evaluate model on OpenWebtext
    print('---- Evaluate model on OpenWebtext')
    streaming = True
    max_eval_samples = 1024
    path, name, split = 'suolyer/pile_openwebtext2', None, 'validation'  # the one sudharsan used
    eval_dataset = load_dataset(path, name, streaming=streaming, split=split).with_format("torch") 
    eval_dataset1 = raw_dataset_2_lm_data(eval_dataset, tokenizer, block_size)
    eval_batch1 = eval_dataset1.take(max_eval_samples)
    print(f'Saving eval results at: {output_dir=}') # The output directory where the model predictions and checkpoints will be written.
    eval_args = TrainingArguments(output_dir=output_dir, fp16=False, bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8)
    trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_batch1)
    eval_hf(trainer)
    # - Evaluate on C4
    print('---- Evaluate model on C4')
    streaming = True
    max_eval_samples = 1024
    path, name, split = 'c4', 'en', 'validation' 
    eval_dataset = load_dataset(path, name, streaming=streaming, split=split).with_format("torch") 
    eval_dataset2 = raw_dataset_2_lm_data(eval_dataset, tokenizer, block_size)
    eval_batch2 = eval_dataset2.take(max_eval_samples)
    print(f'Saving eval results at: {output_dir=}') # The output directory where the model predictions and checkpoints will be written.
    eval_args = TrainingArguments(output_dir=output_dir, fp16=False, bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8)
    trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_batch2)
    eval_hf(trainer)
    # - Evluate on whole datasets
    print('---- Evaluate model on Whole OpenWebtext')
    trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_dataset1)
    eval_hf(trainer)
    print('---- Evaluate model on Whole C4')
    trainer = Trainer(model=model, args=eval_args, train_dataset=None, eval_dataset=eval_dataset2)
    # eval_hf(trainer)
    print('Done!\a')

def main():  
    """Since accelerate config wants this, main_training_function: main"""
    train()

# -- Run __main__

if __name__ == '__main__':
    print(f'\n\n\n------------------- Running {__file__} -------------------')
    # -- Run tests and time it
    import time
    time_start = time.time()
    # -- Run tests
    main()
    # -- End tests, report how long it took in seconds, minutes, hours, days
    print(f'Time it took to run {__file__}: {time.time() - time_start} seconds, {(time.time() - time_start)/60} minutes, {(time.time() - time_start)/60/60} hours, {(time.time() - time_start)/60/60/24} days\a')
