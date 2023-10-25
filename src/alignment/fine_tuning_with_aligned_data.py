"""
Goal: making HF training script for model (e.g., llama v2) using raw text of informal and formal mathematics (unpaired data).

Inspiration:
- ref: SO accelerate + trainer: https://stackoverflow.com/questions/76675018/how-does-one-use-accelerate-with-the-hugging-face-hf-trainer
- ref: The unreasonable effectiveness of few-shot learning for machine translation https://arxiv.org/abs/2302.01398
- ref: colab: https://colab.research.google.com/drive/1io951Ex17-6OUaogCo7OiR-eXga_oUOH?usp=sharing
- ref: SO on collate: https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999

Looks very useful especially for peft:
- peft https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py

python trl/examples/scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2

- qlora https://github.com/artidoro/qlora/blob/main/scripts/finetune_llama2_guanaco_7b.sh, 
- https://github.com/artidoro/qlora/blob/main/qlora.py
"""
from pathlib import Path
import datasets
from datasets import load_dataset, interleave_datasets
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
import math


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
    buffer_size = 500_000
    probabilities = []
    data_mixture_name = None
    streaming = False
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

    # -- Setup wandb
    import wandb
    # - Dryrun
    # mode = 'dryrun'; seed = random.randint(0, 2**32 - 1)
    mode = 'dryrun'; seed = 0; report_to = 'none'

    # - Online (real experiment)
    # mode = 'online'; seed = random.randint(0, 2**32 - 1)
    mode = 'online'; seed = 0; report_to = 'wandb'

    # - c4 wt single
    path, name, data_files, split = ['csv'], [None], [os.path.expanduser('~/data/maf_data/maf_textbooks_csv_v1/train.csv')], ['train']
    # path, name, data_files, split = ['suolyer/pile_pile-cc'] + ['parquet'] * 4, [None] + ['hacker_news', 'nih_exporter', 'pubmed', 'uspto'], [None] + [urls_hacker_news, urls_nih_exporter, urls_pubmed, urls_uspto], ['validation'] + ['train'] * 4
    # pretrained_model_name_or_path = 'gpt2'
    pretrained_model_name_or_path = 'meta-llama/Llama-2-7b-hf'
    # pretrained_model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'
    # pretrained_model_name_or_path = 'meta-llama/Llama-2-13b-hf'
    # pretrained_model_name_or_path = 'meta-llama/Llama-2-70b-hf'
    pretrained_model_name_or_path = 'mistralai/Mistral-7B-v0.1'
    # - important training details or it wont run, mem issues maybe
    num_epochs = 1
    # num_epochs = 2
    # num_epochs = 4
    # single gpu
    # batch_size, gradient_accumulation_steps = 2, 1  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    batch_size, gradient_accumulation_steps = 2, 16  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    # batch_size, gradient_accumulation_steps = 2, 32  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    # -- multiple gpus 3 4096 context len
    # batch_size, gradient_accumulation_steps = 4, 8  # e.g., choosing large number mabe for stability of training? 4 (per_device_train_batch_size) * 8 (gradient_accumulation_steps), based on alpaca https://github.com/tatsu-lab/stanford_alpaca 
    # gradient_checkpointing = False
    gradient_checkpointing = True
    print(f'{batch_size=} {gradient_accumulation_steps=} {gradient_checkpointing=} {num_epochs=}')
    # -- wandb 
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    # run_name = f'{path} div_coeff_{num_batches=} ({today=} ({name=}) {data_mixture_name=} {probabilities=} {pretrained_model_name_or_path=})'
    run_name = f'training maths: {path} ({today=} ({name=}) {data_mixture_name=} {probabilities=} {pretrained_model_name_or_path=} {data_files=} {num_epochs=} {batch_size=} {gradient_accumulation_steps=})'
    print(f'\n---> {run_name=}\n')

    # - Init wandb
    debug: bool = mode == 'dryrun'  # BOOL, debug?
    run = wandb.init(mode=mode, project="maf", name=run_name, save_code=True)
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
    elif 'Llama-2' in pretrained_model_name_or_path or 'Mistral' in pretrained_model_name_or_path:
        # - llama2
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
        # bf16 or fp32
        bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        if bf16:
            torch_dtype = torch.bfloat16
        else: 
            torch_dtype = torch.float32
        # get model
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            # quantization_config=quantization_config,
            # device_map=device_map,  # device_map = None  https://github.com/huggingface/trl/blob/01c4a35928f41ba25b1d0032a085519b8065c843/examples/scripts/sft_trainer.py#L82
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_auth_token=True,
        )
        # HF trainer load to gpu on it's own: https://claude.ai/chat/43796e10-2139-4668-ac5c-aafeeeeeba2e
        # # -- Detect if running with accelerate https://claude.ai/chat/43796e10-2139-4668-ac5c-aafeeeeeba2e
        # from accelerate import Accelerator
        # accelerator = Accelerator()
        # # self.is_deepspeed_enabled = getattr(accelerator.state, "deepspeed_plugin", None) is not None
        # is_fsdp_enabled = getattr(accelerator.state, "fsdp_plugin", None) is not None
        # if not is_fsdp_enabled: # not sure if this is needed but its for sure safer
        #     # maybe figuring out how to run everything with accelerate would fix things...
        #     # ref: https://stackoverflow.com/questions/77204403/does-one-need-to-load-the-model-to-gpu-before-calling-train-when-using-accelerat
        #     device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        #     model = model.to(device)
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
        print(f'{max_length=}')
    # print(f'{device=}')
    print(f'{torch.cuda.device_count()=} (makes sure GPUs are visible and accesible to Pytorch.)')
    print(f'Model is currently on: {next(iter(model.parameters())).device=}')
    # name = "tiiuae/falcon-rw-1b",
    
    # -- Load datasets
    # - Get train data set
    # train_datasets = [load_dataset(path, name, streaming=True, split="train").with_format("torch") for path, name in zip(path, name)]
    train_datasets = [load_dataset(path, name, data_files=data_file, streaming=streaming, split=split).with_format("torch") for path, name, data_file, split in zip(path, name, data_files, split)]
    probabilities = [1.0/len(train_datasets) for _ in train_datasets]  # TODO: perhaps we should change weights to informal and formal have same weight? right now is just in terms of list of data sets perhaps having 2 interleaves one for formal one for informal then use another interleave and do 50/50?. 
    train_dataset = interleave_datasets(train_datasets, probabilities)
    # TODO: suffle data set False, True, note i've experienced that with shuffle_ds.take(512) is slow...
    shuffled_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else train_dataset
    batch = shuffled_dataset.take(batch_size) if streaming else shuffled_dataset.select(random.sample(list(range(len(shuffled_dataset))), batch_size))
    # print(f'{batch=}')
    # column_names = next(iter(batch)).keys()
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        # return tokenizer(examples["text"], padding="max_length", max_length=model.config.context_length, truncation=True, return_tensors="pt")
    # collate function does this already
    # remove_columns = column_names  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    # def map(batch):
    #     return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    # train_dataset = map(train_dataset)

    # - Get eval data set (AF for us), https://huggingface.co/datasets/brando/debug1_af
    per_device_eval_batch_size = 47  # TODO: change to something larger, right now due to size of my debug0
    # TODO: probably need to write a collate_fn for the eval so that the eval is done right?
    # TODO: we need ppl (and ideally token edit distance for eval, reason explained here: https://arxiv.org/abs/2304.15004)
    path, name = 'brando/debug1_af', None
    eval_dataset = load_dataset(path, name, streaming=False, split="test").with_format(type="torch") 
    ## eval_dataset = train_dataset  # TODO: fix obviously to something else using af
    raw_text_batch = eval_dataset.take(per_device_eval_batch_size) if streaming else eval_dataset.select(range(per_device_eval_batch_size))
    print(f'{raw_text_batch=}')
    print(f'{next(iter(raw_text_batch))=}')
    column_names = next(iter(raw_text_batch)).keys()
    def eval_preprocess(examples):
        return tokenizer(examples["formal statement"] + examples["generated informal statement"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    remove_columns = column_names  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    def map(batch):
        return batch.map(eval_preprocess, batched=True, remove_columns=remove_columns)
    eval_dataset = map(eval_dataset)
    train_dataset = train_dataset


    # -- Compute max steps
    per_device_train_batch_size = batch_size
    print(f'{per_device_train_batch_size=}')
    # dataset_size: int = int(1.5e12)  # TODO, doesn't seem easy to solve. Either count all the sequennces/rows or have the meta data have this. Or make this number huge. 
    dataset_size: int = train_dataset.num_rows
    # dataset_size: int = len(train_dataset)
    # TODO dataset.info['split']['train']['num_examples']
    # dataset_size = sum(len(dataset) for dataset in datasets)  # TODO: works on with streaming = False?
    # dataset_size = sum(dataset.cardinality() for dataset in datasets)
    print(f'{dataset_size=}')
    # # TODO: feel free to fix the issue if I'm not seeing all the data points...
    # num_epochs = 1
    max_steps = (dataset_size // per_device_train_batch_size) * num_epochs
    print(f'{num_epochs=} {max_steps=}')
    ## DOESNT WORK num_train_epochs = 3  # TODO: since I decided to do streaming = False and if we collect enough data it's unlikely we see it all hopefully (if we do 3 times seems good given that LLMs are trained to see the data only once this seems a sensible soln, + in the imagenet days things were trained to convergence with no overfitting ref: https://arxiv.org/abs/1801.00173)

    # -- Define custom collate function
    def custom_collate_fn(data: list[dict[str, str]], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
        """ trains on first occurence of eos
        
        ref: https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954/13?u=brando 
        ref: https://chat.openai.com/share/02d16770-a1f3-4bf4-8fc2-464286daa8a1
        ref: https://claude.ai/chat/80565d1f-ece3-4fad-87df-364ce57aec15 on when to call .clone()
        """
        # we are training full context length forllama so remove code bellow, if it triesto pad hopefully it throws an error
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
        # -- Set the mask value for the first eos_token in each sequence to 1
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

    # - Debug before training to see data
    sample_data = train_dataset.select(range(per_device_train_batch_size)) if not isinstance(train_dataset, datasets.iterable_dataset.IterableDataset) else train_dataset.take(per_device_train_batch_size)
    processed_data = custom_collate_fn(sample_data, tokenizer=tokenizer)
    print(f'{processed_data=}')

    # -- Training arguments and trainer instantiation ref: https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments
    output_dir = Path(f'~/data/maf_data/results_{today}/').expanduser() if not debug else Path(f'~/data/maf_data/results/').expanduser()
    print(f'{debug=} {output_dir=} \n {report_to=}')
    training_args = TrainingArguments(
        output_dir=output_dir,  #The output directory where the model predictions and checkpoints will be written.
        # num_train_epochs = num_train_epochs, 
        max_steps=max_steps,  # TODO: hard to fix, see above
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        gradient_checkpointing = gradient_checkpointing,  # TODO depending on hardware set to true?
        optim="paged_adamw_32bit",  # David hall says to keep 32bit opt https://arxiv.org/pdf/2112.11446.pdf TODO: if we are using brain float 16 bf16 should we be using 32 bit? are optimizers always fb32?  https://discuss.huggingface.co/t/is-there-a-paged-adamw-16bf-opim-option/51284
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=500,  # TODO: once real training starts we can select this number for llama v2, what does llama v2 do to make it stable while v1 didn't?
        warmup_ratio=0.03,  # copying alpaca for now, number of steps for a linear warmup, TODO once real training starts change? 
        # weight_decay=0.01,  # TODO once real training change?
        weight_decay=0.00,  # TODO once real training change?
        learning_rate = 1e-5,  # TODO once real training change? anything larger than -3 I've had terrible experiences with
        max_grad_norm=1.0, # TODO once real training change?
        lr_scheduler_type="cosine",  # TODO once real training change? using what I've seen most in vision 
        logging_dir=Path('~/data/maf/logs').expanduser(),
        save_steps=2000,  # alpaca does 2000, other defaults were 500
        # logging_steps=500,
        # logging_steps=50,
        logging_steps=1,
        remove_unused_columns=False,  # TODO don't get why https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999 , https://claude.ai/chat/475a4638-cee3-4ce0-af64-c8b8d1dc0d90
        report_to=report_to,  # change to wandb!
        fp16=False,  # never ever set to True
        bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        do_eval=True,
    )
    # print(f'{training_args=}')
    print(f'{pretrained_model_name_or_path=}')

    # TODO: might be nice to figure our how llamav2 counts the number of token's they've trained on
    trainer = Trainer(
        model=model,
        args=training_args,  
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda data: custom_collate_fn(data, tokenizer=tokenizer)
    )
    # - TODO bellow is for qlora from falcon, has same interface as Trainer later lets use: https://github.com/artidoro/qlora
    # from trl import SFTTrainer
    # peft_config = None
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=trainset,
    #     peft_config=peft_config,
    #     dataset_text_field="text",
    #     max_seq_length=max_seq_length,
    #     tokenizer=tokenizer,
    #     args=training_arguments,
    # )
    # TODO why this? https://discuss.huggingface.co/t/why-do-you-need-to-re-upcast-the-norm-layers-of-hf-falcon-to-fb32/46139
    # for name, module in trainer.model.named_modules():
    #     if "norm" in name:
    #         module = module.to(torch.float32)

    # - Train
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        print(f"CUDA_VISIBLE_DEVICES = {cuda_visible_devices}")
    trainer.train()
    trainer.save_model(output_dir=output_dir)  # TODO is this relaly needed? https://discuss.huggingface.co/t/do-we-need-to-explicity-save-the-model-if-the-save-steps-is-not-a-multiple-of-the-num-steps-with-hf/56745
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
