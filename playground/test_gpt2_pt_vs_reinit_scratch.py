#%%
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from pdb import set_trace as st

def reinit_gpt_neox_20B_inspired_use_case_llama2_mutates(model, 
                                                L: int,  # for beyond scale we filled the data to block size which is 4096 for max seq length llama2
                                                ):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # all linear layers including MLP and attention, let's try this first given it's smaller
            D = module.in_features  # I think this is right size it's xW []
            std = 3 / (L * (D)**0.5)
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:  # don't think biases matter cuz bias=False in all layers
                nn.init.constant_(module.bias, 0)
        elif str(module) == 'LlamaRMSNorm()':
            if hasattr(module, 'weight'):
                if module.weight is not None:  # todo: idk if needed for layer norm
                    nn.init.constant_(module.weight, 1.0)
            if hasattr(module, 'bias'):  # I don't think RMSNorm has bias, the whole point it doesn't think centering matters so bias is similar to centering
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        else:  
            if hasattr(module, 'weight'):
                if module.weight is not None: 
                    D = module.weight.shape[0]
                    std = (2 / (D + 4*D))**0.5  # e.g., small init attention layers
                    nn.init.normal_(module.weight, mean=0, std=std)
            if hasattr(module, 'bias'):
                if module.bias is not None:  # don't think biases matter cuz bias=False in all layers
                    nn.init.constant_(module.bias, 0)
    return

def reinit_gpt2_weights_mutates(
        model, 
        # weight_std: float = 0.00000002,  # 0.02 ref: Hailey S doesn't recommend this huge value! ref: https://x.com/haileysch__/status/1822758486632997102 I'm choosing a really small value due to my previous research with Tommy Poggio suggested to us that larger inits give worse generalization error
        weight_std: float = 2e-6,  # 0.02 ref: Hailey S doesn't recommend this huge value! ref: https://x.com/haileysch__/status/1822758486632997102 I'm choosing a really small value due to my previous research with Tommy Poggio suggested to us that larger inits give worse generalization error
        # weight_std: float = 0.0,
        bias_std: float = 0.0, 
        verbose: bool = False,
        ) -> None:
    """ 
    Why we chose < 0.02 for standard deviation: https://github.com/alycialee/beyond-scale-language-data-diversity/issues/18
    Reinit for gpt2 only test for xl: https://huggingface.co/openai-community/gpt2-xl
    """
    model_weight_norm = sum([torch.norm(param, p=2).item() for param in model.parameters()]) if verbose else None
    print(f'{model_weight_norm=}') if verbose else None
    for module_name, module in model.named_modules():
        print(f'{module_name=} {isinstance(module, nn.Linear)=} {type(module)=}') if verbose else None
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, mean=0, std=0.02) # original, evil!
            print(f'{module.weight.norm(2)=}') if verbose else None
            nn.init.normal_(module.weight, mean=0, std=weight_std)
            print(f'{module.weight.norm(2)=}') if verbose else None
            if module.bias is not None:
                # gpt suggestion: https://chatgpt.com/c/b9d34414-a123-48d6-bbae-334dedb580f3
                nn.init.constant_(module.bias, bias_std)
        elif isinstance(module, nn.Embedding):
            print(f'{module.weight.norm(2)=}') if verbose else None
            nn.init.normal_(module.weight, mean=0, std=weight_std)
            print(f'{module.weight.norm(2)=}') if verbose else None
        elif isinstance(module, nn.Dropout):
            pass # has no params
        elif isinstance(module, nn.LayerNorm):
            # gpt suggestion: https://chatgpt.com/c/b9d34414-a123-48d6-bbae-334dedb580f3
            print(f'{module.weight.norm(2)=}') if verbose else None
            nn.init.constant_(module.weight, 0.0)
            print(f'{module.weight.norm(2)=}') if verbose else None
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d):
            print(f'{module.weight.norm(2)=}') if verbose else None
            nn.init.normal_(module.weight, mean=0, std=weight_std)
            print(f'{module.weight.norm(2)=}') if verbose else None
            if module.bias is not None:
                nn.init.constant_(module.bias, bias_std)
        # elif isinstance(module, nn.NewGELUActivation):
        #     pass
        else:  
            if hasattr(module, 'weight'):
                if module.weight is not None: 
                    D = module.weight.shape[0]
                    # std = (2 / (D + 4*D))**0.5  # e.g., small init attention layers
                    std = weight_std
                    nn.init.normal_(module.weight, mean=0, std=std)
            if hasattr(module, 'bias'):
                if module.bias is not None:  # don't think biases matter cuz bias=False in all layers
                    nn.init.constant_(module.bias, bias_std)
    model_weight_norm = sum([torch.norm(param, p=2).item() for param in model.parameters()]) if verbose else None
    print(f'{model_weight_norm=}') if verbose else None
    return

# Step 1: Load the pre-trained GPT-2 XL model
torch.cuda.empty_cache() # Clear CUDA cache to free up memory
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch_dtype, trust_remote_code=True)
print(f'{model=}')
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
pretrained_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", padding_side="right", trust_remote_code=True)
print(f'log(Vocab Length): {torch.log(torch.tensor(len(pretrained_tokenizer)))=}')
pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token if pretrained_tokenizer.pad_token_id is None else pretrained_tokenizer.pad_token
print(f'{pretrained_tokenizer=}\n{pretrained_tokenizer.bos_token_id=} {pretrained_tokenizer.eos_token_id=} {pretrained_tokenizer.pad_token_id=} {pretrained_tokenizer.vocab_size=}')
# Step 2: Calculate the L2 norm of the weights for the pre-trained model
pretrained_weight_norm = sum([torch.norm(param, p=2).item() for param in model.parameters()])
print(f"Total L2 norm of pre-trained model weights: {pretrained_weight_norm:.2f}")

# Step 1: Initialize a new GPT-2 model from scratch with custom configuration
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch_dtype, trust_remote_code=True)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
config = GPT2Config(
    vocab_size=pretrained_tokenizer.vocab_size,  # Ensure this matches the tokenizer's vocabulary size
    n_ctx=1024,  # Context window size (number of tokens the model can see at once)
    bos_token_id=pretrained_tokenizer.bos_token_id,  # Begin-of-sequence token
    eos_token_id=pretrained_tokenizer.eos_token_id,  # End-of-sequence token
    pad_token_id=pretrained_tokenizer.eos_token_id,  # pad-sequence token
)
model = AutoModelForCausalLM.from_config(config)
# Step 2: Calculate the L2 norm of the weights for the freshly initialized model
scratch_weight_norm = sum([torch.norm(param, p=2).item() for param in model.parameters()])
print(f"Total L2 norm of model initialized from scratch: {scratch_weight_norm:.2f}")

# Step 1: Reinit GPT2 with really small init
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch_dtype, trust_remote_code=True)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
reinit_gpt2_weights_mutates(model)
scratch_weight_norm_small_reinit = sum([torch.norm(param, p=2).item() for param in model.parameters()])
print(f"Total L2 norm of model initialized from scratch with small reinit (not default HF config): {scratch_weight_norm_small_reinit:.2f}")

# Step 1: Reinit GPT2 with really small init
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch_dtype, trust_remote_code=True)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
reinit_gpt_neox_20B_inspired_use_case_llama2_mutates(model, 1024)
scratch_weight_norm_small_reinit = sum([torch.norm(param, p=2).item() for param in model.parameters()])
print(f"Total L2 norm of model initialized from scratch with gpt_neox_20B reinit (not default HF config): {scratch_weight_norm_small_reinit:.2f}")

# Justification:
# If the model is truly being initialized from scratch, the weight norm should be much smaller compared to the pre-trained model. 
# This confirms that the training process is starting from a random initialization and not from any pre-existing pre-trained weights.