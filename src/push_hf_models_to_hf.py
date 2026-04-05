# ref: https://chatgpt.com/c/66f6d1bc-31c0-8001-937e-d3035580c660
import time
from transformers import AutoModel, AutoTokenizer
import os
import torch

def get_first_layer_l1_norm(model):
    """
    Calculate the L1 norm of the weights in the first layer of the model.
    """
    first_layer_weights = next(model.parameters())
    l1_norm = torch.norm(first_layer_weights, p=1).item()
    return l1_norm

def create_model_card(model_path, dataset_name):
    """
    Creates the content for the model card.
    """
    model_card_content = f"""
# LLaMA 2 Model - {dataset_name}

This is a LLaMA 2 model fine-tuned on the `{dataset_name}` dataset.

- **Model Checkpoint Path:** `{model_path}`
- **Dataset:** `{dataset_name}`
- **Model Type:** LLaMA 2

## Citation
If you use this model, please cite:
[https://arxiv.org/abs/2306.13840](https://arxiv.org/abs/2306.13840)
    """
    return model_card_content

def upload_model_and_tokenizer(model_path, repo_name, dataset_name, hf_token):
    """
    Uploads a model and tokenizer to the Hugging Face Hub under the specified organization.
    """
    print(f'-> Uploading model at path {model_path} to {repo_name=} with your hf token.')
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')  # Using the standard tokenizer
    
    # Compute and print the L1 norm of the first layer
    l1_norm = get_first_layer_l1_norm(model)
    print(f"L1 norm of the first layer before push: {l1_norm}")
    
    # Create a model card
    model_card_content = create_model_card(model_path, dataset_name)
    model_card_path = os.path.join(model_path, "README.md")
    with open(model_card_path, "w") as f:
        f.write(model_card_content)
    
    # Upload model, tokenizer, and model card to the Hugging Face Hub as public
    model.push_to_hub(repo_name, use_auth_token=hf_token, private=False)
    tokenizer.push_to_hub(repo_name, use_auth_token=hf_token, private=False)
    print(f"Model and tokenizer successfully uploaded to: https://huggingface.co/{repo_name}")

def test_download_model(repo_name):
    """
    Downloads the model and tokenizer from the Hugging Face Hub to verify successful upload.
    """
    print("Testing download of the model...")
    model = AutoModel.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    
    # Compute and print the L1 norm of the first layer
    l1_norm = get_first_layer_l1_norm(model)
    print(f"L1 norm of the first layer after download: {l1_norm}")
    
    print(f"Model and tokenizer downloaded successfully: {repo_name}")

def main():
    # List of checkpoint paths to upload and their dataset names
    checkpoint_info = [
        ("/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t02h_02m_02s", "uspto"),               # Checkpoint 1
        ("/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_50m_22s", "pubmed"),              # Checkpoint 2
        ("/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_47m_30s", "uspto-pubmed"),        # Checkpoint 3
        ("/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_45m_48s", "uspto-pubmed"),        # Checkpoint 4
        ("/lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_34m_01s", "uspto-pubmed"),        # Checkpoint 5
        ("/lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_01m_30s", "uspto-pubmed"),        # Checkpoint 6
        ("/lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_00m_55s", "pubmed")               # Checkpoint 7
    ]

    # Organization and Hugging Face access token
    organization = "UDACA"
    hf_token = open(os.path.expanduser('~/keys/token_made_for_div_coeff_llama2_pushes.txt')).read().strip()
    print(f'{hf_token=}')

    # Upload each checkpoint
    for i, (ckpt_path, dataset_name) in enumerate(checkpoint_info, start=1):
        repo_name = f"{organization}/llama2-{dataset_name}-ckpt-{i}"
        print(f"Uploading checkpoint {i} from {ckpt_path} to {repo_name}...")
        upload_model_and_tokenizer(ckpt_path, repo_name, dataset_name, hf_token)
        
        # Optionally, test downloading the model and tokenizer after upload
        # test_download_model(repo_name)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Done! Total time taken: {end_time - start_time:.2f} seconds")
