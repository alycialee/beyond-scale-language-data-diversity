# Checkpoints to eval -log P^Vocab(CorrectChoice)

```bash
# https://huggingface.co/datasets/UDACA/PileSubsets
# whole table: https://wandb.ai/brando/beyond-scale/table?nw=nwuserbrando

# Note: example ckpt path
# /lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_32m_24s/checkpoint-1551

# - ckpt 1
# https://wandb.ai/brando/beyond-scale/runs/ghjkc8tc/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t02h_02m_02s
# LLama2_Uspto_Ckpt_1
# uspto <-> 0.158

# - ckpt 2
# https://wandb.ai/brando/beyond-scale/runs/7jqujyv1/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_50m_22s
# LLama2_Pubmed_Ckpt_2
# pubmed <-> 0.168 

# - ckpt 3
# https://wandb.ai/brando/beyond-scale/runs/s9ou7l1n/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_47m_30s
# LLama2_Uspto_Pubmed_Ckpt_3
# uspto + pubmed <-> 0.195

# - ckpt 4
# https://wandb.ai/brando/beyond-scale/runs/3o05mvz6/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_45m_48s
# LLama2_Uspto_Pubmed_Ckpt_4
# uspto + pubmed <-> 0.195

# - ckpt 5
# https://wandb.ai/brando/beyond-scale/runs/ad2f1yew/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_34m_01s
# LLama2_Uspto_Pubmed_Ckpt_5
# uspto + pubmed <-> 0.195

# - ckpt 6
# https://wandb.ai/brando/beyond-scale/runs/2dy8rrcc/logs 
# /lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_01m_30s
# LLama2_Uspto_Pubmed_Ckpt_6
# uspto + pubmed <-> 0.195

# - ckpt 7
# https://wandb.ai/brando/beyond-scale/runs/fj5xd2kj?nw=nwuserbrando
# /lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_00m_55s
# LLama2_Uspto_Pubmed_Ckpt_7
# pubmed <-> 0.168
```

```bash
diversity_coefficients = {
    "LLama2_Uspto_Ckpt_1": 0.158,
    "LLama2_Pubmed_Ckpt_2": 0.168,
    "LLama2_Uspto_Pubmed_Ckpt_3": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_4": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_5": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_6": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_7": 0.168
}
```

Let's push the ckpts
```bash
ssh brando9@ampere9.stanford.edu 

conda activate beyond_scale_div_coeff

python ~/beyond-scale-language-data-diversity/src/push_hf_models_to_hf.py
```

Eval from RS:
```bash

conda create -n eleuther_lm_eval_harness_20240927 python=3.11

conda activate eleuther_lm_eval_harness_20240927

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e . && cd ..
```