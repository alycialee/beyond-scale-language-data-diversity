# Run loss(correct)=-logP^V(CorrectChoice)

```bash
# https://chatgpt.com/c/66f88357-c028-8001-bc3b-b32c86c8d2a3
conda activate beyond_scale_div_coeff

# Add the Submodule
git submodule add https://github.com/brando90/Brando-LLM-Eval-Demo.git Brando-LLM-Eval-Demo

# Initialize and Update the Submodule
git submodule update --init --recursive

# Commit the Submodule
git add .gitmodules Brando-LLM-Eval-Demo
git commit -m "Added Brando-LLM-Eval-Demo as a submodule"

```

```bash
krbtmux
reauth

conda activate eleuther_lm_eval_harness_20240927
cd ~/beyond-scale-language-data-diversity/Brando-LLM-Eval-Demo
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6
python -u queue_evals.py ${CUDA_VISIBLE_DEVICES} 
```