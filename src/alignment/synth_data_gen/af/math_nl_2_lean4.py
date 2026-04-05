import sys
from typing import Union, Callable, Iterator, Optional
import wandb
from alignment.synth_data_gen.af.af_prompts.lean4ai_af_prompt2 import STOP_TOKENS
from alignment.synth_data_gen.af.af_prompts.lean4ai_af_prompt2 import prompt_af_for_align_zero_shot, prompt_temp_2_prompt_af_for_align 

import torch

def main(
        # path_2_eval_dataset: str = '~/snap-cluster-setup/data/MATH/test',
        # path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_original_static_final_21_08_2024/Putnam_MATH_boxed_problems_full.json',
        # path_2_eval_dataset: str = '~/putnam-math/data/OlympiadBench_Dataset/data_math_boxed_21_08_2024_v2',
        # path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_variations_static_constant/test.json',
        # path_2_eval_dataset: str = '~/putnam-math/data/Putnam_MATH_variations_static_constant/original.json',
        model: str = 'deepseek-ai/deepseek-math-7b-instruct',
        output_dir: Optional[str] = '~/data/results_{today}/',  # e.g., where to save completions
        start: int = 0, 
        end: int = sys.maxsize, # Usually used to know what fraction of benchmark to evaluate on
        batch_size: int = sys.maxsize, # the size of batch size from eval set to evaluate per eval step, note: eventually evals on everything
        # batch_size: int = 5_000,  # MATH test has 5_000 
        n: int = 1, # num seqs to return for given prompt
        max_tokens: int = 4096,
        top_p: float = 0.95, 
        temperature: float = 0.8,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None, # max input for HF/vllm models
        gen_type: Optional[str] = 'vllm',
        use_beam_search: bool = False,
        best_of: Optional[int] = None,
        mode: str = 'dryrun',  # 'dryrun' or 'online'
        # mode: str = 'online',  # 'dryrun' or 'online'
        shuffle: bool = False, 
        seed: int = 42, 
):
    problems = [
        "If $A=2+i$, $O=-4$, $P=-i$, and $S=2+4i$, find $A-O+P+S$.", # /lfs/skampere1/0/brando9/snap-cluster-setup/data/MATH/train/algebra/3.json
        "The perimeter of a rectangle is 24 inches. What is the number of square inches in the maximum possible area for this rectangle?", # /lfs/skampere1/0/brando9/snap-cluster-setup/data/MATH/train/algebra/4.json
        "Find the remainder when the sum \\[75+76+77+78+79+80+81+82\\]is divided by 16.", # /lfs/skampere1/0/brando9/snap-cluster-setup/data/MATH/train/number_theory/6.json
        # /lfs/skampere1/0/brando9/snap-cluster-setup/data/OlympiadBench_Dataset/data_math/TP_TO_maths_en_COMP.json, line 6 in file q id 1602
        "Let $P$ and $P^{\\prime}$ be two convex quadrilateral regions in the plane (regions contain their boundary). Let them intersect, with $O$ a point in the intersection. Suppose that for every line $\\ell$ through $O$ the segment $\\ell \\cap P$ is strictly longer than the segment $\\ell \\cap P^{\\prime}$. Is it possible that the ratio of the area of $P^{\\prime}$ to the area of $P$ is greater than $1.9 ?$",
        # /lfs/skampere1/0/brando9/putnam-math/putnam-math/data/Putnam_MATH_original_static_final_21_08_2024/Putnam_MATH_boxed_problems.json , problem id id 2023_A1 
        "For a positive integer $n$, let $f_n(x) = \\cos(x) \\cos(2x) \\cos(3x) \\cdots \\cos(nx)$. Find the smallest $n$ such that $|f_n''(0)| > 2023$."
        ]
    
    # - Get vllm generator
    prompt_template: str = prompt_af_for_align_zero_shot
    print(f'--> {prompt_template=}')
    prompt_gen_func: Callable = prompt_temp_2_prompt_af_for_align
    print(f'{prompt_gen_func=}')
    # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop: list[str] = STOP_TOKENS
    # push to config before loading model to avoid any common llm issues
    wandb.config.update(dict(prompt_template=prompt_template, prompt_gen_func=str(prompt_gen_func), model=model, path_2_eval_dataset=path_2_eval_dataset, output_dir=output_dir, stop_tokens=stop, extract_answer_func=extract_answer_func))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'{dtype=}')

    # sampling_params: SamplingParams = SamplingParams(n=n, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stop=stop, use_beam_search=use_beam_search, best_of=best_of)
    # Note: some sampling params are not present in all inference frameworks so they need to be removed later
    from collections import namedtuple
    SamplingParams = namedtuple('SamplingParams', ['n', 'max_tokens', 'top_p', 'temperature', 'stop', 'use_beam_search', 'best_of', 'max_length', 'num_beams'])
    sampling_params = SamplingParams(n=n, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stop=stop, use_beam_search=use_beam_search, best_of=best_of, max_length=max_length, num_beams=num_beams)
    print(f'{sampling_params=}')
    print(f'--> {model=} {gen_type=}')
    if 'vllm' in str(gen_type).lower():
        from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput # here otherwise warning when doing api calls in cpu laptop, vllm only works for linux 100% ref: https://github.com/vllm-project/vllm/issues/2747
        llm: LLM = LLM(model=model, dtype=dtype, trust_remote_code=True)
        # remove any field not in vllm's SamplingParams code e.g., max_length is mostly a HF model concept
        default_vllm_sp_keys = vars(SamplingParams()).keys()
        _sampling_params = {key: field for key, field in sampling_params._asdict().items() if key in default_vllm_sp_keys}
        sampling_params = SamplingParams(**(_sampling_params))
        # from /lfs/skampere1/0/brando9/snap-cluster-setup/py_src/evals/inference_eval.py
        from evals.inference_eval import VllmGenerator
        gen: VllmGenerator = VllmGenerator(llm, sampling_params)
    # TODO figure out how to install snap cluster tutorial in beyond scale pip
    # TODO then run the inference call and gen the data
    # TODO put all the hf loaders from my train load statasets.py
    # TODO then run the zero shot

    


if __name__ == '__main__':
    import fire
    import time
    start = time.time()
    fire.Fire(main)
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")