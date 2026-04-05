import copy
import math
import os
from pathlib import Path
from tqdm import tqdm

import torch

import json
from typing import Optional
import sys
from pprint import pprint
from vllm import LLM, SamplingParams, CompletionOutput, RequestOutput

import tenacity
import fire


# -- Mathematica gen prompt

# HELM_MATH_PROMPT_8SHOT_COT2_GEN_SYN_METHEMATICA_TEMPLATE: str = (
# """Given a mathematics problem, its answer, math category, and problem type, generate a solution to the problem, with step by step reasoning that justifies the answer. 
# Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}.### 
# This is the format:
# Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
# Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
# Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction.
# Solution: Let's think step by step. Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}. The final answer is: \\boxed{\\frac{1}{2}}.###
# Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$?
# Solution: Let's think step by step.Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$. The final answer is: \\boxed{0}.###
# Problem: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?
# Solution: Let's think step by step. Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\\cdot800=19\\cdot200=\\boxed{3800}$ feet. The final answer is: \\boxed{3800}###
# Problem: If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$?
# Solution: Let's think step by step. We know that $(x + y)^2 = (x^2 + y^2) + 2xy = 25$. We are given that $xy = 6$. So, by substitution, $x^2 + y^2 + 2xy = x^2 + y^2 + 2(6) = 25$. It follows that $x^2 + y^2 = 25 - 12 = \\boxed{13}$. The final answer is: \\boxed{13}###
# Problem: On a hot day, Megan likes to eat a Popsicle every 15 minutes. Assuming she keeps up that rate of consumption, how many Popsicles can Megan finish in 4 hours and 30 minutes?
# Solution: Let's think step by step. Let $p$ be the number of Popsicles Megan can finish in 4 hours and 30 minutes. If we convert that period of time into minutes, we find that 4 hours and 30 minutes is equal to $(4)(60)+30=270$ minutes. From here, we can set up the proportion \\begin{align*} \\frac{x}{270}& =\\frac{1}{15}\\\\\\Rightarrow \\qquad x& =\\left(\\frac{1}{15}\\right)(270)\\\\\\Rightarrow \\qquad x& =\\boxed{18}\\end{align*}. The final answer is: \\boxed{18}###
# Problem: Compute $95^2$ in your head.
# Solution: Let's think step by step. We have $(90 + 5)^2 = 90^2 + 2(90)(5) + 5^2 = 8100 + 900 + 25 = \\boxed{9025}$. The final answer is: \\boxed{9025}.###
# Problem: If $2^8=16^x$, find $x$.
# Solution: Let's think step by step. We can write $16$ as $2^4$. Therefore, we can write our equation as $2^8 = 2^{4 \\cdot x}$. Solving, we get that $x = \\boxed{2}$. The final answer is: \\boxed{2}.###
# This is the new problem to generate a solution string from the answer, problem, category, and problem type:
# Problem: {problem}
# The answer is {answer}, the math category is {category}, the problem type is {prob_type}.
# Solution: Let's think step by step.""")

# HELM_MATH_PROMPT_8SHOT_COT2_USE_SOLN_CODE_GEN_SYN_METHEMATICA_TEMPLATE: str = (
# """Given a mathematics problem, its answer, math category, problem type, and mathematica code that describes the solution to the answer, generate a solution to the problem, with step by step reasoning followed by the answer. 
# Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}.### 
# This is the format:
# Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
# Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
# Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction.
# Solution: Let's think step by step. Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}. The final answer is: \\boxed{\\frac{1}{2}}.###
# Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$?
# Solution: Let's think step by step.Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$. The final answer is: \\boxed{0}.###
# This is the new problem to generate a solution string from the answer, problem, category, and problem type:
# Problem: {problem}
# The answer is {answer}, the math category is {category}, the problem type is {prob_type}, and the mathematica code is {mathematica_code}. 
# Generate a solution using that information:
# Solution: Let's think step by step.""")

# # Goal of prompt is to generate (missing) solution strings from a the mathematica code, given 1 few shot example. ref: https://chatgpt.com/c/fb17c070-bd1e-4aad-9078-88c44abc4ae9
# mathematica_code1: str = open(os.path.expanduser('~/gold-ai-olympiad/py_src/training/algebraic_manipulation.nb')).read()
# MATHEMATICA_SYNTH_GEN_PROMPT_TEMPLATE: str = (
# f"""
# Given a mathematics problem, its answer, math category, problem type, and mathematica code that describes the solution to the answer, generate a solution to the problem, with step by step reasoning followed by the answer. 
# Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}.###
# Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
# Answer: 238
# Math Category: Algebra
# Problem Type: Algebraic Manipulation
# Mathematica Code: {mathematica_code1}
# Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
# Problem: {{problem}}
# Answer: {{answer}}
# Math Category: {{category}}
# Problem Type: {{prob_type}}
# Mathematica Code: {{mathematica_code}}
# Solution: Let's think step by step."""
# )
# Goal of prompt is to generate (missing) solution strings from a the mathematica code, given 1 few shot example. ref: https://chatgpt.com/c/fb17c070-bd1e-4aad-9078-88c44abc4ae9
MATHEMATICA_SYNTH_GEN_PROMPT_TEMPLATE: str = (
f"""
Given a mathematics problem, its answer, math category, and problem type that describes the solution to the answer, generate a solution to the problem, with step by step reasoning followed by the answer. 
Simplify your answer as much as possible. Always give the final answer inside a \\boxed{{answer}}.###
Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
Answer: 238
Math Category: Algebra
Problem Type: Algebraic Manipulation
Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
Problem: {{problem}}
Answer: {{answer}}
Math Category: {{category}}
Problem Type: {{prob_type}}
Solution: Let's think step by step."""
)

# --

@tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential(multiplier=2, max=64))
def gen_synthetic_solution(gen, prompt_template: str, problem, answer, category, prob_type = None) -> str:
    """
    Generates a synthetic solution to a math problem using a language model.

    Args:
        gen (GPT): The language model used to generate the solution.
        problem (str): The math problem.
        answer (str): The correct answer to the problem.
        category (str): The category of the problem.
        prob_type (str): The type of problem.

    Returns:
        str: The synthetic solution to the problem.
    """
    prompt: str = prompt_template.replace('{problem}', problem)
    # solution = generate_completion(problem, answer, category, prob_type)
    response: dict = gen.llm.chat.completions.create(
        model=gen.model,
        messages=[
            {"role": "system", "content": gen.system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=gen.sampling_params.temperature,
        top_p=gen.sampling_params.top_p,
        n=gen.sampling_params.n,
        stop=gen.sampling_params.stop[:3],
        )
    solution: str = response.choices[0].message.content
    return solution

def extract_problem_data(file_path: str) -> dict[str, str]:
    """
    Extracts the problem and answer from a .txt file.

    Args:
        file_path (str): The path to the .txt file containing the problem.

    Returns:
        Dict[str, str]: A dictionary with keys 'problem' and 'answer'.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().split("\nAnswer:\n")
        problem = content[0].strip()
        answer = content[1].strip()
    return {'problem': problem, 'solution': '', 'answer': answer}

def gen_data_set_from_ans_str_2_jsonl(
                                        gen, 
                                        source_path: str, 
                                        output_path: str,
                                        prompt_template: str, 
                                        num_data_gens_per_txt_files: int,
                                        batch_size: Optional[int] = sys.maxsize,
                                        batched: Optional[bool] =True,
                                        use_mathematica_code: bool = True,
                                ) -> list[dict]:
    """
    Creates a JSON Lines file from .txt files containing math problems across directories.

    Args:
        source_path (str): Path to the source directory containing folders of .txt files.
        output_path (str): Path to save the .jsonl output file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the json lines.
    """
    print(f'{batched=}')
    source_path: Path = Path(source_path).expanduser()
    output_path: Path = Path(output_path).expanduser()
    print(f'{source_path=} {output_path=}')

    # -- Collect all the raw data
    print()
    all_data: list[dict] = []
    for root, dirs, files in os.walk(source_path):
        # print(f'\n{root=} \n{dirs[:3]=} \n{files[:3]=}')
        for f in files[:num_data_gens_per_txt_files]:  # Select up to 50 .txt files per directory
            if f.endswith('.txt'):
                # print(f'\n{root=} \n{dirs[:3]=} \n{files[:3]=}')
                # selected_files = [os.path.join(root, f) for f in files if f.endswith('.txt')][:num_data_gens_per_txt_files]  
                file_path = os.path.join(root, f)
                category: str = root.split('/')[-2]
                problem_type: str = root.split('/')[-1]
                mathematica_script_name: str = f'{problem_type}.nb'
                # extract problem, answer, category, prob_type, (empty) solution
                problem_data: dict = extract_problem_data(file_path)
                all_data.append(problem_data)
                problem_data['problem_type'] = ' '.join(problem_type.split('_'))  # "variance_and_std" -> "variance and std"
                problem_data['category'] = ' '.join(category.split('_'))  # "counting_and_statics" -> "counting and statics"
                problem_data['mathematica_script_name'] = mathematica_script_name
                # get everything up to the last but last is not included
                root_path_2_mathematica_scripts: str = '/'.join(root.split('/')[:-1])  # <path>/<cat> since cat has the nb files
                path_2_mathematica_script: str = f'{root_path_2_mathematica_scripts}/{mathematica_script_name}'
                # mathematica_script: str = open(path_2_mathematica_script, 'r').read()
                # print(f'{problem_data=}')
                problem_data['mathematica_script'] = ''
                problem_data['mathematica_script_path'] = path_2_mathematica_script
                # print(f'{problem_data=}')
                # print()
    print(f'{len(all_data)=}')
    
    # TODO -- Batch the data
    if batched:
        from evals.utils import batch_data
        assert batch_size > 0, f'batch_size should be greater than 0 but got: {batch_size=}'
        data: list[dict] = batch_data(all_data, batch_size=batch_size)
        num_batches: int = len(data)
        print(f'len(all_data)/batch_size = {len(data)}/{batch_size} = {math.ceil(len(data)/batch_size)}')
        print(f'{num_batches=}')

    # -- Gen synthetic data
    from evals.inference_eval import VllmGenerator, OpenAIGenerator
    print(f'{type(data)=} {type(data[0])=}')  # to see which type of prompts we will make
    if isinstance(gen, VllmGenerator) and isinstance(data[0], dict):  # list of single string prompt 
        print(f'Number of data points to be generated: {len(all_data) * gen.sampling_params.n=}')
        data: list[dict]
        # - Generate synthetic solution string for each problem and answer (using meta-data too)
        all_data_with_gen_synth_soln: list[dict] = []
        for raw_data in tqdm(all_data, total=len(all_data)):
            # Make prompt to generate synthetic solution
            problem: str = raw_data['problem']
            answer: str = raw_data['answer']
            category: str = raw_data['category']
            prob_type: str = raw_data['problem_type']
            mathematica_code: str = raw_data['mathematica_script']
            # prompt: str = prompt_template.replace('{problem}', problem).replace('{answer}', answer).replace('{category}', category).replace('{prob_type}', prob_type).replace('{mathematica_code}', mathematica_code)
            prompt: str = prompt_template.replace('{problem}', problem).replace('{answer}', answer).replace('{category}', category).replace('{prob_type}', prob_type)
            # Generate synthetic solutions (completions), n>1 means same reasoning per solution but phrased differently
            request_outputs_per_batch_prompts: list[RequestOutput] = gen.llm.generate(prompt, gen.sampling_params) 
            completions_per_prompt: list[CompletionOutput] = request_outputs_per_batch_prompts[0].outputs
            synthetic_solutions: list[str] = [completion.text for completion in completions_per_prompt]
            # Create dict data point per problem answer pair with its synthetic solutions
            print(f'First solution: {synthetic_solutions[0]=}')
            print(f'Real answer: {answer=}')
            if len(synthetic_solutions) == 1:
                raw_data['solution'] = synthetic_solutions[0]
                all_data_with_gen_synth_soln.append(raw_data)
            else:
                data_points_per_ans: list[dict] = [{**raw_data, 'solution': solution} for solution in synthetic_solutions]
                all_data_with_gen_synth_soln.extend(data_points_per_ans)
                assert len(synthetic_solutions) == gen.sampling_params.n, f'{len(synthetic_solutions)=} != {gen.sampling_params.n=}'
        assert len(all_data_with_gen_synth_soln) == len(all_data) * len(synthetic_solutions), f'{len(all_data_with_gen_synth_soln)=} != {len(all_data) * len(synthetic_solutions)=}'
    elif isinstance(gen, VllmGenerator) and isinstance(data[0], list):  # list of batch of prompts
        print(f'Number of data points to be generated: {len(all_data) * gen.sampling_params.n=}')
        data: list[list[dict]]
        # - Generate synthetic solution string for each problem and answer (using meta-data too)
        all_data_with_gen_synth_soln: list[dict] = []
        for batch_probs_ans in data:
            batch_prompts: list[str] = [prompt_template.replace('{problem}', prob_ans['problem']).replace('{answer}', prob_ans['answer']).replace('{category}', prob_ans['category']).replace('{prob_type}', prob_ans['problem_type']) for prob_ans in batch_probs_ans]
            request_outputs: list[RequestOutput] = gen.llm.generate(batch_prompts, gen.sampling_params)
            for prob_ans_dict, request_output_per_prompt in zip(batch_probs_ans, request_outputs):
                completions_per_prompt: list[CompletionOutput] = request_output_per_prompt.outputs
                synthetic_solutions: list[str] = [completion.text for completion in completions_per_prompt]
                assert len(synthetic_solutions) == gen.sampling_params.n, f'{len(synthetic_solutions)=} != {gen.sampling_params.n=}'
                # Create dict data point per problem answer pair with its synthetic solutions
                for synthetic_solution in synthetic_solutions:
                    prob_ans_dict = copy.deepcopy(prob_ans_dict)
                    prob_ans_dict['solution'] = synthetic_solution
                    all_data_with_gen_synth_soln.append(prob_ans_dict)
        assert len(all_data_with_gen_synth_soln) == len(all_data) * len(synthetic_solutions), f'{len(all_data_with_gen_synth_soln)=} != {len(all_data) * len(synthetic_solutions)=}'
        raise NotImplemented
    elif isinstance(gen, OpenAIGenerator):
        raise ValueError(f'Invalid value for {gen=}.')
    else:
        raise ValueError(f'Invalid value for {gen=}.')
    print(f'{len(all_data_with_gen_synth_soln)=} {len(all_data)=}')
    assert isinstance(all_data_with_gen_synth_soln, list) and isinstance(all_data_with_gen_synth_soln[0], dict), f'{type(all_data_with_gen_synth_soln)=} {type(all_data_with_gen_synth_soln[0])=}'
    
    # -- Save all data to a jsonlines file
    output_path: Path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.is_dir() else output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'{output_path=}')
    with open(output_path, 'w', encoding='utf-8') as file:
        for synth_data in all_data_with_gen_synth_soln:
            json.dump(synth_data, file)
            file.write('\n')
    return all_data_with_gen_synth_soln

# -- Main

def main_gen_synth_data(
        path_2_src_dataset: str = '~/data/amps/mathematica/',
        output_path: str = '~/gold-ai-olympiad/data/amps/mathematica/train.jsonl',
        # model: str = 'DEFAULT',  # e.g., Mistral-7B-Instrcut-v0.2 on http://120.77.8.29:12345 
        # model: str = 'deepseek-ai/deepseek-math-7b-instruct', 
        # model: str = 'mistralai/Mistral-7B-Instruct-v0.1', 
        model: str = 'deepseek-ai/DeepSeek-Prover-V1.5-RL', 
        # model: str = 'gpt2', 
        # batch_size: int = 10,  # putnam has 348 
        n: int = 4, # num seqs to return for given prompt
        # max_tokens: int = 2048,
        max_tokens: int = 4096,
        top_p: float = 0.95, 
        temperature: float = 0.8,
        num_beams: int = None,
        best_of: int = None,
        use_beam_search: bool = False,
        num_data_gens_per_txt_files: int = 100,
        mathematica: bool = False,
    ):
    """ Gen synthetic data from math problems. """
    # from evals.prompts_evals import STOP_TOKENS
    from evals.inference_eval import VllmGenerator, OpenAIGenerator
    # -- Print
    print(f'{model=} {num_data_gens_per_txt_files=}')
    print()

    # -- Get vllm generator
    prompt_template: str = MATHEMATICA_SYNTH_GEN_PROMPT_TEMPLATE
    stop: list[str] = STOP_TOKENS
    sampling_params: SamplingParams = SamplingParams(n=n, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stop=stop, use_beam_search=use_beam_search, best_of=best_of)
    print(f'{model=}, \n{sampling_params=}')
    # gen: OpenAIGenerator = OpenAIGenerator(model=None, sampling_params=sampling_params)
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    llm: LLM = LLM(model=model, dtype=dtype)
    gen: VllmGenerator = VllmGenerator(llm, sampling_params)
    print(f'{sampling_params} \n {sampling_params=}')

    # -- Generate synthetic data & save it as a jsonlines file
    lst: list[dict] = gen_data_set_from_ans_str_2_jsonl(gen, path_2_src_dataset, output_path, prompt_template, num_data_gens_per_txt_files=num_data_gens_per_txt_files)
    print(f'Number of jsonlines: {len(lst)=}, written to {output_path=}, from {path_2_src_dataset=}')

if __name__ == '__main__':
    import time
    start = time.time()
    # main_gen_synth_data()
    fire.Fire(main_gen_synth_data)
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")
    