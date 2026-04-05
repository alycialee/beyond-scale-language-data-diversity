# HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE_MISTRAL7B_INS_V1: str = (
# """<s>[INST] Given a mathematics problem, determine the answer. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}###
# Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$? [/INST]</s>
# Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
# [INST] Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction. [/INST]
# Solution: Let's think step by step. Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}. The final answer is: \\boxed{\\frac{1}{2}}.###
# [INST] Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$? [/INST]
# Solution: Let's think step by step.Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$. The final answer is: \\boxed{0}.
# [INST] Problem: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet? [/INST]
# Solution: Let's think step by step. Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\\cdot800=19\\cdot200=\\boxed{3800}$ feet. The final answer is: \\boxed{3800}###
# [INST] Problem: If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$? [/INST]
# Solution: Let's think step by step. We know that $(x + y)^2 = (x^2 + y^2) + 2xy = 25$. We are given that $xy = 6$. So, by substitution, $x^2 + y^2 + 2xy = x^2 + y^2 + 2(6) = 25$. It follows that $x^2 + y^2 = 25 - 12 = \\boxed{13}$. The final answer is: \\boxed{13}###
# [INST] Problem: On a hot day, Megan likes to eat a Popsicle every 15 minutes. Assuming she keeps up that rate of consumption, how many Popsicles can Megan finish in 4 hours and 30 minutes? [/INST]
# Solution: Let's think step by step. Let $p$ be the number of Popsicles Megan can finish in 4 hours and 30 minutes. If we convert that period of time into minutes, we find that 4 hours and 30 minutes is equal to $(4)(60)+30=270$ minutes. From here, we can set up the proportion \\begin{align*} \\frac{x}{270}& =\\frac{1}{15}\\\\\\Rightarrow \\qquad x& =\\left(\\frac{1}{15}\\right)(270)\\\\\\Rightarrow \\qquad x& =\\boxed{18}\\end{align*}. The final answer is: \\boxed{18}###
# [INST] Problem: Compute $95^2$ in your head. [/INST]
# Solution: Let's think step by step. We have $(90 + 5)^2 = 90^2 + 2(90)(5) + 5^2 = 8100 + 900 + 25 = \\boxed{9025}$. The final answer is: \\boxed{9025}
# [INST] Problem: If $2^8=16^x$, find $x$. [/INST]
# Solution: Let's think step by step. We can write $16$ as $2^4$. Therefore, we can write our equation as $2^8 = 2^{4 \\cdot x}$. Solving, we get that $x = \\boxed{2}$. The final answer is: \\boxed{2}###
# [INST] Problem: {problem} [/INST]
# Solution: Let's think step by step.""")

# note: this is the prompt from Lean4AI and it has a mistake, e.g., they refer to proofs where they aren't actually proofs
gpt_assistant_prompt = "You are an expert in the Lean formal math language."
gpt_user_prompt_original = (
"At the end of this explanation, I will give you 2 things. The first is a list of tuples that are the translations of entire proofs written in Lean, "
"which we will denote the formal language, to plain English, also known as natural language, as tuples or pairs. This is not an exhaustive list, these "
"are just examples of informalizations. I will then have a proof written in Lean represented as a string following the newline character after the list of pairs.\n"
"Give me the tuple pair of the proof I give you written in Lean and what you think their natural language equivalent is given your knowledge of Lean, "
"formatted using LaTeX. Do not output anything else, just the python tuple I requested. In your output match the exact format \"('formal', 'informal')\"\n"
"[\"theorem exists_le_sylow {p : ℕ} {G : Type*} [group G] {P : subgroup G} \n (hP : is_p_group p P) : \n ∃ (Q : sylow p G), P ≤ Q :=\", \"Let $P$ be a $p$-subgroup of $G$. "
"Then $P$ is contained in a Sylow $p$-subgroup of $G$.\"),(\"theorem exists_eq_const_of_bounded {E : Type u} [normed_group E] \n [normed_space ℂ E] {F : Type v} [normed_group F] "
"[normed_space ℂ F] \n {f : E → F} (hf : differentiable ℂ f) (hb : metric.bounded (set.range f)) : \n ∃ (c : F), f = function.const E c :=\", \"Let $E$ and $F$ be complex normed spaces "
"and let $f:E\\to F$. If $f$ is differentiable and bounded, then $f$ is constant.\"), (\"theorem subset_of_open_subset_is_open (X : Type*) [topological_space X] \n (A : set X) "
"(hA : ∀ x ∈ A, ∃ U : set X, is_open U ∧ x ∈ U ∧ U ⊆ A): \n is_open A :=\",\"Let $X$ be a topological space; let $A$ be a subset of $X$. Suppose that for each $x\\in A$ there is an open set $U$ "
"containing $x$ such that $U\\subset A$. Then $A$ is open in $X$.\"), (\"theorem is_multiplicative.eq_iff_eq_on_prime_powers {R : Type*} \n [comm_monoid_with_zero R] "
"(f : nat.arithmetic_function R) \n (hf : f.is_multiplicative) (g : nat.arithmetic_function R) \n (hg : g.is_multiplicative) : \n f = g ↔ ∀ (p i : ℕ), nat.prime p → f (p ^ i) = g (p ^ i) :=\","
"\"Two multiplicative functions $f, g:\\mathbb{N}\\to R$ are equal if and only if $f(p^i)=f(g^i)$ for all primes $p$.\"), (\"theorem abs_sum_leq_sum_abs (n : ℕ) (f : ℕ → ℂ) : \n abs "
"(∑ i in finset.range n, f i) ≤ ∑ i in finset.range n, abs (f i) :=\",\"If $z_1, \\\\dots, z_n$ are complex, then $|z_1 + z_2 + \\\\dots + z_n|\\\\leq |z_1| + |z_2| + |z_n|\".), "
"(\"theorem distinct_powers_of_infinite_order_element (G : Type*) [group G] (x : G) \n (hx_inf : ∀ n : ℕ, x ^ n ≠ 1) : \n ∀ m n : ℤ, m ≠ n → x ^ m ≠ x ^ n :=\",\"If $x$ is an element "
"of infinite order in $G$, prove that the elements $x^n$, $n\\\\in\\\\mathbb{Z}$ are all distinct.\")]\n"
)

STOP_TOKENS: list[str] = ["Solution:", "Problem:", "Question:", "USER:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
# STOP_TOKENS: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS: list[str] = ["Question:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS_worse: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]

# current prompt template
prompt_af_for_align_zero_shot: str = (
    "You are an expert mathematician and a master of the Lean 4 programming language, "
    "with advanced knowledge in autoformalization techniques. Your task is to translate "
    "(autoformalize) the following English mathematical statements into precise Lean 4 code, "
    "effectively autoformalizing them. Focus on accuracy, correctness, and adherence to formal proof standards. "
    "The translation should reflect your deep understanding of both the mathematical concepts and the correct syntax of Lean 4.\n\n"
    "You will give your answer in the following format:\n<english>ENGLISH</english>\n<lean_code4>LEAN_CODE4</lean_code4>\n"
    "Please translate (autoformalize) the following mathematical statement in English into Lean 4:\n"
    "<english>{$ENGLISH}</english>\n<lean_code4>"
)
# q: curious english vs nl? but keep lean4, better ouput?
# q: wonder what happens if input is a find answer problem and not a proof problem, is it a good formalization?
# q: <tag></tag> vs other formats? list of tuples like lean4ai team did? best for deepseekmath, https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct/discussions/3
# q: lean-4 vs lean4 vs lean 4?

def prompt_temp_2_prompt_af_for_align(data_pt: dict, prompt_template: str = prompt_af_for_align1) -> str:
    english: str = data_pt['english']
    prompt: str = prompt_template.replace("{$ENGLISH}", english)
    return prompt

examples = [
    ("theorem exercise_1_1a (x : ℝ) (y : ℚ) : ( irrational x ) -> irrational ( x + y ) :=",
     "If $r$ is rational $(r \\neq 0)$ and $x$ is irrational, prove that $r+x$ is irrational."),
    
    ("theorem exercise_13_1 (X : Type*) [topological_space X] (A : set X) (h1 : ∀ x ∈ A, ∃ U : set X, x ∈ U ∧ is_open U ∧ U ⊆ A) : is_open A :=",
     "Let $X$ be a topological space; let $A$ be a subset of $X$. Suppose that for each $x \\in A$ there is an open set $U$ containing $x$ such that $U \\subset A$. Show that $A$ is open in $X$."),
    
    ("theorem exercise_1_3 {F V : Type*} [add_comm_group V] [field F] [module F V] {v : V} : -(-v) = v :=",
     "Prove that $-(-v) = v$ for every $v \\in V$."),
    
    ("theorem exercise_1_27 {n : ℕ} (hn : odd n) : 8 ∣ (n^2 - 1) :=",
     "For all odd $n$ show that $8 \\mid n^{2}-1$."),
    
    ("theorem exercise_1_13a {f : ℂ → ℂ} (Ω : set ℂ) (a b : Ω) (h : is_open Ω) (hf : differentiable_on ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).re = c) : f a = f b :=",
     "Suppose that $f$ is holomorphic in an open set $\\Omega$. Prove that if $\\text{Re}(f)$ is constant, then $f$ is constant."),
    
    ("theorem exercise_2020_b5 (z : fin 4 → ℂ) (hz0 : ∀ n, ‖z n‖ < 1) (hz1 : ∀ n : fin 4, z n ≠ 1) : 3 - z 0 - z 1 - z 2 - z 3 + (z 0) * (z 1) * (z 2) * (z 3) ≠ 0 :=",
     "For $j \\in\\{1,2,3,4\\}$, let $z_{j}$ be a complex number with $\\left|z_{j}\\right|=1$ and $z_{j} \\neq 1$. Prove that $3-z_{1}-z_{2}-z_{3}-z_{4}+z_{1} z_{2} z_{3} z_{4} \\neq 0 .$"),
    
    ("theorem exercise_2_12a (f : ℕ → ℕ) (p : ℕ → ℝ) (a : ℝ) (hf : injective f) (hp : tendsto p at_top (𝓝 a)) : tendsto (λ n, p (f n)) at_top (𝓝 a) :=",
     "Let $(p_n)$ be a sequence and $f:\\mathbb{N}\\to\\mathbb{N}$. The sequence $(q_k)_{k\\in\\mathbb{N}}$ with $q_k=p_{f(k)}$ is called a rearrangement of $(p_n)$. Show that if $f$ is an injection, the limit of a sequence is unaffected by rearrangement."),
    
    ("theorem exercise_2_1_21 (G : Type*) [group G] [fintype G] (hG : card G = 5) : comm_group G :=",
     "Show that a group of order 5 must be abelian.")
]

# Create the first prompt with 4 examples
af_4_pf_val_prompt = prompt_af_for_align_zero_shot + str(examples[:4])

# Create the second prompt with the remaining 8 examples
af_8_pf_val_prompt = prompt_af_for_align_zero_shot + str(examples)