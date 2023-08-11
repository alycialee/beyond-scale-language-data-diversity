"""
Estimates on data mixtures from lit.
idea respect METHOD relative ratios according to similarity to coresponding datasets.

I think for now we support
- uniform
- doremi_based
- llama v1

- decided to exclude
    - the pile (their models aren't good enough)
    - gpt3 (too similar to llama v1)
    - llama v2 (nda, so can't do!)

note: it's usually nda, trade secret. 
"""

def get_uniform_data_mixture_for_c4_wt103() -> list[float]:
    return [0.5, 0.5], 'uniform'

def get_doremi_based_data_mixture_for_c4_wt103() -> list[float]:
    """ idea respect METHOD relative ratios according to similarity to coresponding datasets """
    doremi_pile_cc = 0.6057 # pile-cc used originally, cloesest to c4
    doremi_wikiepdia = 0.0699  # wikipedia en used originally
    doremi_probabilies = [doremi_pile_cc, doremi_wikiepdia]
    # new probabilties
    c4 = doremi_pile_cc / (doremi_wikiepdia + doremi_pile_cc)
    wt103 = doremi_wikiepdia / (doremi_wikiepdia + doremi_pile_cc)
    probabilities_c4_wt103 = [c4, wt103]
    # - print ratios
    print('Make sure ratios are similar')
    print(f'{doremi_probabilies=}')
    print(f'{probabilities_c4_wt103=}')
    print(f'{doremi_pile_cc/doremi_wikiepdia=}')
    print(f'{c4/wt103=}')
    return probabilities_c4_wt103, 'doremi_based'

def get_llama_v1_based_data_mixture_for_c4_wt103() -> list[float]:
    """ idea respect METHOD relative ratios according to similarity to coresponding datasets """
    llama_v1_c4 = 0.15 # llama v1 also uses cc, but decided to exclude it since c4 was exactly given + its more different than doremi which has a large weight to cc anyway
    llama_v1_wikiepdia = 0.045  # wikipedia en used originally
    llama_v1_probabilies = [llama_v1_c4, llama_v1_wikiepdia]
    # new probabilties
    c4 = llama_v1_c4 / (llama_v1_wikiepdia + llama_v1_c4)
    wt103 = llama_v1_wikiepdia / (llama_v1_wikiepdia + llama_v1_c4)
    probabilities_c4_wt103 = [c4, wt103]
    # - print ratios
    print('Make sure ratios are similar')
    print(f'{llama_v1_probabilies=}')
    print(f'{probabilities_c4_wt103=}')
    print(f'{llama_v1_c4/llama_v1_wikiepdia=}')
    print(f'{c4/wt103=}')
    return probabilities_c4_wt103, 'llama_v1_based'