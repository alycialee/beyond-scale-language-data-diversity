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

# -- 5 subsets of pile

def get_uniform_data_mixture_5subsets_of_pile(name: list = [None, None, None, None, None]) -> typle[list[float], str]:
    """
    Default is 5 subsets of pile. 
    """
    probabilities = [1/len(name)] * len(name)
    mixture_name = f'{probabilities}'
    return probabilities, mixture_name

def get_doremi_data_mixture_5subsets_of_pile(name: list) -> typle[list[float], str]:
    """
    Default is 5 subsets of pile. 

    ['sep_ds'] + ['hacker_news', 'nih_exporter', 'pubmed', 'uspto']
    """
    # hardcoded dictionary with doremi mixtures 
    mix_doremi = {
        'sep_ds': 0.23,  # pile-cc
        'hacker_news': 0.45,
        'nih_exporter': 0.67,
        'pubmed': 0.89,
        'uspto': 0.12
    }
    # - transform mixtures to respect the original dore mi

    # - convert the mix doremi to list respect order given by user
    probabilities = []
    for subset_name in name:
        mixture_subset = doremi_mixture_value[name]
        probabilities.append(mixture_subset)
    mixture_name = 'doremi_5subsets_of_pile'
    return probabilities, mixture_name

