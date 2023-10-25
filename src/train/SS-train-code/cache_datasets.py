from datasets import load_dataset

"""
Use: 
- Run this file to cache the Dataset data for all desired subsets. Data is cached in your LFS cache.
- E.g. run ```python [path to folder]/cache_datasets.py``` while located in your LFS folder.

Reason: 
- Loading a Dataset variable from cache (happens by default when the dataset is in the cache), then converting to an 
IterableDataset allows you to work with the data much *faster* and more *reliably* than streaming directly from the 
HF server, e.g. when computing the diversity coefficient of a dataset.

Default cache location: 
~/.cache/huggingface/datasets

Set custom cache location by running this in termnial:
export HF_DATASETS_CACHE="/path/to/another/directory"

Or modifying the load_dataset calls this way:
load_dataset('Name', 'subset', split='train').with_format('torch') --> load_dataset('Name', 'subset', split='train', cache_dir='PATH/TO/MY/CACHE/DIR').with_format('torch')
"""


def main():
    # Each call leads to the given dataset being downloaded and stored in cache. Feel free to add datasets by following the same format.
    # The following 9 datasets (ending with uspto_val) are relatively small datasets and won't take too much time/space to cache (< ~20 GB for the most part)
    pubmed_train = load_dataset('SudharsanSundar/PileSubsets', 'pubmed', split='train').with_format('torch')
    uspto_train = load_dataset('SudharsanSundar/PileSubsets', 'USPTO', split='train').with_format('torch')
    wikitext_val = load_dataset('wikitext', 'wikitext-103-v1', split='validation').with_format('torch')
    opensubs_val = load_dataset('suolyer/pile_opensubtitles', split='validation').with_format('torch')
    nihexporter_val = load_dataset('suolyer/pile_nih-exporter', split='validation').with_format('torch')
    hnews_val = load_dataset('suolyer/pile_hackernews', split='validation').with_format('torch')
    tinystories_val = load_dataset('roneneldan/TinyStories', split='validation').with_format('torch')
    pubmed_val = load_dataset('suolyer/pile_pubmed-abstracts', split='validation').with_format('torch')
    uspto_val = load_dataset('suolyer/pile_uspto', split='validation').with_format('torch')

    # These two are really large datasets, so caching them will take a lot of time/space
    openwebtext2_val = load_dataset('suolyer/pile_openwebtext2', split='validation').with_format('torch')       # ~60 GB
    pile_all_train = load_dataset('SudharsanSundar/PileSubsets', 'all', split='train').with_format('torch')     # > ~800 GB. NOTE: There might already be a cached version on SNAP somewhere--ask Rok for specifics


if __name__ == "__main__":
    main()
