"""
{
    "text": ...,
    "SubSet": "CommomCrawl" | "StackExchange" | "Textbooks" | "Wikipedia" | "ProofWiki" | "arXiv"
    "meta": {"language_detection_score": , "idx": , "contain_at_least_two_stop_words": ,
}
"""

from datasets import load_dataset

dataset = load_dataset()