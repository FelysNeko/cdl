from itertools import islice

from datasets import Dataset, concatenate_datasets, load_dataset

SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text"),
    "math": ("HuggingFaceTB/finemath", "finemath-3plus", "text"),
    "math_high": ("HuggingFaceTB/finemath", "finemath-4plus", "text"),
    "infiwebmath": ("HuggingFaceTB/finemath", "infiwebmath-3plus", "text"),
}


def get_dataset(ratios: dict[str, float], num_docs: int, seed: int = 0) -> Dataset:
    total = sum(ratios.values())
    parts = []
    for name, ratio in ratios.items():
        repo, config, column = SOURCES[name]
        stream = load_dataset(repo, config, split="train", streaming=True)
        docs = [
            {"text": doc[column]}
            for doc in islice(stream, round(ratio / total * num_docs))
        ]
        parts.append(Dataset.from_list(docs))
    return concatenate_datasets(parts).shuffle(seed=seed)
