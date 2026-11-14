from datasets import Dataset, concatenate_datasets, load_dataset

SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text"),
    "math": ("HuggingFaceTB/finemath", "finemath-3plus", "text"),
    "math_high": ("HuggingFaceTB/finemath", "finemath-4plus", "text"),
    "infiwebmath": ("HuggingFaceTB/finemath", "infiwebmath-3plus", "text"),
}


def source_counts(ratios: dict[str, float], num_docs: int) -> dict[str, int]:
    """Docs drawn from each source for a given total and ratio mix."""
    total = sum(ratios.values())
    return {name: round(ratio / total * num_docs) for name, ratio in ratios.items()}


def get_dataset(ratios: dict[str, float], num_docs: int, seed: int = 0) -> Dataset:
    parts = []
    for name, count in source_counts(ratios, num_docs).items():
        repo, config, column = SOURCES[name]
        stream = load_dataset(repo, config, split="train", streaming=True)
        docs = [{"text": doc[column]} for doc in stream.take(count)]
        parts.append(Dataset.from_list(docs))
    return concatenate_datasets(parts).shuffle(seed=seed)
