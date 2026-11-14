"""Append-only cache for tokenized, fixed-length training blocks.

The cache is a single token file plus metadata::

    cache_dir/
        tokens.u16   [test blocks][train blocks ...]
        meta.json
        cache.lock

Test blocks are written once and never change, so perplexity stays comparable
across runs. Train blocks are appended in place when more iterations need more
data. The first build follows the old ``get_dataset`` path exactly; appends
continue each HuggingFace stream from its recorded per-source offset.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cdl.dataset.smollm_corpus import SOURCES, get_dataset, source_counts

logger = logging.getLogger(__name__)

CACHE_FORMAT = 1
CACHE_ROOT = Path("cache")
TOKENIZE_BATCH = 1024
APPEND_CHUNK_DOCS = 2048


@dataclass(frozen=True)
class CacheSpec:
    """Everything that determines the token stream and its split."""

    model_id: str
    tokenizer: PreTrainedTokenizerBase
    ratios: dict[str, float]
    seed: int
    seq_len: int
    num_docs: int
    num_test_blocks: int
    num_calib_blocks: int


@dataclass
class CacheMeta:
    format: int
    model_id: str
    tokenizer: str
    tokenizer_revision: str | None
    eos_token_id: int
    seq_len: int
    dtype: str
    ratios: dict[str, float]
    seed: int
    num_test_blocks: int
    num_calib_blocks: int
    n_train_blocks: int
    n_test_blocks: int
    tail: list[int]
    docs_consumed: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> CacheMeta:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(self), indent=2, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


class BlockDataset(Dataset):
    """Read fixed-length blocks from a memory-mapped token file."""

    def __init__(
        self,
        path: Path,
        dtype: np.dtype,
        offset_blocks: int,
        n_blocks: int,
        seq_len: int,
    ) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.offset_blocks = offset_blocks
        self.n_blocks = n_blocks
        self.seq_len = seq_len
        self._mmap: np.memmap | np.ndarray | None = None

    def _open(self) -> np.memmap | np.ndarray:
        if self._mmap is None:
            if self.n_blocks == 0:
                self._mmap = np.empty((0, self.seq_len), dtype=self.dtype)
            else:
                self._mmap = np.memmap(
                    self.path,
                    dtype=self.dtype,
                    mode="r",
                    offset=self.offset_blocks * self.seq_len * self.dtype.itemsize,
                    shape=(self.n_blocks, self.seq_len),
                )
        return self._mmap

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> torch.Tensor:
        block = np.asarray(self._open()[index], dtype=np.int64)
        return torch.from_numpy(block)

    def __getstate__(self) -> dict:
        # Do not pickle the memmap into DataLoader workers.
        state = self.__dict__.copy()
        state["_mmap"] = None
        return state


class TokenCache:
    def __init__(self, directory: Path, meta: CacheMeta, spec: CacheSpec) -> None:
        self.directory = Path(directory)
        self.meta = meta
        self.spec = spec
        self.tokens_path = self.directory / "tokens.u16"
        self.meta_path = self.directory / "meta.json"
        self.dtype = np.dtype(meta.dtype)
        self._refresh_datasets()

    @classmethod
    def open(
        cls,
        spec: CacheSpec,
        directory: str | Path | None = None,
        rebuild: bool = False,
        target_blocks: int = 0,
    ) -> TokenCache:
        path = Path(directory) if directory is not None else default_cache_dir(spec)
        path.mkdir(parents=True, exist_ok=True)
        with _cache_lock(path / "cache.lock"):
            if rebuild:
                _remove_cache_files(path)
            meta = _load_or_build(path, spec)
            cache = cls(path, meta, spec)
            cache._ensure_blocks_locked(target_blocks)
        return cache

    @property
    def train_dataset(self) -> BlockDataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> BlockDataset:
        return self._test_dataset

    @property
    def n_train_blocks(self) -> int:
        return self.meta.n_train_blocks

    def calib_blocks(self, n_blocks: int) -> torch.Tensor:
        n = min(n_blocks, self.meta.n_train_blocks)
        if n <= 0:
            return torch.empty((0, self.meta.seq_len), dtype=torch.long)
        blocks = np.asarray(self._train_dataset._open()[:n], dtype=np.int64)
        return torch.from_numpy(blocks)

    def _refresh_datasets(self) -> None:
        self._train_dataset = BlockDataset(
            self.tokens_path,
            self.dtype,
            self.meta.n_test_blocks,
            self.meta.n_train_blocks,
            self.meta.seq_len,
        )
        self._test_dataset = BlockDataset(
            self.tokens_path,
            self.dtype,
            0,
            self.meta.n_test_blocks,
            self.meta.seq_len,
        )

    def _ensure_blocks_locked(self, target_blocks: int) -> None:
        if target_blocks <= self.meta.n_train_blocks:
            return

        logger.info(
            "appending train data: %d -> %d blocks",
            self.meta.n_train_blocks,
            target_blocks,
        )
        while self.meta.n_train_blocks < target_blocks:
            texts, fetched = self._fetch_docs(APPEND_CHUNK_DOCS)
            tokens = _tokenize_texts(self.spec.tokenizer, texts, desc="append")
            self._append_tokens_locked(tokens)
            for name, count in fetched.items():
                self.meta.docs_consumed[name] = (
                    self.meta.docs_consumed.get(name, 0) + count
                )
            self.meta.save(self.meta_path)
            self._refresh_datasets()
            logger.info(
                "appended %d docs -> %d train blocks",
                sum(fetched.values()),
                self.meta.n_train_blocks,
            )

    def _fetch_docs(self, chunk_docs: int) -> tuple[list[str], dict[str, int]]:
        texts: list[str] = []
        fetched: dict[str, int] = {}
        for name, count in source_counts(self.spec.ratios, chunk_docs).items():
            if count <= 0:
                fetched[name] = 0
                continue
            repo, config, column = SOURCES[name]
            stream = load_dataset(repo, config, split="train", streaming=True)
            consumed = self.meta.docs_consumed.get(name, 0)
            if consumed:
                stream = stream.skip(consumed)
            # Fully consume take(count) so the HF streaming generator closes
            # cleanly; islice on a live iterator can hang interpreter shutdown.
            batch = list(stream.take(count))
            fetched[name] = len(batch)
            texts.extend(doc[column] for doc in batch)
        if not texts:
            raise RuntimeError(
                "data sources exhausted before reaching the requested block count"
            )
        return texts, fetched

    def _append_tokens_locked(self, tokens: np.ndarray) -> None:
        if tokens.size and int(tokens.max()) > np.iinfo(self.dtype).max:
            raise ValueError(
                f"new token id {int(tokens.max())} does not fit {self.dtype}"
            )
        tokens = tokens.astype(self.dtype, copy=False)
        tail = np.asarray(self.meta.tail, dtype=self.dtype)
        stream = np.concatenate([tail, tokens])
        n_full = len(stream) // self.meta.seq_len
        if n_full:
            expected_bytes = (
                (self.meta.n_test_blocks + self.meta.n_train_blocks)
                * self.meta.seq_len
                * self.dtype.itemsize
            )
            with open(self.tokens_path, "r+b") as f:
                f.seek(expected_bytes)
                f.truncate()
                f.seek(0, os.SEEK_END)
                stream[: n_full * self.meta.seq_len].tofile(f)
                f.flush()
                os.fsync(f.fileno())
            self.meta.n_train_blocks += n_full
        self.meta.tail = stream[n_full * self.meta.seq_len :].tolist()


def _tokenizer_revision(tokenizer: PreTrainedTokenizerBase) -> str | None:
    return (getattr(tokenizer, "init_kwargs", None) or {}).get("_commit_hash")


def default_cache_dir(spec: CacheSpec) -> Path:
    payload = {
        "model": spec.model_id,
        "tokenizer": spec.tokenizer.name_or_path,
        "revision": _tokenizer_revision(spec.tokenizer),
        "ratios": spec.ratios,
        "seed": spec.seed,
        "seq_len": spec.seq_len,
        "num_test_blocks": spec.num_test_blocks,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    slug = spec.model_id.replace("/", "--").replace(" ", "-")
    return CACHE_ROOT / f"{slug}-seq{spec.seq_len}-seed{spec.seed}-{digest}"


@contextmanager
def _cache_lock(path: Path) -> Iterator[None]:
    with open(path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _remove_cache_files(directory: Path) -> None:
    for name in ("tokens.u16", "tokens.u16.tmp", "meta.json", "meta.json.tmp"):
        (directory / name).unlink(missing_ok=True)


def _load_or_build(directory: Path, spec: CacheSpec) -> CacheMeta:
    meta_path = directory / "meta.json"
    tokens_path = directory / "tokens.u16"
    if meta_path.exists():
        meta = CacheMeta.load(meta_path)
        _validate_meta(meta, spec)
        _ensure_file_size(tokens_path, meta)
        return meta
    return _build_initial(directory, spec)


def _build_initial(directory: Path, spec: CacheSpec) -> CacheMeta:
    pool = get_dataset(spec.ratios, spec.num_docs, spec.seed)
    tokens = _tokenize_texts(spec.tokenizer, pool["text"], desc="tokenize")
    n_blocks = len(tokens) // spec.seq_len
    if n_blocks <= spec.num_test_blocks:
        raise ValueError(
            f"corpus produced only {n_blocks} blocks of seq_len={spec.seq_len}; "
            f"need more than num_test_blocks={spec.num_test_blocks}"
        )

    test_start = n_blocks - spec.num_test_blocks
    test_tokens = tokens[test_start * spec.seq_len : n_blocks * spec.seq_len]
    train_tokens = tokens[: test_start * spec.seq_len]

    tokens_path = directory / "tokens.u16"
    tmp_path = directory / "tokens.u16.tmp"
    with open(tmp_path, "wb") as f:
        test_tokens.tofile(f)
        train_tokens.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, tokens_path)

    meta = CacheMeta(
        format=CACHE_FORMAT,
        model_id=spec.model_id,
        tokenizer=spec.tokenizer.name_or_path,
        tokenizer_revision=_tokenizer_revision(spec.tokenizer),
        eos_token_id=spec.tokenizer.eos_token_id,
        seq_len=spec.seq_len,
        dtype=tokens.dtype.name,
        ratios=dict(spec.ratios),
        seed=spec.seed,
        num_test_blocks=spec.num_test_blocks,
        num_calib_blocks=spec.num_calib_blocks,
        n_train_blocks=test_start,
        n_test_blocks=spec.num_test_blocks,
        tail=[],
        docs_consumed=source_counts(spec.ratios, spec.num_docs),
    )
    meta.save(directory / "meta.json")
    logger.info(
        "built token cache %s: train %d, test %d, %d tokens",
        directory,
        meta.n_train_blocks,
        meta.n_test_blocks,
        len(tokens),
    )
    return meta


def _tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts,
    desc: str,
) -> np.ndarray:
    eos = tokenizer.eos_token_id
    ids = array("q")
    for start in tqdm.tqdm(range(0, len(texts), TOKENIZE_BATCH), desc=desc):
        batch = tokenizer(
            texts[start : start + TOKENIZE_BATCH], add_special_tokens=False
        )["input_ids"]
        for seq in batch:
            ids.extend(seq)
            ids.append(eos)
    tokens = np.asarray(ids, dtype=np.int64)
    if tokens.size == 0:
        return tokens.astype(np.uint16)
    max_id = int(tokens.max())
    if max_id <= np.iinfo(np.uint16).max:
        return tokens.astype(np.uint16)
    if max_id <= np.iinfo(np.uint32).max:
        return tokens.astype(np.uint32)
    raise ValueError(f"token id {max_id} does not fit in uint32")


def _validate_meta(meta: CacheMeta, spec: CacheSpec) -> None:
    checks = {
        "format": (meta.format, CACHE_FORMAT),
        "model_id": (meta.model_id, spec.model_id),
        "tokenizer": (meta.tokenizer, spec.tokenizer.name_or_path),
        "eos_token_id": (meta.eos_token_id, spec.tokenizer.eos_token_id),
        "seq_len": (meta.seq_len, spec.seq_len),
        "ratios": (meta.ratios, spec.ratios),
        "seed": (meta.seed, spec.seed),
        "num_test_blocks": (meta.num_test_blocks, spec.num_test_blocks),
    }
    for name, (have, want) in checks.items():
        if have != want:
            raise ValueError(
                f"token cache mismatch for {name}: cache has {have!r}, "
                f"config wants {want!r}; rebuild the cache or change cache_dir"
            )
    cache_revision = meta.tokenizer_revision
    spec_revision = _tokenizer_revision(spec.tokenizer)
    if cache_revision and spec_revision and cache_revision != spec_revision:
        raise ValueError(
            f"token cache tokenizer revision mismatch: {cache_revision} != "
            f"{spec_revision}; rebuild the cache"
        )


def _ensure_file_size(path: Path, meta: CacheMeta) -> None:
    if not path.exists():
        raise ValueError(f"token cache is missing {path}; rebuild the cache")
    dtype = np.dtype(meta.dtype)
    expected = (
        (meta.n_test_blocks + meta.n_train_blocks) * meta.seq_len * dtype.itemsize
    )
    actual = path.stat().st_size
    if actual < expected:
        raise ValueError(
            f"token cache {path} is truncated: {actual} < {expected} bytes"
        )
    if actual > expected:
        # A crashed append left bytes past the last committed block count.
        with open(path, "r+b") as f:
            f.truncate(expected)
