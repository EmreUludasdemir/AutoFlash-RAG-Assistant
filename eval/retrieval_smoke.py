"""Compare dense-only and hybrid retrieval over a small labeled eval set."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieval import (
    INDEX_PATH,
    TOP_K,
    build_bm25,
    dense_rank,
    hybrid_rank,
    load_index,
    source_label,
    tokenize,
)


APP_NAME = "autoflash_rag"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
EVAL_SET_PATH = REPO_ROOT / "eval" / "eval_set.json"
NO_RELEVANT_SENTINEL = 999


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError(f"Eval set must be a JSON list: {path}")
    return cases


def is_relevant(record: dict[str, Any], relevant_substrings: list[str]) -> bool:
    text = str(record.get("text", "")).lower()
    return any(fragment in text for fragment in relevant_substrings)


def first_relevant_rank(
    ranking: list[int],
    records: list[dict[str, Any]],
    relevant_substrings: list[str],
) -> int:
    for rank, record_index in enumerate(ranking, start=1):
        if is_relevant(records[record_index], relevant_substrings):
            return rank
    return NO_RELEVANT_SENTINEL


def hit_at_k(
    ranking: list[int],
    records: list[dict[str, Any]],
    relevant_substrings: list[str],
    k: int = TOP_K,
) -> int:
    return int(
        any(is_relevant(records[record_index], relevant_substrings) for record_index in ranking[:k])
    )


def top_sources(ranking: list[int], records: list[dict[str, Any]], k: int = TOP_K) -> list[str]:
    return [source_label(records[record_index]) for record_index in ranking[:k]]


def reciprocal_rank(rank: int) -> float:
    return 0.0 if rank == NO_RELEVANT_SENTINEL else 1.0 / rank


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index: {INDEX_PATH.as_posix()}. Run `python src/ingest.py` first.")

    records = load_index()
    cases = load_eval_set()
    bm25_index = build_bm25(records)

    config = Configuration(app_name=APP_NAME)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    model = manager.catalog.get_model(EMBEDDING_MODEL)
    if model is None:
        raise RuntimeError(f"Embedding model not found: {EMBEDDING_MODEL}")

    model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    print()
    model.load()
    client = model.get_embedding_client()

    dense_hits = 0
    hybrid_hits = 0
    dense_rr = 0.0
    hybrid_rr = 0.0

    try:
        for case in cases:
            query = str(case["query"])
            relevant_substrings = [
                str(fragment).lower() for fragment in case["relevant_substrings"]
            ]

            query_response = client.generate_embedding(query)
            query_embedding = query_response.data[0].embedding

            dense_ranking = [index for index, _ in dense_rank(query_embedding, records)]
            hybrid_ranking = hybrid_rank(
                query_embedding,
                tokenize(query),
                records,
                top_k=len(records),
                bm25_index=bm25_index,
            )

            dense_hit = hit_at_k(dense_ranking, records, relevant_substrings)
            hybrid_hit = hit_at_k(hybrid_ranking, records, relevant_substrings)
            dense_first = first_relevant_rank(
                dense_ranking,
                records,
                relevant_substrings,
            )
            hybrid_first = first_relevant_rank(
                hybrid_ranking,
                records,
                relevant_substrings,
            )

            dense_hits += dense_hit
            hybrid_hits += hybrid_hit
            dense_rr += reciprocal_rank(dense_first)
            hybrid_rr += reciprocal_rank(hybrid_first)

            print(query)
            print(
                f"  dense  : hit@{TOP_K}={dense_hit}  "
                f"first_rel_rank={dense_first}  "
                f"top sources={top_sources(dense_ranking, records)}"
            )
            print(
                f"  hybrid : hit@{TOP_K}={hybrid_hit}  "
                f"first_rel_rank={hybrid_first}  "
                f"top sources={top_sources(hybrid_ranking, records)}"
            )
            print()
    finally:
        model.unload()

    total = len(cases)
    print(f"=== Retrieval comparison ({total} cases) ===")
    print("              dense    hybrid")
    print(f"hit@{TOP_K:<10}{dense_hits}/{total:<7}{hybrid_hits}/{total}")
    print(f"MRR           {dense_rr / total:.2f}     {hybrid_rr / total:.2f}")


if __name__ == "__main__":
    main()
