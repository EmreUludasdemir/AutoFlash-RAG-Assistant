"""Compare dense, hybrid, and hybrid+rerank retrieval over a labeled eval set."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from foundry_setup import initialize_manager, load_model_with_webgpu_fallback
from retrieval import (
    INDEX_PATH,
    RERANK_CANDIDATES,
    RERANK_GATE,
    TOP_K,
    build_bm25,
    dense_rank,
    hybrid_rank,
    load_index,
    retrieve_reranked,
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


def update_metrics(
    metrics: dict[str, dict[str, float]],
    mode: str,
    hit: int,
    first_rank: int,
) -> None:
    metrics[mode]["hits"] += hit
    metrics[mode]["rr"] += reciprocal_rank(first_rank)


def print_mode_line(
    mode: str,
    hit: int,
    first_rank: int,
    sources: list[str],
    best_score: float | None = None,
) -> None:
    score_text = "" if best_score is None else f"  best_rerank_score={best_score:.3f}"
    print(
        f"  {mode:<14}: hit@{TOP_K}={hit}  "
        f"first_rel_rank={first_rank}{score_text}  top sources={sources}"
    )


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index: {INDEX_PATH.as_posix()}. Run `python src/ingest.py` first.")

    records = load_index()
    cases = load_eval_set()
    bm25_index = build_bm25(records)

    manager = initialize_manager(APP_NAME)
    model, model_status = load_model_with_webgpu_fallback(
        manager,
        EMBEDDING_MODEL,
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True),
    )
    print()
    print(
        f"Embedding model loaded: {model_status.model_id} "
        f"({model_status.device}/{model_status.execution_provider})"
    )
    client = model.get_embedding_client()

    metrics = {
        "dense": {"hits": 0.0, "rr": 0.0},
        "hybrid": {"hits": 0.0, "rr": 0.0},
        "hybrid+rerank": {"hits": 0.0, "rr": 0.0},
    }
    in_scope_count = 0
    in_scope_scores: list[float] = []
    out_of_scope_score: float | None = None
    out_of_scope_query = ""

    try:
        for case in cases:
            query = str(case["query"])
            query_tokens = tokenize(query)
            query_response = client.generate_embedding(query)
            query_embedding = query_response.data[0].embedding
            reranked_results, best_rerank_score = retrieve_reranked(
                query,
                query_embedding,
                query_tokens,
                records,
                top_k=RERANK_CANDIDATES,
                bm25_index=bm25_index,
            )
            reranked_ranking = [result.record_index for result in reranked_results]

            if case.get("out_of_scope"):
                out_of_scope_score = best_rerank_score
                out_of_scope_query = query
                below_gate = "yes" if best_rerank_score < RERANK_GATE else "no"
                print(query)
                print(
                    f"  out-of-scope: best_rerank_score={best_rerank_score:.3f}  "
                    f"below gate({RERANK_GATE})? {below_gate}  "
                    f"top sources={top_sources(reranked_ranking, records)}"
                )
                print()
                continue

            relevant_substrings = [
                str(fragment).lower() for fragment in case["relevant_substrings"]
            ]
            dense_ranking = [index for index, _ in dense_rank(query_embedding, records)]
            hybrid_ranking = hybrid_rank(
                query_embedding,
                query_tokens,
                records,
                top_k=len(records),
                bm25_index=bm25_index,
            )

            dense_hit = hit_at_k(dense_ranking, records, relevant_substrings)
            hybrid_hit = hit_at_k(hybrid_ranking, records, relevant_substrings)
            rerank_hit = hit_at_k(reranked_ranking, records, relevant_substrings)
            dense_first = first_relevant_rank(dense_ranking, records, relevant_substrings)
            hybrid_first = first_relevant_rank(hybrid_ranking, records, relevant_substrings)
            rerank_first = first_relevant_rank(reranked_ranking, records, relevant_substrings)

            update_metrics(metrics, "dense", dense_hit, dense_first)
            update_metrics(metrics, "hybrid", hybrid_hit, hybrid_first)
            update_metrics(metrics, "hybrid+rerank", rerank_hit, rerank_first)
            in_scope_count += 1
            in_scope_scores.append(best_rerank_score)

            print(query)
            print_mode_line(
                "dense",
                dense_hit,
                dense_first,
                top_sources(dense_ranking, records),
            )
            print_mode_line(
                "hybrid",
                hybrid_hit,
                hybrid_first,
                top_sources(hybrid_ranking, records),
            )
            print_mode_line(
                "hybrid+rerank",
                rerank_hit,
                rerank_first,
                top_sources(reranked_ranking, records),
                best_rerank_score,
            )
            print()
    finally:
        model.unload()

    print(f"=== Retrieval comparison ({in_scope_count} in-scope cases) ===")
    print("              dense    hybrid   hybrid+rerank")
    print(
        f"hit@{TOP_K:<10}"
        f"{int(metrics['dense']['hits'])}/{in_scope_count:<7}"
        f"{int(metrics['hybrid']['hits'])}/{in_scope_count:<9}"
        f"{int(metrics['hybrid+rerank']['hits'])}/{in_scope_count}"
    )
    print(
        "MRR           "
        f"{metrics['dense']['rr'] / in_scope_count:.2f}     "
        f"{metrics['hybrid']['rr'] / in_scope_count:.2f}     "
        f"{metrics['hybrid+rerank']['rr'] / in_scope_count:.2f}"
    )
    print()
    print("Out-of-scope gate check:")
    if out_of_scope_score is None:
        print("  no out-of-scope case found")
    else:
        below_gate = "yes" if out_of_scope_score < RERANK_GATE else "no"
        label = "seed/key" if "seed/key" in out_of_scope_query.lower() else out_of_scope_query
        print(
            f"  {label} best_rerank_score={out_of_scope_score:.3f}  "
            f"below gate({RERANK_GATE})? {below_gate}"
        )

    if in_scope_scores:
        print(
            "In-scope best_rerank_score range: "
            f"min={min(in_scope_scores):.3f} max={max(in_scope_scores):.3f}"
        )


if __name__ == "__main__":
    main()
