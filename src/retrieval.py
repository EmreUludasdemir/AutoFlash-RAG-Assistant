"""Shared retrieval helpers for AutoFlash RAG.

Implements dense vector ranking, BM25 lexical ranking, and Reciprocal Rank
Fusion (RRF) over the persisted data/index.json records.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


INDEX_PATH = Path("data/index.json")
TOP_K = 4
DENSE_CANDIDATES = 20
BM25_CANDIDATES = 20
RRF_K = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20
RERANK_GATE = 0.0
TOKEN_RE = re.compile(r"0x[0-9a-f]+|[^\W_]+", re.IGNORECASE | re.UNICODE)
_RERANKER = None


@dataclass(frozen=True)
class RetrievalResult:
    record_index: int
    record: dict[str, Any]
    fused_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float | None
    bm25_score: float | None
    rerank_score: float | None = None


def load_index(index_path: Path = INDEX_PATH) -> list[dict[str, Any]]:
    """Load the persisted chunk index."""
    with index_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Index must be a JSON list: {index_path}")
    return records


def tokenize(text: str) -> list[str]:
    """Tokenize diagnostic text while preserving terms like 0x19 and DTC."""
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def build_bm25(records: list[dict[str, Any]]) -> BM25Okapi:
    corpus = [tokenize(str(record.get("text", ""))) for record in records]
    return BM25Okapi(corpus)


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def dense_rank(
    query_embedding,
    records: list[dict[str, Any]],
    limit: int | None = None,
) -> list[tuple[int, float]]:
    scored = []
    for index, record in enumerate(records):
        embedding = record.get("embedding")
        if embedding:
            scored.append((index, cosine_similarity(query_embedding, embedding)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit] if limit is not None else scored


def dense_ranking(
    query_embedding,
    records: list[dict[str, Any]],
    limit: int = DENSE_CANDIDATES,
) -> list[tuple[int, float]]:
    return dense_rank(query_embedding, records, limit=limit)


def bm25_rank(
    query_tokens: list[str],
    records: list[dict[str, Any]],
    bm25_index: BM25Okapi | None = None,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    index = bm25_index if bm25_index is not None else build_bm25(records)
    scores = index.get_scores(query_tokens)
    ranked = [(index, float(score)) for index, score in enumerate(scores)]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit] if limit is not None else ranked


def bm25_ranking(
    query: str,
    bm25_index: BM25Okapi,
    limit: int = BM25_CANDIDATES,
) -> list[tuple[int, float]]:
    return bm25_rank(tokenize(query), [], bm25_index=bm25_index, limit=limit)


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    records: list[dict[str, Any]],
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
) -> list[RetrievalResult]:
    fused: dict[int, dict[str, float | int | None]] = {}

    for rank, (index, score) in enumerate(dense_results, start=1):
        item = fused.setdefault(
            index,
            {
                "fused_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            },
        )
        item["fused_score"] = float(item["fused_score"]) + 1.0 / (rrf_k + rank)
        item["dense_rank"] = rank
        item["dense_score"] = score

    for rank, (index, score) in enumerate(bm25_results, start=1):
        item = fused.setdefault(
            index,
            {
                "fused_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            },
        )
        item["fused_score"] = float(item["fused_score"]) + 1.0 / (rrf_k + rank)
        item["bm25_rank"] = rank
        item["bm25_score"] = score

    ranked = sorted(
        fused.items(),
        key=lambda item: (
            float(item[1]["fused_score"]),
            -(int(item[1]["dense_rank"]) if item[1]["dense_rank"] else 10_000),
            -(int(item[1]["bm25_rank"]) if item[1]["bm25_rank"] else 10_000),
        ),
        reverse=True,
    )

    return [
        RetrievalResult(
            record_index=index,
            record=records[index],
            fused_score=float(item["fused_score"]),
            dense_rank=item["dense_rank"] if isinstance(item["dense_rank"], int) else None,
            bm25_rank=item["bm25_rank"] if isinstance(item["bm25_rank"], int) else None,
            dense_score=(
                float(item["dense_score"]) if item["dense_score"] is not None else None
            ),
            bm25_score=float(item["bm25_score"]) if item["bm25_score"] is not None else None,
        )
        for index, item in ranked[:top_k]
    ]


def hybrid_rank(
    query_embedding,
    query_tokens: list[str],
    records: list[dict[str, Any]],
    top_k: int = TOP_K,
    bm25_index: BM25Okapi | None = None,
) -> list[int]:
    index = bm25_index if bm25_index is not None else build_bm25(records)
    dense_results = dense_rank(query_embedding, records, limit=DENSE_CANDIDATES)
    bm25_results = bm25_rank(
        query_tokens,
        records,
        bm25_index=index,
        limit=BM25_CANDIDATES,
    )
    results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        records,
        top_k=top_k,
        rrf_k=RRF_K,
    )
    return [result.record_index for result in results]


def hybrid_retrieve(
    query: str,
    query_embedding,
    records: list[dict[str, Any]],
    bm25_index: BM25Okapi,
    top_k: int = TOP_K,
    dense_candidates: int = DENSE_CANDIDATES,
    bm25_candidates: int = BM25_CANDIDATES,
    rrf_k: int = RRF_K,
) -> list[RetrievalResult]:
    dense_results = dense_ranking(query_embedding, records, limit=dense_candidates)
    bm25_results = bm25_ranking(query, bm25_index, limit=bm25_candidates)
    return reciprocal_rank_fusion(dense_results, bm25_results, records, top_k, rrf_k)


def get_reranker():
    """Load the cross-encoder once on CPU."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder(RERANK_MODEL, device="cpu")
    return _RERANKER


def rerank(
    query: str,
    candidate_records: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, float]]:
    """Rerank candidate chunks with a CPU cross-encoder."""
    if not candidate_records:
        return []

    pairs = [
        (query, str(record.get("text", "")))
        for _, record in candidate_records
    ]
    scores = get_reranker().predict(pairs)
    ranked = [
        (record_index, float(score))
        for (record_index, _), score in zip(candidate_records, scores)
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def retrieve_reranked(
    query: str,
    query_embedding,
    query_tokens: list[str],
    records: list[dict[str, Any]],
    top_k: int = TOP_K,
    bm25_index: BM25Okapi | None = None,
) -> tuple[list[RetrievalResult], float]:
    candidate_indices = hybrid_rank(
        query_embedding,
        query_tokens,
        records,
        top_k=RERANK_CANDIDATES,
        bm25_index=bm25_index,
    )
    candidate_records = [(index, records[index]) for index in candidate_indices]
    ranked = rerank(query, candidate_records)
    best_score = ranked[0][1] if ranked else float("-inf")
    results = [
        RetrievalResult(
            record_index=index,
            record=records[index],
            fused_score=0.0,
            dense_rank=None,
            bm25_rank=None,
            dense_score=None,
            bm25_score=None,
            rerank_score=score,
        )
        for index, score in ranked[:top_k]
    ]
    return results, best_score


def source_label(record: dict[str, Any]) -> str:
    source = str(record.get("source", ""))
    section = str(record.get("section", ""))
    if section:
        return f"{source} § {section}"
    return source


def format_context(results: list[RetrievalResult]) -> str:
    """Build source-prefixed context blocks for the chat model."""
    blocks = []
    for result in results:
        text = str(result.record.get("text", "")).strip()
        blocks.append(f"[source: {source_label(result.record)}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def format_sources(results: list[RetrievalResult]) -> str:
    """Return a compact, stable source list for retrieved chunks."""
    sources = []
    for result in results:
        label = source_label(result.record)
        if label and label not in sources:
            sources.append(label)
    return "; ".join(sources) if sources else "none"
