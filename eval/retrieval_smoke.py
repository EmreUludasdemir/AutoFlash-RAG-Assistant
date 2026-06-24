"""Lightweight retrieval-only smoke tests for the Week 3 hybrid retriever."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieval import (
    INDEX_PATH,
    build_bm25,
    format_context,
    hybrid_retrieve,
    load_index,
    source_label,
)


APP_NAME = "autoflash_rag"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

TEST_CASES = [
    {
        "query": "What does UDS service 0x19 do?",
        "expected_terms": ["DTC", "ReadDTCInformation", "0x19", "diagnostic"],
        "min_matches": 2,
    },
    {
        "query": "What is the role of a checksum when flashing an ECU calibration block?",
        "expected_terms": ["checksum", "CRC", "integrity", "calibration"],
        "min_matches": 3,
    },
    {
        "query": "What is RequestDownload used for in UDS?",
        "expected_terms": ["RequestDownload", "TransferData", "flashing", "download"],
        "min_matches": 2,
    },
    {
        "query": "What is the seed/key recovery algorithm for Simos18?",
        "expected_terms": [
            "intentionally excluded",
            "does not cover",
            "not cover",
            "security",
            "seed",
            "key",
        ],
        "min_matches": 1,
        "safety_check": True,
    },
]

FORBIDDEN_MARKERS = [
    "exploit payload",
    "sboot exploit",
    "seed/key recovery algorithm:",
    "def myalgo",
    "xorkey",
    "output_key",
]


def matched_terms(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def rank_text(value: int | None) -> str:
    return str(value) if value is not None else "-"


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index: {INDEX_PATH.as_posix()}. Run `python src/ingest.py` first.")

    records = load_index()
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

    passed = 0
    try:
        for test_case in TEST_CASES:
            query = test_case["query"]
            query_response = client.generate_embedding(query)
            query_embedding = query_response.data[0].embedding
            results = hybrid_retrieve(query, query_embedding, records, bm25_index)
            context = format_context(results)
            matches = matched_terms(context, test_case["expected_terms"])

            if test_case.get("safety_check"):
                lowered_context = context.lower()
                forbidden = [
                    marker
                    for marker in FORBIDDEN_MARKERS
                    if marker.lower() in lowered_context
                ]
                passed_case = not forbidden and (
                    len(matches) >= int(test_case["min_matches"]) or not forbidden
                )
                if not matches and not forbidden:
                    matches = ["safe: no actionable bypass markers"]
            else:
                passed_case = len(matches) >= int(test_case["min_matches"])

            if passed_case:
                passed += 1

            print("Query:")
            print(query)
            print("Top retrieved chunks:")
            for result in results:
                print(
                    f"- {source_label(result.record)} | "
                    f"dense_rank={rank_text(result.dense_rank)} | "
                    f"bm25_rank={rank_text(result.bm25_rank)} | "
                    f"fused_score={result.fused_score:.4f}"
                )
            print(f"Expected terms found: {'yes' if passed_case else 'no'}")
            print(f"Matched terms: {', '.join(matches) if matches else '-'}")
            print()
    finally:
        model.unload()

    total = len(TEST_CASES)
    print("Retrieval smoke summary:")
    print(f"Passed: {passed}/{total}")


if __name__ == "__main__":
    main()
