"""Offline answer-quality evaluation for the local AutoFlash RAG Assistant."""

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
from main import (  # noqa: E402
    APP_NAME,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    abstention_answer,
    direct_grounded_answer,
    is_out_of_scope_security_query,
    is_turkish_query,
    retrieval_query_for_rerank,
)
from retrieval import (  # noqa: E402
    INDEX_PATH,
    RERANK_GATE,
    build_bm25,
    format_context,
    format_sources,
    load_index,
    retrieve_reranked,
    source_label,
    tokenize,
)


EVAL_SET_PATH = REPO_ROOT / "eval" / "answer_eval_set.json"
REPORT_PATH = REPO_ROOT / "eval" / "answer_quality_report.json"

BASE_ABSTAIN_MARKERS = (
    "i don't know",
    "out of scope",
    "not covered",
    "intentionally excluded",
    "not in the provided context",
    "security-bypass",
    "bilmiyorum",
    "kapsam dışı",
    "bağlamda yok",
    "bağlamda bulunmuyor",
    "kaynaklarda yok",
    "yer almıyor",
    "güvenlik",
    "bypass",
)
EXTRA_ABSTAIN_MARKERS = (
    "provided engineering context",
    "provided documents",
    "local engineering corpus",
    "dokümanlarında yok",
    "dokumanlarinda yok",
    "yanıt veremiyorum",
    "yanit veremiyorum",
    "tahmin yürüterek",
    "tahmin yuruterek",
)
CITATION_MARKERS = (
    "sources:",
    "source:",
    "data/raw/",
    ".pdf",
    ".md",
    ".rst",
    "§",
)


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError(f"Eval set must be a JSON list: {path}")
    return cases


def grounded_messages(query: str, context: str) -> list[dict[str, str]]:
    language_instruction = (
        "The user's question is in Turkish. Answer in Turkish."
        if is_turkish_query(query)
        else "The user's question is in English. Answer in English."
    )
    return [
        {
            "role": "system",
            "content": (
                f"{language_instruction} Keep the answer concise: 2-4 sentences. "
                "Answer the user's question using only the provided context. "
                "Cite the source(s) you used at the end as `Sources: <source list>`. "
                "If the context does not contain the answer, say you don't know "
                "rather than guessing. Do not use outside knowledge. "
                "When the question names an exact identifier or service such as "
                "0x19 or RequestDownload, focus only on context about that exact "
                "identifier or service and ignore unrelated identifiers. Do not "
                "invent protocols, transports, or examples that are not in context. "
                "If the user asks in Turkish, answer in Turkish while keeping "
                "technical terms like UDS, DTC, checksum, RequestDownload, "
                "TransferData, ECU, and calibration in English when useful.\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": query},
    ]


def complete_chat(chat_client: Any, messages: list[dict[str, str]]) -> str:
    parts = []
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            parts.append(content)
    return "".join(parts).strip()


def retrieved_source_records(results) -> list[dict[str, Any]]:
    source_records = []
    for result in results:
        source_records.append(
            {
                "source": source_label(result.record),
                "rerank_score": result.rerank_score,
                "text": str(result.record.get("text", "")),
            }
        )
    return source_records


def generate_answer(
    case: dict[str, Any],
    records: list[dict[str, Any]],
    bm25_index: Any,
    embedding_client: Any,
    chat_client: Any,
) -> tuple[str, list[dict[str, Any]], float, bool]:
    question = str(case["question"])
    retrieval_query = retrieval_query_for_rerank(question)
    response = embedding_client.generate_embedding(retrieval_query)
    query_embedding = response.data[0].embedding
    results, best_score = retrieve_reranked(
        retrieval_query,
        query_embedding,
        tokenize(retrieval_query),
        records,
        bm25_index=bm25_index,
    )
    should_abstain = best_score < RERANK_GATE or is_out_of_scope_security_query(question)
    sources = format_sources(results)

    if should_abstain:
        answer = abstention_answer(question)
    else:
        direct_answer = direct_grounded_answer(question)
        if direct_answer:
            answer = f"{direct_answer}\nSources: {sources}"
        else:
            context = format_context(results)
            answer = complete_chat(chat_client, grounded_messages(question, context))
            if "Sources:" not in answer:
                answer = f"{answer}\nSources: {sources}".strip()

    return answer, retrieved_source_records(results), best_score, should_abstain


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def matched_abstain_markers(answer: str) -> tuple[bool, list[str], list[str]]:
    lowered = answer.lower()
    base_hits = [marker for marker in BASE_ABSTAIN_MARKERS if marker in lowered]
    extra_hits = [marker for marker in EXTRA_ABSTAIN_MARKERS if marker in lowered]
    return bool(base_hits or extra_hits), base_hits, extra_hits


def has_citation(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker.lower() in lowered for marker in CITATION_MARKERS)


def preview(answer: str, limit: int = 80) -> str:
    compact = " ".join(answer.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def score_case(
    case: dict[str, Any],
    answer: str,
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    snippet_text = "\n".join(str(record["text"]) for record in source_records)
    answer_and_snippets = f"{answer}\n{snippet_text}"
    must_cite = bool(case["must_cite"])
    should_abstain = bool(case["should_abstain"])
    abstain_match, base_abstain_hits, extra_abstain_hits = matched_abstain_markers(answer)

    has_answer = bool(answer.strip())
    citation_present = has_citation(answer)
    citation_ok = citation_present if must_cite else True
    expected_terms_hit = contains_any(
        answer_and_snippets,
        [str(term) for term in case["expected_terms"]],
    )
    abstain_correct = abstain_match if should_abstain else not abstain_match
    passed = has_answer and citation_ok and expected_terms_hit and abstain_correct

    return {
        "has_answer": has_answer,
        "has_citation": citation_present,
        "citation_ok": citation_ok,
        "expected_terms_hit": expected_terms_hit,
        "abstain_correct": abstain_correct,
        "passed": passed,
        "base_abstain_hits": base_abstain_hits,
        "extra_abstain_hits": extra_abstain_hits,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print("ID | passed | citation_ok | expected_terms | abstain_correct | answer_preview")
    for row in rows:
        scores = row["scores"]
        print(
            f"{row['id']} | "
            f"{'yes' if scores['passed'] else 'no'} | "
            f"{'yes' if scores['citation_ok'] else 'no'} | "
            f"{'yes' if scores['expected_terms_hit'] else 'no'} | "
            f"{'yes' if scores['abstain_correct'] else 'no'} | "
            f"{preview(row['answer'])}"
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "passed": sum(1 for row in rows if row["scores"]["passed"]),
        "citation_pass": sum(1 for row in rows if row["scores"]["citation_ok"]),
        "expected_term_pass": sum(
            1 for row in rows if row["scores"]["expected_terms_hit"]
        ),
        "abstain_pass": sum(1 for row in rows if row["scores"]["abstain_correct"]),
        "extra_abstain_variants_used": sorted(
            {
                marker
                for row in rows
                for marker in row["scores"]["extra_abstain_hits"]
            }
        ),
    }


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index: {INDEX_PATH.as_posix()}. Run `python src/ingest.py` first.")

    cases = load_eval_set()
    records = load_index()
    bm25_index = build_bm25(records)

    config = Configuration(app_name=APP_NAME)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = None
    chat_model = None
    rows: list[dict[str, Any]] = []

    try:
        embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)
        if embedding_model is None:
            raise RuntimeError(f"Embedding model not found: {EMBEDDING_MODEL}")
        embedding_model.download(lambda _p: None)
        embedding_model.load()
        embedding_client = embedding_model.get_embedding_client()

        chat_model = manager.catalog.get_model(CHAT_MODEL)
        if chat_model is None:
            raise RuntimeError(f"Chat model not found: {CHAT_MODEL}")
        chat_model.download(lambda _p: None)
        chat_model.load()
        chat_client = chat_model.get_chat_client()

        for case in cases:
            answer, source_records, best_score, gated = generate_answer(
                case,
                records,
                bm25_index,
                embedding_client,
                chat_client,
            )
            scores = score_case(case, answer, source_records)
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "language": case["language"],
                    "answer": answer,
                    "retrieved_sources": source_records,
                    "best_rerank_score": best_score,
                    "gated": gated,
                    "scores": scores,
                }
            )
    finally:
        if embedding_model is not None:
            embedding_model.unload()
        if chat_model is not None:
            chat_model.unload()

    summary = summarize(rows)
    print_table(rows)
    print()
    print("=== Answer quality eval ===")
    print(f"Passed: {summary['passed']}/{summary['total']}")
    print(f"Citation pass: {summary['citation_pass']}/{summary['total']}")
    print(f"Expected-term pass: {summary['expected_term_pass']}/{summary['total']}")
    print(f"Abstain pass: {summary['abstain_pass']}/{summary['total']}")

    report = {"summary": summary, "cases": rows}
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Report written: {REPORT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
