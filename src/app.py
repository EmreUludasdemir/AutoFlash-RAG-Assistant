"""Streamlit UI for the local AutoFlash RAG Assistant."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from foundry_setup import initialize_manager, load_model_with_webgpu_fallback  # noqa: E402
from main import (  # noqa: E402
    APP_NAME,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    abstention_answer,
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
    get_reranker,
    load_index,
    retrieve_reranked,
    source_label,
    tokenize,
)


def language_instruction(query: str) -> str:
    if is_turkish_query(query):
        return "The user's question is in Turkish. Answer in Turkish."
    return "The user's question is in English. Answer in English."


HISTORY_TURNS = 3


def recent_history(messages: list[dict[str, str]], turns: int = HISTORY_TURNS) -> list[dict[str, str]]:
    """Return the last `turns` user/assistant exchanges, oldest first."""
    return messages[-(turns * 2):] if messages else []


def grounded_messages(
    query: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            f"{language_instruction(query)} Give a thorough, well-structured, "
            "technically precise answer grounded only in the provided context. "
            "Explain the relevant service flow, roles, constraints, and "
            "terminology when the context supports it. Cite the source file(s) "
            "you used at the end as `Sources: <source list>`. If the context "
            "does not contain the answer, say that the provided context does "
            "not contain the information rather than using outside knowledge. "
            "When the question names an exact identifier or service such as "
            "0x19 or RequestDownload, focus only on context about that exact "
            "identifier or service and ignore unrelated identifiers. Do not "
            "invent protocols, transports, or examples that are not in context. "
            "If the user asks in Turkish, answer in Turkish while keeping "
            "technical terms like UDS, DTC, checksum, RequestDownload, "
            "TransferData, ECU, and calibration in English when useful. Use the "
            "prior conversation turns only for continuity (e.g. follow-up "
            "references); ground the actual answer in the context below.\n\n"
            f"Context:\n{context}"
        ),
    }
    return [system, *(history or []), {"role": "user", "content": query}]


def stream_chat(chat_client: Any, messages: list[dict[str, str]]):
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


def history_transcript(history: list[dict[str, str]]) -> str:
    lines = []
    for message in history:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def rewrite_query_with_history(chat_client: Any, query: str, history: list[dict[str, str]]) -> str:
    """Resolve an elliptical follow-up into a standalone search query.

    Retrieval has no memory of prior turns, so "What does it return exactly?"
    finds nothing on its own. Ask the chat model to fold in the conversation
    history before we embed/search; on any failure or empty output, fall back
    to the raw query so retrieval still runs.
    """
    if not history:
        return query

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's latest message into a short, standalone "
                "search query for a document retrieval system. Resolve any "
                "pronouns or references (e.g. 'it', 'that service') using the "
                "conversation history below. Output ONLY the rewritten query "
                "text on a single line: no quotes, no explanation, no prefix. "
                "Prefer English technical terms even if the conversation is in "
                "Turkish.\n\n"
                f"Conversation history:\n{history_transcript(history)}"
            ),
        },
        {"role": "user", "content": query},
    ]
    try:
        parts = []
        for chunk in chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                parts.append(content)
        raw = "".join(parts).strip()
        rewritten = raw.splitlines()[0].strip().strip('"').strip() if raw else ""
        return rewritten or query
    except Exception:
        return query


@st.cache_resource(show_spinner="Loading local RAG resources...")
def load_resources() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing index: {INDEX_PATH.as_posix()}. Run `python src/ingest.py` first."
        )

    records = load_index()
    bm25_index = build_bm25(records)

    manager = initialize_manager(APP_NAME)

    embedding_model, embedding_status = load_model_with_webgpu_fallback(
        manager,
        EMBEDDING_MODEL,
        lambda _p: None,
    )
    embedding_client = embedding_model.get_embedding_client()

    chat_model, chat_status = load_model_with_webgpu_fallback(
        manager,
        CHAT_MODEL,
        lambda _p: None,
    )
    chat_client = chat_model.get_chat_client()

    reranker = get_reranker()
    print(
        "Streamlit resource init complete: "
        f"{len(records)} chunks, BM25, Foundry models, CPU CrossEncoder loaded."
    )

    return {
        "records": records,
        "bm25_index": bm25_index,
        "embedding_client": embedding_client,
        "chat_client": chat_client,
        "embedding_status": embedding_status,
        "chat_status": chat_status,
        "reranker": reranker,
    }


def retrieve_for_query(query: str, resources: dict[str, Any]):
    retrieval_query = retrieval_query_for_rerank(query)
    response = resources["embedding_client"].generate_embedding(retrieval_query)
    query_embedding = response.data[0].embedding
    return retrieve_reranked(
        retrieval_query,
        query_embedding,
        tokenize(retrieval_query),
        resources["records"],
        bm25_index=resources["bm25_index"],
    )


def show_sources(
    results,
    best_score: float,
    decision: str,
    search_query: str | None = None,
    raw_query: str | None = None,
) -> None:
    with st.expander("Retrieved sources", expanded=False):
        if search_query and raw_query and search_query != raw_query:
            st.caption(f"Rewritten search query: {search_query!r}")
        st.caption(
            f"Confidence: best_rerank_score={best_score:.3f} "
            f"(gate={RERANK_GATE}) -> {decision}"
        )
        if not results:
            st.write("No sources shown because no confident in-scope match was found.")
            return
        for result in results:
            score = result.rerank_score if result.rerank_score is not None else 0.0
            st.markdown(f"- `{score:.3f}` {source_label(result.record)}")


def main() -> None:
    st.set_page_config(
        page_title="AutoFlash RAG Assistant",
        layout="wide",
    )
    st.title("AutoFlash RAG Assistant — Local ECU/UDS Knowledge")
    st.caption(
        "Runs fully offline on Microsoft Foundry Local. "
        "Engineering/educational sources only; security-bypass material is excluded."
    )

    try:
        resources = load_resources()
    except Exception as exc:  # pragma: no cover - visible UI startup failure
        st.error(str(exc))
        st.stop()

    st.sidebar.metric("Indexed chunks", len(resources["records"]))
    st.sidebar.caption(f"Confidence gate: {RERANK_GATE}")
    st.sidebar.caption(
        f"Chat: {resources['chat_status'].model_id} "
        f"({resources['chat_status'].execution_provider})"
    )
    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about ECU diagnostics, UDS services, DTCs, or flashing concepts")
    if not query:
        return

    history = recent_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving local context..."):
            search_query = rewrite_query_with_history(resources["chat_client"], query, history)
            results, best_score = retrieve_for_query(search_query, resources)

        should_abstain = best_score < RERANK_GATE or is_out_of_scope_security_query(query)
        decision = "abstained" if should_abstain else "answered"

        if should_abstain:
            answer = abstention_answer(query)
            st.markdown(answer)
            show_sources([], best_score, decision, search_query, query)
        else:
            context = format_context(results)
            sources = format_sources(results)
            answer = st.write_stream(
                stream_chat(
                    resources["chat_client"],
                    grounded_messages(query, context, history),
                )
            )
            if "Sources:" not in answer:
                source_line = f"\n\nSources: {sources}"
                st.markdown(source_line)
                answer += source_line
            show_sources(results, best_score, decision, search_query, query)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
