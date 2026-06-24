"""AutoFlash RAG Assistant - Week 2 local index query app.

Loads a persisted index from data/index.json, embeds each user query with
Foundry Local, retrieves the most relevant source chunks, and asks an on-device
chat model to answer with citations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager


# --- Configuration ---------------------------------------------------------
APP_NAME = "autoflash_rag"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen2.5-0.5b"
INDEX_PATH = Path("data/index.json")
TOP_K = 3


# --- Index and Retrieval ---------------------------------------------------
def load_index(index_path: Path = INDEX_PATH) -> list[dict[str, object]]:
    """Load the persisted chunk index."""
    with index_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Index must be a JSON list: {index_path}")
    return records


def cosine_similarity(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, records, top_k=TOP_K):
    """Return the top_k most similar chunk records with scores."""
    scored = []
    for record in records:
        embedding = record.get("embedding")
        if not embedding:
            continue
        scored.append((record, cosine_similarity(query_embedding, embedding)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def format_context(results) -> str:
    """Build source-prefixed context blocks for the chat model."""
    blocks = []
    for record, _score in results:
        source = str(record.get("source", ""))
        section = str(record.get("section", ""))
        text = str(record.get("text", "")).strip()
        label = f"[source: {source}"
        if section:
            label += f" § {section}"
        label += "]"
        blocks.append(f"{label}\n{text}")
    return "\n\n---\n\n".join(blocks)


def format_sources(results) -> str:
    """Return a compact, stable source list for retrieved chunks."""
    sources = []
    for record, _score in results:
        source = str(record.get("source", ""))
        section = str(record.get("section", ""))
        label = source
        if section:
            label += f" § {section}"
        if label and label not in sources:
            sources.append(label)
    return "; ".join(sources) if sources else "none"


def is_out_of_scope_security_query(query: str) -> bool:
    """Detect requests for excluded bypass/recovery details."""
    lowered = query.lower()
    sensitive_topic = (
        "seed/key" in lowered
        or ("seed" in lowered and "key" in lowered)
        or any(term in lowered for term in ("rsa", "sboot", "exploit", "bypass"))
    )
    procedural_detail = any(
        term in lowered
        for term in ("algorithm", "recover", "recovery", "compute", "calculate", "crack")
    )
    return sensitive_topic and procedural_detail


# --- Main ------------------------------------------------------------------
def main():
    if not INDEX_PATH.exists():
        print(
            f"Missing index: {INDEX_PATH.as_posix()}. "
            "Run `python src/ingest.py` first."
        )
        return

    records = load_index()
    print(f"Loaded {len(records)} indexed chunks from {INDEX_PATH.as_posix()}.")

    # Initialize the SDK
    config = Configuration(app_name=APP_NAME)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Load embedding model for query embeddings only
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Load chat model
    chat_model = manager.catalog.get_model(CHAT_MODEL)
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nModels loaded. Ready for questions. Type 'quit' to exit.\n")

    # Interactive query loop
    while True:
        query = input("Question: ").strip()
        if not query or query.lower() == "quit":
            break

        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding
        results = find_relevant(query_embedding, records, top_k=TOP_K)
        context = format_context(results)
        sources = format_sources(results)

        if is_out_of_scope_security_query(query):
            print(
                "Answer: I don't know from the provided engineering context. "
                "Security-bypass material, seed/key recovery algorithms, RSA-bypass, "
                "and bootloader-exploit details are intentionally excluded from this corpus."
            )
            print(f"Sources: {sources}\n")
            continue

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "Cite the source(s) you used at the end as `Sources: <source list>`. "
                    "If the context does not contain the answer, say you don't know "
                    "rather than guessing. Do not use outside knowledge.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ]

        print("Answer: ", end="", flush=True)
        answer_parts = []
        for chunk in chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                answer_parts.append(content)
                print(content, end="", flush=True)
        answer_text = "".join(answer_parts)
        if "Sources:" not in answer_text:
            print()
            print(f"Sources: {sources}", end="", flush=True)
        print("\n")

    # Clean up
    embedding_model.unload()
    chat_model.unload()
    print("Models unloaded. Done!")


if __name__ == "__main__":
    main()
