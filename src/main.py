"""
AutoFlash RAG Assistant - Week 1 baseline.

Local RAG demo on Microsoft Foundry Local: embeds a small in-memory knowledge
base, retrieves the most relevant entries for a question via cosine similarity,
and has an on-device chat model answer using only that context.

Adapted from the official Foundry Local "Build a RAG application" tutorial as
the starting point for the ECU/UDS knowledge assistant.

Roadmap TODOs (see inline markers):
  Week 2 - Replace the static DOCUMENTS list with chunks parsed from ECU/UDS
           PDFs in data/raw/ (UDS ISO 14229, OBD-II, Simos18, bri3d/VW_Flash).
           Persist (chunk, embedding, source) rows in SQLite so ingestion runs
           once and retrieval reads from disk.
  Week 3 - Move retrieval to a SQLite-backed store (sqlite-vec); optionally add
           BM25 hybrid search + cross-encoder reranking.
  Week 4 - Attach source file/page to each context line for citations.
"""

import math
from foundry_local_sdk import Configuration, FoundryLocalManager

# --- Configuration ---------------------------------------------------------
APP_NAME = "autoflash_rag"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen2.5-0.5b"  # Week 2: try "phi-4-mini" / "qwen2.5-7b" for better answers
TOP_K = 2

# Week 1 knowledge base. Week 2 TODO: build this from real ECU/UDS document chunks.
DOCUMENTS = [
    "Foundry Local runs AI models directly on your device without cloud connectivity.",
    "The Foundry Local SDK supports Python, C#, JavaScript, and Rust.",
    "Embedding models convert text into numerical vectors for similarity search.",
    "Foundry Local uses ONNX Runtime for efficient model inference on CPUs and GPUs.",
    "The model catalog provides pre-optimized models that you can download and run locally.",
    "Retrieval-augmented generation grounds model responses in your own data.",
    "Vector similarity search finds documents that are semantically close to a query.",
    "Chat completions generate natural language responses from a prompt and context.",
]


# --- Retrieval -------------------------------------------------------------
def cosine_similarity(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, doc_embeddings, top_k=TOP_K):
    """Return (index, score) for the top_k most similar documents.

    Brute-force scan - fine for a small in-memory set. Week 3 TODO: replace
    with a SQLite-backed vector store.
    """
    scores = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(doc_embeddings)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# --- Main ------------------------------------------------------------------
def main():
    # Initialize the SDK
    config = Configuration(app_name=APP_NAME)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Load embedding model and index the knowledge base
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    response = embedding_client.generate_embeddings(DOCUMENTS)
    doc_embeddings = [item.embedding for item in response.data]
    print(f"Indexed {len(doc_embeddings)} documents.")

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

        # Embed the query and retrieve context
        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding
        results = find_relevant(query_embedding, doc_embeddings, top_k=TOP_K)
        context = "\n".join(f"- {DOCUMENTS[i]}" for i, _ in results)

        # Week 4 TODO: append source file/page to each context line for citations.
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "If the context doesn't contain enough information, say so.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ]

        print("Answer: ", end="", flush=True)
        for chunk in chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print("\n")

    # Clean up
    embedding_model.unload()
    chat_model.unload()
    print("Models unloaded. Done!")


if __name__ == "__main__":
    main()
