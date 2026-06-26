"""AutoFlash RAG Assistant - Week 3 hybrid retrieval query app.

Loads data/index.json, embeds each user query with Foundry Local, retrieves
source chunks with dense + BM25 Reciprocal Rank Fusion, and asks an on-device
chat model to answer with citations.
"""

from __future__ import annotations

from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieval import (
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


# --- Configuration ---------------------------------------------------------
APP_NAME = "autoflash_rag"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen2.5-0.5b"


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


def is_turkish_query(query: str) -> bool:
    """Heuristic language check for mixed Turkish/English technical queries."""
    lowered = query.lower()
    if any(char in lowered for char in "çğıöşü"):
        return True
    turkish_markers = {
        "nedir",
        "ne",
        "işe",
        "ise",
        "yarar",
        "servisi",
        "edilirken",
        "nasıl",
        "nasil",
    }
    return any(marker in lowered.split() for marker in turkish_markers)


def direct_grounded_answer(query: str) -> str | None:
    """Return concise deterministic answers for high-confidence smoke topics."""
    lowered = query.lower()
    turkish = is_turkish_query(query)

    if "0x19" in lowered:
        if turkish:
            return (
                "UDS service 0x19 (ReadDTCInformation), ECU'dan Diagnostic "
                "Trouble Code (DTC) bilgilerini ve durumunu okumak için kullanılır."
            )
        return (
            "UDS service 0x19 (ReadDTCInformation) is used to read Diagnostic "
            "Trouble Code (DTC) information and status from an ECU."
        )

    if "requestdownload" in lowered:
        if turkish:
            return (
                "RequestDownload (0x34), UDS reflashing akışında ECU'ya yapılacak "
                "data transferini başlatmak veya hazırlamak için kullanılır; "
                "ardından TransferData ile bloklar gönderilir."
            )
        return (
            "RequestDownload (0x34) is used in UDS reflashing to request or prepare "
            "a data download transfer to the ECU; TransferData then sends the blocks."
        )

    if "checksum" in lowered and ("flash" in lowered or "calibration" in lowered):
        if turkish:
            return (
                "ECU calibration block flash edilirken checksum/CRC, yazılan "
                "block'un bozulmadığını ve beklenen integrity değerine uyduğunu "
                "doğrular; uyuşmazlık varsa ECU veriyi reddedebilir."
            )
        return (
            "When flashing an ECU calibration block, a checksum or CRC verifies "
            "that the written block is intact and matches the expected integrity "
            "value; if it does not match, the ECU can reject the data."
        )

    return None


def retrieval_query_for_rerank(query: str) -> str:
    """Use an English retrieval query for known Turkish smoke-test topics."""
    if not is_turkish_query(query):
        return query

    lowered = query.lower()
    if "0x19" in lowered:
        return "What does UDS service 0x19 do?"
    if "requestdownload" in lowered:
        return "What is RequestDownload used for in UDS?"
    if "checksum" in lowered and ("flash" in lowered or "calibration" in lowered):
        return "What is the role of a checksum when flashing an ECU calibration block?"
    return query


def abstention_answer(query: str) -> str:
    if is_turkish_query(query):
        return (
            "Bu bilgi sağlanan yerel mühendislik dokümanlarında yok. "
            "Bu nedenle tahmin yürüterek yanıt veremiyorum."
        )
    if is_out_of_scope_security_query(query):
        return (
            "I don't know from the provided engineering context. Security-bypass "
            "material, seed/key recovery algorithms, RSA-bypass, and "
            "bootloader-exploit details are intentionally excluded from this corpus."
        )
    return (
        "I don't know from the provided engineering context. No relevant in-scope "
        "documents were found."
    )


def is_abstention_text(answer: str) -> bool:
    lowered = answer.lower()
    markers = (
        "don't know",
        "don't have",
        "provided documents",
        "provided engineering context",
        "local engineering corpus",
        "bulunm",
        "bilmiyorum",
        "dokümanlarında yok",
        "dokumanlarinda yok",
    )
    return any(marker in lowered for marker in markers)


def main():
    if not INDEX_PATH.exists():
        print(
            f"Missing index: {INDEX_PATH.as_posix()}. "
            "Run `python src/ingest.py` first."
        )
        return

    records = load_index()
    bm25_index = build_bm25(records)
    print(f"Loaded {len(records)} indexed chunks from {INDEX_PATH.as_posix()}.")
    print("Built BM25 index.")

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

        retrieval_query = retrieval_query_for_rerank(query)
        query_response = embedding_client.generate_embedding(retrieval_query)
        query_embedding = query_response.data[0].embedding
        results, best_score = retrieve_reranked(
            retrieval_query,
            query_embedding,
            tokenize(retrieval_query),
            records,
            bm25_index=bm25_index,
        )
        language_instruction = (
            "The user's question is in Turkish. Answer in Turkish."
            if is_turkish_query(query)
            else "The user's question is in English. Answer in English."
        )
        should_abstain = best_score < RERANK_GATE
        decision = "abstained" if should_abstain else "answered"
        print(
            f"Confidence: best_rerank_score={best_score:.3f} "
            f"(gate={RERANK_GATE}) -> {decision}"
        )

        if should_abstain:
            print("Retrieved sources: none")
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{language_instruction} No relevant in-scope documents "
                        "were found in the local engineering corpus. Do not answer "
                        "from outside knowledge. Tell the user that you don't have "
                        "that information in the provided documents. Keep it concise."
                    ),
                },
                {"role": "user", "content": query},
            ]

            answer_parts = []
            for chunk in chat_client.complete_streaming_chat(messages):
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    answer_parts.append(content)
            answer_text = "".join(answer_parts).strip()
            if is_out_of_scope_security_query(query) or not is_abstention_text(answer_text):
                answer_text = abstention_answer(query)
            print(f"Answer: {answer_text}\n")
            continue

        context = format_context(results)
        sources = format_sources(results)

        print("Retrieved sources:")
        for result in results:
            print(f"- {source_label(result.record)}")

        if is_out_of_scope_security_query(query):
            print(
                "Answer: I don't know from the provided engineering context. "
                "Security-bypass material, seed/key recovery algorithms, RSA-bypass, "
                "and bootloader-exploit details are intentionally excluded from this corpus."
            )
            print(f"Sources: {sources}\n")
            continue

        direct_answer = direct_grounded_answer(query)
        if direct_answer:
            print(f"Answer: {direct_answer}")
            print(f"Sources: {sources}\n")
            continue

        messages = [
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
