from config import (
    BM25_CANDIDATES,
    CHAT_MODEL,
    DENSE_CANDIDATES,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    RERANK_GATE,
    RERANK_MODEL,
    RRF_K,
    TOP_K,
)


def test_default_values_match_documented_invariants():
    assert EMBEDDING_MODEL == "qwen3-embedding-0.6b"
    assert CHAT_MODEL == "phi-4-mini"
    assert TOP_K == 6
    assert DENSE_CANDIDATES == 20
    assert BM25_CANDIDATES == 20
    assert RRF_K == 60
    assert RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert RERANK_CANDIDATES == 20
    assert RERANK_GATE == 0.0
