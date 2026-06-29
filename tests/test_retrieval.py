from retrieval import reciprocal_rank_fusion, tokenize


def test_tokenize_preserves_hex_service_ids_and_lowercases():
    tokens = tokenize("Service 0x19 and 0x22 read DTC info")
    assert "0x19" in tokens
    assert "0x22" in tokens
    assert "dtc" in tokens


def test_reciprocal_rank_fusion_orders_by_combined_rank():
    records = [{"text": f"doc{i}"} for i in range(4)]
    dense_results = [(0, 0.9), (1, 0.8), (2, 0.5)]
    bm25_results = [(0, 5.0), (2, 4.0), (3, 1.0)]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, records, top_k=4, rrf_k=60)

    ranking = [r.record_index for r in fused]
    assert ranking[0] == 0
    assert ranking.index(2) < ranking.index(1)
    assert ranking.index(2) < ranking.index(3)
