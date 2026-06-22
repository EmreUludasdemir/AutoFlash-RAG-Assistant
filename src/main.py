"""AutoFlash RAG Assistant — Week 1: official Foundry Local RAG tutorial skeleton.

Paste the official tutorial code body here:
https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app

The tutorial uses:
  - from foundry_local_sdk import Configuration, FoundryLocalManager
  - embedding model: qwen3-embedding-0.6b  (batch: generate_embeddings(documents))
  - chat model:      qwen2.5-0.5b          (single query: generate_embedding(query))
  - cosine_similarity + find_relevant(top_k=2)
  - chat_client.complete_streaming_chat(messages) for streaming answers

Week 2 TODO: replace the static `documents` list with chunks parsed from
ECU/UDS PDFs in data/raw/ (UDS ISO 14229, Simos18, bri3d/VW_Flash docs).
"""


def main():
    # TODO: paste the tutorial's main() body here, then run `python src/main.py`
    raise NotImplementedError("Copy the official tutorial code from the URL above.")


if __name__ == "__main__":
    main()
