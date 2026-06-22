"""Week 1 smoke test: verifies the Foundry Local SDK works end-to-end and
gives a rough GPU-vs-CPU signal via embedding latency.

Run after `pip install -r requirements.txt` and `winget install Microsoft.FoundryLocal`.
"""
import time
from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    config = Configuration(app_name="autoflash_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    print("SDK initialized OK.")

    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    print("Downloading embedding model if needed...")
    model.download(lambda p: print(f"\r{p:.1f}%", end="", flush=True))
    print()
    model.load()
    client = model.get_embedding_client()

    t0 = time.time()
    resp = client.generate_embedding("hello world")
    dt = (time.time() - t0) * 1000
    dim = len(resp.data[0].embedding)
    print(f"Embedding OK. dim={dim}, latency={dt:.0f} ms")
    print("Note: <~150 ms usually means GPU is engaged; much slower likely CPU.")
    print("Authoritative check: run `foundry model list` and look at variants.")

    model.unload()
    print("Done.")


if __name__ == "__main__":
    main()
