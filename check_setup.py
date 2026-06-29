"""Week 1 smoke test: verifies the Foundry Local SDK works end-to-end.

Run after `pip install -r requirements.txt` and `winget install Microsoft.FoundryLocal`.

Note: this build runs on CPU (GPU GenAI is unavailable — see src/foundry_setup.py),
so the embedding latency below reflects CPU inference.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from foundry_setup import initialize_manager, load_model_with_webgpu_fallback  # noqa: E402


def main():
    manager = initialize_manager("autoflash_rag")
    print("SDK initialized OK.")

    print("Downloading embedding model if needed...")
    model, status = load_model_with_webgpu_fallback(
        manager,
        "qwen3-embedding-0.6b",
        lambda p: print(f"\r{p:.1f}%", end="", flush=True),
        prefer_webgpu=False,
    )
    print()
    print(f"Model loaded: {status.model_id} ({status.device}/{status.execution_provider})")
    client = model.get_embedding_client()

    t0 = time.time()
    resp = client.generate_embedding("hello world")
    dt = (time.time() - t0) * 1000
    dim = len(resp.data[0].embedding)
    print(f"Embedding OK. dim={dim}, latency={dt:.0f} ms")

    model.unload()
    print("Done.")


if __name__ == "__main__":
    main()
