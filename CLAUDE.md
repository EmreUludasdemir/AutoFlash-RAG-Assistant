# CLAUDE.md — AutoFlash RAG Assistant

Guidance for AI coding sessions in this repo. Read this first; it captures
context and invariants that are expensive to re-derive.

## What this project is

A **fully offline** RAG assistant over local ECU/UDS engineering documents
(UDS / ISO 14229, OBD-II, checksum/flashing concepts, the `python-udsoncan`
docs). Everything runs on-device via **Microsoft Foundry Local**.

**Scope is engineering/educational only.** Security-bypass material — seed/key
recovery algorithms, RSA-bypass, SBOOT/bootloader exploits, ECU unlock/patch
workflows — is intentionally **excluded** from the corpus (see
[data/SOURCES.md](data/SOURCES.md), which lists fetched-then-removed files).
Out-of-scope queries **must abstain**, never fabricate.

## Stack

- Python **3.12** in a venv at `.venv`. Always invoke as
  `.venv\Scripts\python.exe`.
- `foundry-local-sdk-winml` **1.2.3** (Windows ML / WebGPU path, chosen because
  the CUDA EP does not ship Blackwell `sm_120` kernels yet). Foundry CLI ~0.8.x.
- Hardware: RTX 5060 Laptop (Blackwell, 8 GB VRAM) + AMD NPU + CPU.
- Models are loaded **by alias** via `manager.catalog.get_model("<alias>")`,
  then `.download()` -> `.load()` -> `.get_embedding_client()` /
  `.get_chat_client()`.
  - Embedding: `qwen3-embedding-0.6b` (**keep on alias; do not change**).
  - Chat: alias `phi-4-mini`.
- Important SDK behavior: `foundry-local-sdk-winml` runs its own embedded core,
  separate from the `foundry` CLI service. The SDK catalog exposes model
  variants only for execution providers registered in the current Python
  process. With no registered GPU EPs it silently showed/loaded only
  `generic-cpu` variants, so the historical ~53 s answers were CPU inference,
  not NPU inference.
- `get_model_variant(...)` returning `None` for GPU ids was caused by the SDK
  catalog not exposing GPU variants before EP registration, not by a colon-id
  format issue.
- The app now calls [src/foundry_setup.py](src/foundry_setup.py) on startup:
  it registers EPs (the single-name WebGPU registration path currently returns
  an empty-name error, so it uses the all-EP workaround), prefers
  `WebGpuExecutionProvider` variants, and falls back to CPU if GPU load fails.
  On this machine WebGPU registration succeeds and `generic-gpu` variants are
  visible/cached, but loading `Phi-4-mini-instruct-generic-gpu:5` currently
  fails in the SDK runtime (`WebGPU execution provider is not supported in this
  build` / CUDA GenAI DLL path when CUDA is also registered), so CPU fallback is
  the stable working path.

## Architecture (read the code at these paths)

- [src/ingest.py](src/ingest.py) — read `data/raw/` (`.pdf` via `pymupdf4llm`,
  `.md`, `.rst`), split by MD/RST headings, pack ~450-token chunks (15%
  overlap), embed, write `data/index.json`.
- [src/retrieval.py](src/retrieval.py) — dense cosine + BM25 → **RRF**
  (`RRF_K=60`) → top-20 candidates → **cross-encoder rerank**
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU) → `TOP_K=6`, plus the
  **confidence gate** `RERANK_GATE=0.0`.
- [src/main.py](src/main.py) — CLI query loop: English retrieval-query rewrite
  for known Turkish topics, gate → abstain/answer, deterministic direct answers
  for smoke topics, out-of-scope security guard, citations, streaming.
- [src/app.py](src/app.py) — Streamlit UI reusing `main`/`retrieval`; model load
  via `@st.cache_resource`; sources expander shows rerank scores.

## Invariants — do NOT regress

- Retrieval: dense + BM25 + **RRF**, then **cross-encoder rerank**.
- **Confidence gate** `RERANK_GATE=0.0`. In-scope best rerank scores ~3.6–6.75;
  out-of-scope (e.g. seed/key) ~-2.99, so it falls below the gate and abstains.
- `TOP_K=6`. Source citations on every grounded answer.
- **Turkish-in → Turkish-out** answers (keep technical terms like UDS, DTC,
  checksum, RequestDownload, TransferData, ECU, calibration in English).
- **Empty-`choices` streaming guard**: `if not chunk.choices: continue` before
  reading `chunk.choices[0].delta.content` (Foundry can emit empty chunks).
- **Clean model unload** (`.unload()`) for every model that was loaded.
- Keep the **embedding model on its alias**.

## Eval commands (the regression gate)

```
.venv\Scripts\python.exe eval\retrieval_smoke.py
.venv\Scripts\python.exe eval\answer_quality_eval.py
```

Expected baselines:
- `retrieval_smoke.py`: MRR **0.69 / 0.90 / 1.00** for dense / hybrid /
  hybrid+rerank; out-of-scope seed/key below gate.
- `answer_quality_eval.py`: **7/7** passed; the seed/key cases abstain without
  fabricating.

After ANY change, re-run BOTH and confirm: MRR unchanged, answer-quality still
7/7, seed/key still abstains. Requires `data/index.json` (rebuild with
`.venv\Scripts\python.exe src\ingest.py` if missing).

## Never commit (already in .gitignore)

`.venv/`, `__pycache__/`, `*.pyc`, `data/index.json`, `data/raw/*` (except
`.gitkeep`), `eval/answer_quality_report.json`, `.env`, `desktop.ini`, caches
(`.mypy_cache/`).

## Workflow expectations

- Show a diff and a one-line rationale before committing; commit per logical
  change with a clear message.
- Ask before starting each roadmap item (multi-turn chat, Streamlit UX, corpus
  expansion, config/tests/logging, RAGAS).
