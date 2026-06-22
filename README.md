# AutoFlash RAG Assistant
**Local ECU/UDS Knowledge Assistant with Microsoft Foundry Local**

A fully offline RAG assistant that answers engineering questions about automotive
diagnostics and ECU flashing (UDS / ISO 14229, OBD-II, Simos18, checksum/container
logic) by retrieving from a local document collection and generating grounded,
source-cited answers — all on-device via Microsoft Foundry Local.

> Scope note: engineering/educational focus (protocol specs, diagnostic standards,
> open-source tooling). Not a guide for emissions-defeat or illegal ECU modification.

## Hardware note
Built and tested on an RTX 5060 Laptop (Blackwell). Uses the `foundry-local-sdk-winml`
package (Windows ML / WebGPU acceleration) instead of the CUDA execution provider,
which does not ship Blackwell (sm_120) kernels at time of writing.

## Setup
```bash
winget install Microsoft.FoundryLocal
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py        # smoke test
python src/main.py           # RAG demo
```

## Roadmap
- [x] Week 1 — Foundry Local setup + official RAG tutorial
- [ ] Week 2 — ECU PDF ingest + chunking
- [ ] Week 3 — retrieval + grounded answer generation
- [ ] Week 4 — source citation, tests, README, 5-min presentation

## Status
Week 1 — scaffold.
