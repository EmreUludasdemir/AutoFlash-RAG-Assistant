"""Centralized configuration for the AutoFlash RAG Assistant.

Every value here defaults to the constant that was previously hardcoded in
retrieval.py/main.py/ingest.py. Defaults are unchanged so retrieval/rerank/
gate behavior and the eval baselines do not move; set the corresponding env
var to override a value for local experimentation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value is not None else default


# --- Foundry Local models --------------------------------------------------
APP_NAME = _env_str("AUTOFLASH_APP_NAME", "autoflash_rag")
EMBEDDING_MODEL = _env_str("AUTOFLASH_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
CHAT_MODEL = _env_str("AUTOFLASH_CHAT_MODEL", "phi-4-mini")

# --- Index ------------------------------------------------------------------
INDEX_PATH = _env_path("AUTOFLASH_INDEX_PATH", Path("data/index.json"))

# --- Retrieval (dense + BM25 + RRF) -----------------------------------------
TOP_K = _env_int("AUTOFLASH_TOP_K", 6)
DENSE_CANDIDATES = _env_int("AUTOFLASH_DENSE_CANDIDATES", 20)
BM25_CANDIDATES = _env_int("AUTOFLASH_BM25_CANDIDATES", 20)
RRF_K = _env_int("AUTOFLASH_RRF_K", 60)

# --- Cross-encoder rerank + confidence gate ---------------------------------
RERANK_MODEL = _env_str("AUTOFLASH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = _env_int("AUTOFLASH_RERANK_CANDIDATES", 20)
RERANK_GATE = _env_float("AUTOFLASH_RERANK_GATE", 0.0)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up readable console logging for entry-point scripts. Idempotent."""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
