"""Build the Week 2 local RAG index from files in data/raw/.

The index is a JSON list of chunks with source metadata and Foundry Local
embeddings. Raw documents stay untracked; the generated data/index.json is also
ignored and can be rebuilt.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import pymupdf4llm
from foundry_setup import initialize_manager, load_model_with_webgpu_fallback

from config import APP_NAME, EMBEDDING_MODEL, INDEX_PATH, configure_logging

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".rst"}
BATCH_SIZE = 32
CHUNK_TOKEN_TARGET = 450
CHUNK_OVERLAP_RATIO = 0.15


def discover_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return supported raw files, recursively, with deterministic ordering."""
    if not raw_dir.exists():
        return []
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_document(path: Path) -> str:
    """Read a supported document as Markdown-ish text."""
    if path.suffix.lower() == ".pdf":
        return pymupdf4llm.to_markdown(str(path))
    return path.read_text(encoding="utf-8", errors="ignore")


def is_rst_underline(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and len(set(stripped)) == 1 and stripped[0] in "=-~^\"'"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown/RST text into (nearest heading, section text)."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        markdown_match = re.match(r"^\s{0,3}(#{1,3})\s+(.+?)\s*$", line)
        rst_heading = (
            i + 1 < len(lines)
            and line.strip()
            and is_rst_underline(lines[i + 1])
            and len(lines[i + 1].strip()) >= min(len(line.strip()), 3)
        )

        if markdown_match or rst_heading:
            if current_lines:
                sections.append((current_heading, current_lines))
                current_lines = []

            if markdown_match:
                current_heading = markdown_match.group(2).strip()
                current_lines.append(line)
                i += 1
                continue

            current_heading = line.strip()
            current_lines.extend([line, lines[i + 1]])
            i += 2
            continue

        current_lines.append(line)
        i += 1

    if current_lines:
        sections.append((current_heading, current_lines))

    return [(heading, "\n".join(section_lines).strip()) for heading, section_lines in sections]


def estimate_tokens(text: str) -> int:
    return int(len(re.findall(r"\S+", text)) * 1.3)


def pack_section(section_text: str) -> list[str]:
    """Pack a section into roughly CHUNK_TOKEN_TARGET chunks with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if current and current_tokens + paragraph_tokens > CHUNK_TOKEN_TARGET:
            chunks.append("\n\n".join(current).strip())
            overlap_target = max(1, int(CHUNK_TOKEN_TARGET * CHUNK_OVERLAP_RATIO))
            overlap: list[str] = []
            overlap_tokens = 0
            for item in reversed(current):
                item_tokens = estimate_tokens(item)
                if overlap and overlap_tokens + item_tokens > overlap_target:
                    break
                overlap.insert(0, item)
                overlap_tokens += item_tokens
            current = overlap
            current_tokens = overlap_tokens

        if paragraph_tokens > CHUNK_TOKEN_TARGET:
            words = paragraph.split()
            max_words = max(1, int(CHUNK_TOKEN_TARGET / 1.3))
            overlap_words = max(1, int(max_words * CHUNK_OVERLAP_RATIO))
            start = 0
            while start < len(words):
                part = " ".join(words[start : start + max_words]).strip()
                if current and current_tokens + estimate_tokens(part) > CHUNK_TOKEN_TARGET:
                    chunks.append("\n\n".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.append(part)
                if start + max_words >= len(words):
                    break
                start += max_words - overlap_words
            continue

        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk.strip()]


def make_records(files: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    repo_root = Path.cwd().resolve()
    for path in files:
        relative_path = path.resolve().relative_to(repo_root).as_posix()
        text = read_document(path)
        chunk_index = 0
        for section, section_text in split_sections(text):
            for chunk in pack_section(section_text):
                records.append(
                    {
                        "id": f"{relative_path}#{chunk_index}",
                        "source": relative_path,
                        "section": section,
                        "text": chunk,
                    }
                )
                chunk_index += 1
    return records


def embed_records(records: list[dict[str, object]]) -> float | None:
    if not records:
        return None

    # Load on CPU (prefer_webgpu=False): GPU GenAI is unavailable in this SDK
    # build, and once GPU EPs are registered the catalog default would otherwise
    # be an unloadable GPU variant. Index embeddings must stay on the same CPU
    # variant the query path uses so cosine similarities remain comparable.
    manager = initialize_manager(APP_NAME)
    model, _status = load_model_with_webgpu_fallback(
        manager,
        EMBEDDING_MODEL,
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True),
        prefer_webgpu=False,
    )
    print()
    client = model.get_embedding_client()

    first_batch_latency_ms: float | None = None
    try:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            texts = [str(record["text"]) for record in batch]
            t0 = time.time()
            response = client.generate_embeddings(texts)
            elapsed_ms = (time.time() - t0) * 1000
            if first_batch_latency_ms is None:
                first_batch_latency_ms = elapsed_ms
            for record, item in zip(batch, response.data):
                record["embedding"] = item.embedding
            logger.info(
                "Embedded %d/%d chunks.",
                min(start + len(batch), len(records)),
                len(records),
            )
    finally:
        model.unload()

    return first_batch_latency_ms


def write_index(records: list[dict[str, object]], index_path: Path = INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    configure_logging()
    files = discover_files()
    records = make_records(files)
    first_batch_latency_ms = embed_records(records)
    write_index(records)

    logger.info("Files processed: %d", len(files))
    logger.info("Chunks indexed: %d", len(records))
    logger.info("Output path: %s", INDEX_PATH.as_posix())
    if first_batch_latency_ms is None:
        logger.info("First batch embedding latency: n/a")
    else:
        logger.info("First batch embedding latency: %.0f ms", first_batch_latency_ms)


if __name__ == "__main__":
    main()
