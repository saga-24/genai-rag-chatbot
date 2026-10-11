"""Build the vector index from documents in ./docs (supports .pdf, .txt, .md).

Usage: python ingest.py [--docs docs] [--index-dir index] [--chunk-size 200] [--overlap 40]
"""
import argparse
import json
from pathlib import Path

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SUPPORTED = {".pdf", ".txt", ".md"}


def chunk_words(text, size, overlap):
    """Split text into overlapping chunks of ~`size` words."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_documents(docs_dir):
    """Returns a list of (filename, page_number, text)."""
    pages = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((path.name, i, text))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                pages.append((path.name, 1, text))
    return pages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="docs")
    p.add_argument("--index-dir", default="index")
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=40)
    p.add_argument("--embed-model", default=EMBED_MODEL)
    args = p.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer
    try:
        import faiss
    except ImportError:
        faiss = None  # optional: retrieval falls back to NumPy

    pages = load_documents(args.docs)
    if not pages:
        raise SystemExit(f"No .pdf/.txt/.md files with text found in '{args.docs}/'. Add documents and retry.")

    chunks = []
    for source, page, text in pages:
        for piece in chunk_words(text, args.chunk_size, args.overlap):
            chunks.append({"text": piece, "source": source, "page": page})
    n_files = len({c["source"] for c in chunks})
    print(f"Loaded {n_files} file(s) -> {len(chunks)} chunks")

    embedder = SentenceTransformer(args.embed_model)
    vectors = embedder.encode([c["text"] for c in chunks], batch_size=32,
                              normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype="float32")

    out = Path(args.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)  # always saved; used by the NumPy fallback
    if faiss is not None:
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine for normalized vectors
        index.add(vectors)
        faiss.write_index(index, str(out / "faiss.index"))
        backend = "FAISS"
    else:
        backend = "NumPy (faiss not installed; that's fine)"
    (out / "chunks.json").write_text(json.dumps(
        {"embed_model": args.embed_model, "chunks": chunks}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved index ({len(vectors)} vectors, dim {vectors.shape[1]}, backend: {backend}) to {out}/")


if __name__ == "__main__":
    main()
