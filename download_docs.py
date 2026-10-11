"""Download the 8 arXiv papers used as the RAG corpus into ./docs.

Usage: python download_docs.py
The PDFs are NOT committed to the repo (see .gitignore); everyone downloads them with this script.
"""
import time
import urllib.request
from pathlib import Path

PAPERS = {
    "transformer": "1706.03762",  # Attention Is All You Need
    "bert": "1810.04805",         # BERT
    "vit": "2010.11929",          # An Image Is Worth 16x16 Words (ViT)
    "rag": "2005.11401",          # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    "lora": "2106.09685",         # LoRA
    "gpt3": "2005.14165",         # Language Models are Few-Shot Learners
    "resnet": "1512.03385",       # Deep Residual Learning for Image Recognition
    "dpr": "2004.04906",          # Dense Passage Retrieval for Open-Domain QA
}


def main():
    out = Path("docs")
    out.mkdir(exist_ok=True)
    failed = []
    for name, arxiv_id in PAPERS.items():
        dest = out / f"{name}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"skip   {dest.name} (already downloaded)")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (RAG demo project)"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response was not a PDF")
            dest.write_bytes(data)
            print(f"ok     {dest.name}  ({len(data) / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"FAILED {dest.name}: {exc}")
            failed.append((name, url))
        time.sleep(3)  # be polite to arXiv

    if failed:
        print("\nSome downloads failed. Open these links in your browser and save each PDF into docs/ with the name shown:")
        for name, url in failed:
            print(f"  {url}  ->  docs/{name}.pdf")
    else:
        print(f"\nAll {len(PAPERS)} papers are in {out}/. Next: python check_questions.py")


if __name__ == "__main__":
    main()
