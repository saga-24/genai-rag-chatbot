"""RAG pipeline: retrieve relevant chunks with FAISS, then answer with a local open-source LLM.

CLI test: python rag.py --question "What is the CLS token?"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import faiss  # optional
except ImportError:
    faiss = None

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NO_ANSWER = "I don't know based on the provided documents."
SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the numbered context passages provided. "
    "Be concise. Cite the passages you used like [1] or [2]. "
    f"If the answer is not contained in the context, reply exactly: {NO_ANSWER}"
)


def default_llm():
    # 1.5B model for GPU; 0.5B so that CPU-only machines still work.
    return "Qwen/Qwen2.5-1.5B-Instruct" if torch.cuda.is_available() else "Qwen/Qwen2.5-0.5B-Instruct"


class RAGPipeline:
    def __init__(self, index_dir="index", llm_name=None, top_k=4, min_score=None):
        """Initialise the RAG pipeline.

        Args:
            index_dir: Directory containing chunks.json and vectors.npy (built by ingest.py).
            llm_name: HuggingFace model ID for the LLM. Defaults to Qwen2.5 based on device.
            top_k: Number of chunks to retrieve per query.
            min_score: Optional minimum cosine-similarity score (0-1). Chunks whose score is
                strictly below this threshold are discarded. When *all* retrieved chunks fall
                below the threshold the pipeline returns NO_ANSWER immediately without calling
                the LLM, which saves GPU/CPU time and improves abstention on off-topic questions.
                Set to None (the default) to disable the guard and preserve original behaviour.
        """
        index_dir = Path(index_dir)
        if not (index_dir / "chunks.json").exists() or not (index_dir / "vectors.npy").exists():
            raise SystemExit(f"No index found in '{index_dir}/'. Run: python ingest.py")
        self.top_k = top_k
        self.min_score = min_score
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        meta = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = meta["chunks"]
        self.vectors = np.load(index_dir / "vectors.npy").astype("float32")
        faiss_path = index_dir / "faiss.index"
        self.index = faiss.read_index(str(faiss_path)) if (faiss is not None and faiss_path.exists()) else None
        print("Retrieval backend:", "FAISS" if self.index is not None else "NumPy")
        self.embedder = SentenceTransformer(meta["embed_model"], device=str(self.device))

        self.llm_name = llm_name or default_llm()
        print(f"Loading LLM: {self.llm_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_name, torch_dtype=dtype)
        self.model.to(self.device).eval()

    def retrieve(self, question, k=None):
        """Embed *question* and return the top-k most similar chunks.

        Each returned item is a dict with keys: text, source, page, score.
        If ``self.min_score`` is set, chunks whose similarity score is strictly
        below the threshold are removed from the results before returning.
        """
        k = min(k or self.top_k, len(self.chunks))
        q = self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            scores, ids = scores[0], ids[0]
        else:  # cosine similarity (vectors are normalized) via NumPy
            sims = (self.vectors @ q[0])
            ids = np.argsort(-sims)[:k]
            scores = sims[ids]
        candidates = [{**self.chunks[int(i)], "score": float(s)} for s, i in zip(scores, ids) if i >= 0]
        if self.min_score is not None:
            candidates = [c for c in candidates if c["score"] >= self.min_score]
        return candidates

    def build_messages(self, question, contexts):
        context_block = "\n\n".join(
            f"[{n}] (source: {c['source']}, page {c['page']})\n{c['text']}" for n, c in enumerate(contexts, 1))
        user = f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=256):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def ask(self, question, k=None):
        """Answer *question* using retrieved context.

        Returns a dict with keys:
            answer  - the generated (or fallback) answer string.
            sources - list of retrieved chunk dicts (may be empty when the threshold
                      filtered out all candidates).
        """
        contexts = self.retrieve(question, k)
        if not contexts:
            # All chunks were below min_score — abstain without calling the LLM.
            return {"answer": NO_ANSWER, "sources": []}
        answer = self.generate(self.build_messages(question, contexts))
        return {"answer": answer, "sources": contexts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument(
        "--min-score", type=float, default=None,
        help="Minimum cosine-similarity score (0-1). Chunks below this threshold are "
             "discarded; if no chunk meets the threshold the pipeline abstains without "
             "calling the LLM. Typical relevant scores are ~0.80; off-topic scores are "
             "~0.49-0.55. A value around 0.65 is a reasonable starting point.")
    a = ap.parse_args()
    rag = RAGPipeline(a.index_dir, a.llm, a.top_k, min_score=a.min_score)
    res = rag.ask(a.question)
    print("\nANSWER:", res["answer"])
    print("\nSOURCES:")
    for n, c in enumerate(res["sources"], 1):
        print(f"[{n}] {c['source']} (page {c['page']}, score {c['score']:.3f}): {c['text'][:150]}...")
