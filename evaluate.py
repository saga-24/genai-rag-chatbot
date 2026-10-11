"""Evaluate the RAG pipeline on eval/questions.json.

Metrics
  retrieval_hit_rate : answerable questions where a top-k chunk comes from the expected file AND contains a gold keyword
  answer_accuracy    : answerable questions whose answer contains at least one gold keyword
  abstention_rate    : unanswerable questions where the bot correctly says it doesn't know
Usage: python evaluate.py [--questions eval/questions.json]
"""
import argparse
import json
import time
from pathlib import Path


def norm(s):
    return s.lower().replace("\u2019", "'")


def contains_any(text, keywords):
    t = norm(text)
    return any(norm(k) in t for k in keywords)


def abstained(answer):
    a = norm(answer)
    return "don't know" in a or "do not know" in a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="eval/questions.json")
    p.add_argument("--index-dir", default="index")
    p.add_argument("--llm", default=None)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    from rag import RAGPipeline
    rag = RAGPipeline(args.index_dir, args.llm, args.top_k)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    details, hits, correct, abst, n_ans, n_unans, times = [], 0, 0, 0, 0, 0, []
    for item in questions:
        t0 = time.time()
        res = rag.ask(item["question"])
        times.append(time.time() - t0)
        row = {"question": item["question"], "answerable": item["answerable"], "answer": res["answer"]}
        if item["answerable"]:
            n_ans += 1
            kws = item["answer_keywords_any"]
            hit = any(c["source"] == item["source"] and contains_any(c["text"], kws) for c in res["sources"])
            ok = contains_any(res["answer"], kws)
            hits += hit
            correct += ok
            row.update({"retrieval_hit": hit, "answer_correct": ok})
        else:
            n_unans += 1
            ok = abstained(res["answer"])
            abst += ok
            row.update({"abstained_correctly": ok})
        details.append(row)
        print(f"{'OK ' if ok else 'BAD'} | {item['question']}\n      -> {res['answer'][:140]}")

    summary = {
        "llm": rag.llm_name, "top_k": args.top_k,
        "num_answerable": n_ans, "num_unanswerable": n_unans,
        "retrieval_hit_rate": round(hits / n_ans, 3) if n_ans else None,
        "answer_accuracy": round(correct / n_ans, 3) if n_ans else None,
        "abstention_rate": round(abst / n_unans, 3) if n_unans else None,
        "avg_latency_seconds": round(sum(times) / len(times), 2),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text(json.dumps({"summary": summary, "details": details}, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out / 'eval_results.json'}")


if __name__ == "__main__":
    main()
