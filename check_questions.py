"""Sanity-check eval/questions.json against the documents BEFORE running the evaluation.

For every answerable question it verifies that the source file exists and that at least one gold keyword
actually appears in that file's text. This catches wrong gold keywords or unreadable PDFs.
Usage: python check_questions.py
"""
import argparse
import json
import re
from pathlib import Path

from ingest import load_documents


def norm(s):
    s = s.lower().replace("\u2019", "'")
    s = re.sub(r"-\s+", "-", s)      # re-join hyphenated line breaks: "non- parametric" -> "non-parametric"
    return re.sub(r"\s+", " ", s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="docs")
    p.add_argument("--questions", default="eval/questions.json")
    args = p.parse_args()

    pages = load_documents(args.docs)
    text = {}
    for name, _, t in pages:
        text[name] = text.get(name, "") + " " + t
    print("Documents found:", ", ".join(f"{n} ({len(t.split())} words)" for n, t in sorted(text.items())) or "NONE")
    text = {k: norm(v) for k, v in text.items()}

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    problems = 0
    answerable = 0
    for item in questions:
        if not item["answerable"]:
            continue
        answerable += 1
        src = item["source"]
        if src not in text:
            print(f"MISSING FILE  {src}: {item['question']}")
            problems += 1
            continue
        if not any(norm(k) in text[src] for k in item["answer_keywords_any"]):
            print(f"NO KEYWORD    {src}: {item['question']}  (tried {item['answer_keywords_any']})")
            problems += 1

    unans = len(questions) - answerable
    print(f"\n{answerable} answerable + {unans} unanswerable questions checked; {problems} problem(s).")
    if problems == 0:
        print("All gold keywords were found in their source documents. Safe to run: python evaluate.py")
    else:
        print("Fix or remove the problem questions (or send them to me) before evaluating.")


if __name__ == "__main__":
    main()
