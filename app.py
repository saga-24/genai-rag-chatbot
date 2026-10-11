"""Gradio web UI for the RAG chatbot.

Usage: python app.py            (local; open the printed URL)
       python app.py --share    (public link; handy on Colab)
"""
import argparse

import gradio as gr

from rag import RAGPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", default="index")
    p.add_argument("--llm", default=None)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    rag = RAGPipeline(args.index_dir, args.llm, args.top_k)

    def answer(question):
        if not question or not question.strip():
            return "Please type a question.", ""
        res = rag.ask(question.strip())
        sources = "\n\n".join(
            f"**[{n}] {c['source']}** (page {c['page']}, similarity {c['score']:.2f})\n\n> {c['text'][:400]}..."
            for n, c in enumerate(res["sources"], 1))
        return res["answer"], sources

    with gr.Blocks(title="Document Q&A (RAG)") as demo:
        gr.Markdown("# Document Q&A Chatbot (RAG)\nAsk questions about the indexed documents. "
                    "Answers are generated only from retrieved passages, shown below.")
        q = gr.Textbox(label="Your question", lines=2, placeholder="e.g. What is the CLS token used for?")
        btn = gr.Button("Ask", variant="primary")
        ans = gr.Textbox(label="Answer", lines=6)
        src = gr.Markdown()
        btn.click(answer, inputs=q, outputs=[ans, src])
        q.submit(answer, inputs=q, outputs=[ans, src])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
