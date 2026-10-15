"""Unit tests for the similarity-threshold guard added to RAGPipeline.

These tests do NOT require a downloaded model or a built index. They mock
the heavy components (embedder, index, LLM) so they run in milliseconds.

Run with:
    python -m pytest tests/test_threshold_guard.py -v

or without pytest (from the project root):
    python tests/test_threshold_guard.py
"""
import sys
import unittest
from pathlib import Path

# Ensure the project root is on sys.path so `from rag import ...` works
# whether tests are run from the project root or from within tests/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Build a lightweight fake RAGPipeline that mirrors the real one's interface
# without loading models or reading from disk.
# ---------------------------------------------------------------------------


# Inline the constant to avoid importing rag.py (which needs torch).
_NO_ANSWER = "I don't know based on the provided documents."


def _make_fake_pipeline(chunks, scores_by_question, min_score=None):
    """Return a RAGPipeline-like object with mocked internals.

    Args:
        chunks: list of chunk dicts (text, source, page).
        scores_by_question: dict mapping question string to list of float scores
            (must be same length as `chunks`).
        min_score: threshold to pass to the pipeline.
    """

    class FakePipeline:
        """Mirrors RAGPipeline's public interface without any I/O."""

        def __init__(self):
            self.chunks = chunks
            self.min_score = min_score
            self.llm_name = "fake-llm"
            self._generated = "Fake generated answer."

        def retrieve(self, question, k=None):
            k = k or 4
            scores = scores_by_question.get(question, [0.5] * len(self.chunks))
            candidates = [
                {**self.chunks[i], "score": float(scores[i])}
                for i in range(min(k, len(self.chunks)))
            ]
            if self.min_score is not None:
                candidates = [c for c in candidates if c["score"] >= self.min_score]
            return candidates

        def generate(self, messages, max_new_tokens=256):  # noqa: D102
            return self._generated

        def build_messages(self, question, contexts):  # noqa: D102
            return []

        def ask(self, question, k=None):
            contexts = self.retrieve(question, k)
            if not contexts:
                return {"answer": _NO_ANSWER, "sources": []}
            answer = self.generate(self.build_messages(question, contexts))
            return {"answer": answer, "sources": contexts}

    return FakePipeline()


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

CHUNKS = [
    {"text": "LoRA freezes the pre-trained model weights.", "source": "lora.pdf", "page": 3},
    {"text": "ResNet uses skip connections to ease training.", "source": "resnet.pdf", "page": 1},
    {"text": "BERT is pre-trained on masked language modelling.", "source": "bert.pdf", "page": 2},
    {"text": "The Transformer uses self-attention.", "source": "transformer.pdf", "page": 5},
]

NO_ANSWER_STR = "I don't know based on the provided documents."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThresholdGuardDisabled(unittest.TestCase):
    """When min_score is None (default) all chunks pass through."""

    def setUp(self):
        self.pipeline = _make_fake_pipeline(
            chunks=CHUNKS,
            scores_by_question={
                "What does LoRA freeze?": [0.82, 0.41, 0.38, 0.35],
                "What is the boiling point of water?": [0.51, 0.49, 0.48, 0.47],
            },
            min_score=None,
        )

    def test_relevant_question_returns_chunks(self):
        """Relevant question: all 4 chunks returned when no threshold set."""
        res = self.pipeline.ask("What does LoRA freeze?")
        self.assertEqual(len(res["sources"]), 4)
        self.assertEqual(res["answer"], "Fake generated answer.")

    def test_off_topic_question_still_returns_chunks_without_threshold(self):
        """Off-topic question: chunks still returned when min_score is None."""
        res = self.pipeline.ask("What is the boiling point of water?")
        self.assertEqual(len(res["sources"]), 4)
        self.assertNotEqual(res["answer"], NO_ANSWER_STR)


class TestThresholdGuardEnabled(unittest.TestCase):
    """When min_score is set, low-scoring chunks are filtered out."""

    def _make(self, min_score):
        return _make_fake_pipeline(
            chunks=CHUNKS,
            scores_by_question={
                "What does LoRA freeze?": [0.82, 0.79, 0.41, 0.38],
                "What is the boiling point of water?": [0.55, 0.51, 0.49, 0.47],
            },
            min_score=min_score,
        )

    def test_relevant_question_passes_threshold(self):
        """Relevant question with threshold=0.65: only high-scoring chunks survive."""
        pipeline = self._make(min_score=0.65)
        res = pipeline.ask("What does LoRA freeze?")
        # Scores 0.82 and 0.79 survive; 0.41 and 0.38 are filtered.
        self.assertEqual(len(res["sources"]), 2)
        self.assertGreaterEqual(res["sources"][0]["score"], 0.65)
        self.assertGreaterEqual(res["sources"][1]["score"], 0.65)
        self.assertEqual(res["answer"], "Fake generated answer.")

    def test_off_topic_question_triggers_abstention(self):
        """Off-topic question with threshold=0.65: all chunks filtered → _NO_ANSWER."""
        pipeline = self._make(min_score=0.65)
        res = pipeline.ask("What is the boiling point of water?")
        self.assertEqual(res["sources"], [])
        self.assertEqual(res["answer"], NO_ANSWER_STR)

    def test_llm_not_called_when_threshold_filters_all(self):
        """When all chunks are filtered, generate() must NOT be called."""
        pipeline = self._make(min_score=0.65)
        called = []
        original_generate = pipeline.generate

        def spy_generate(messages, **kwargs):
            called.append(True)
            return original_generate(messages, **kwargs)

        pipeline.generate = spy_generate
        pipeline.ask("What is the boiling point of water?")
        self.assertEqual(called, [], "generate() was called despite no chunks passing the threshold")

    def test_generate_called_when_chunks_pass_threshold(self):
        """When chunks pass the threshold, generate() must be called."""
        pipeline = self._make(min_score=0.65)
        called = []
        original_generate = pipeline.generate

        def spy_generate(messages, **kwargs):
            called.append(True)
            return original_generate(messages, **kwargs)

        pipeline.generate = spy_generate
        pipeline.ask("What does LoRA freeze?")
        self.assertTrue(len(called) > 0, "generate() was not called even though chunks passed the threshold")


class TestThresholdBoundaryConditions(unittest.TestCase):
    """Test exact boundary behaviour (score == threshold)."""

    def _make_single_chunk(self, score, min_score):
        chunks = [{"text": "Test passage.", "source": "test.pdf", "page": 1}]
        return _make_fake_pipeline(
            chunks=chunks,
            scores_by_question={"test question": [score]},
            min_score=min_score,
        )

    def test_score_exactly_at_threshold_passes(self):
        """A chunk whose score equals the threshold exactly should pass (>=)."""
        pipeline = self._make_single_chunk(score=0.70, min_score=0.70)
        res = pipeline.ask("test question")
        self.assertEqual(len(res["sources"]), 1)
        self.assertAlmostEqual(res["sources"][0]["score"], 0.70)

    def test_score_just_below_threshold_is_filtered(self):
        """A chunk just below the threshold should be filtered out."""
        pipeline = self._make_single_chunk(score=0.6999, min_score=0.70)
        res = pipeline.ask("test question")
        self.assertEqual(res["sources"], [])
        self.assertEqual(res["answer"], NO_ANSWER_STR)

    def test_score_just_above_threshold_passes(self):
        """A chunk just above the threshold should pass."""
        pipeline = self._make_single_chunk(score=0.7001, min_score=0.70)
        res = pipeline.ask("test question")
        self.assertEqual(len(res["sources"]), 1)


class TestRetrieveFiltering(unittest.TestCase):
    """Test retrieve() directly to verify per-chunk filtering logic."""

    def test_mixed_scores_only_above_threshold_returned(self):
        """retrieve() returns only chunks whose score >= min_score."""
        chunks = [
            {"text": f"chunk {i}", "source": f"doc{i}.pdf", "page": i}
            for i in range(5)
        ]
        scores = [0.90, 0.75, 0.65, 0.55, 0.40]
        pipeline = _make_fake_pipeline(
            chunks=chunks,
            scores_by_question={"q": scores},
            min_score=0.65,
        )
        result = pipeline.retrieve("q")
        # scores >= 0.65: indices 0, 1, 2
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertGreaterEqual(chunk["score"], 0.65)

    def test_all_scores_above_threshold(self):
        """All chunks pass when every score is above the threshold."""
        chunks = [{"text": "x", "source": "a.pdf", "page": 1} for _ in range(4)]
        scores = [0.85, 0.80, 0.75, 0.70]
        pipeline = _make_fake_pipeline(
            chunks=chunks,
            scores_by_question={"q": scores},
            min_score=0.65,
        )
        result = pipeline.retrieve("q")
        self.assertEqual(len(result), 4)

    def test_all_scores_below_threshold(self):
        """All chunks filtered when every score is below the threshold."""
        chunks = [{"text": "x", "source": "a.pdf", "page": 1} for _ in range(4)]
        scores = [0.60, 0.55, 0.50, 0.45]
        pipeline = _make_fake_pipeline(
            chunks=chunks,
            scores_by_question={"q": scores},
            min_score=0.65,
        )
        result = pipeline.retrieve("q")
        self.assertEqual(result, [])


class TestBackwardCompatibility(unittest.TestCase):
    """Verify that min_score=None preserves the original behaviour."""

    def test_no_threshold_returns_all_chunks_even_low_scores(self):
        """With min_score=None, even chunks with score 0.0 are returned."""
        chunks = [{"text": f"chunk {i}", "source": f"doc{i}.pdf", "page": 1} for i in range(4)]
        scores = [0.50, 0.40, 0.30, 0.10]
        pipeline = _make_fake_pipeline(
            chunks=chunks,
            scores_by_question={"off-topic question": scores},
            min_score=None,
        )
        result = pipeline.retrieve("off-topic question")
        self.assertEqual(len(result), 4)

    def test_no_threshold_ask_always_calls_llm(self):
        """With min_score=None, ask() always calls generate() regardless of scores."""
        chunks = [{"text": "chunk", "source": "doc.pdf", "page": 1}]
        pipeline = _make_fake_pipeline(
            chunks=chunks,
            scores_by_question={"any question": [0.10]},
            min_score=None,
        )
        called = []
        original_generate = pipeline.generate

        def spy(messages, **kwargs):
            called.append(True)
            return original_generate(messages, **kwargs)

        pipeline.generate = spy
        pipeline.ask("any question")
        self.assertTrue(len(called) > 0)


# ---------------------------------------------------------------------------
# Allow running without pytest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
