import contextlib
import copy
import io
import unittest

from evaluate_decisions import evaluate, interpret_answers, source_state
from trend_hunter.decisions import BudgetExceeded


CASE = {
    "case_id": "example-1",
    "source_id": "source-1",
    "source_text": "原始消息",
    "source_title": "标题",
    "literal_english_translation": "GOLD TRANSLATION",
    "intended_meaning": "GOLD INTERPRETATION",
    "expected": {"relevance": "relevant", "stance": "positive"},
    "confidence": {"level": "high"},
    "context": [{"source_id": "parent-1", "source_text": "上文", "source_title": None,
                 "intended_meaning": "GOLD PARENT INTERPRETATION"}],
}


def answers(confidence=0.9):
    return {
        "relevance": {"type": "choice", "choice": "relevant", "confidence": confidence,
                      "probabilities": {"relevant": 0.9, "off_topic": 0.05, "uncertain": 0.05}},
        "stance": {"type": "choice", "choice": "positive", "confidence": 0.9,
                   "probabilities": {"positive": 0.9, "negative": 0.025, "mixed": 0.025, "neutral": 0.025, "uncertain": 0.025}},
    }


class FakeDecisions:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.stats = {"model_calls": 0}

    @property
    def usage(self):
        count = self.stats["model_calls"]
        return {"input_tokens": count * 100, "output_tokens": count * 10,
                "total_tokens": count * 110, "accounted_tokens": count * 110,
                "reserved_tokens": 0, "attempts": count, "target_exceeded": False}

    def ask(self, state, questions):
        self.calls.append((state, questions))
        self.stats["model_calls"] += 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class EvaluationTests(unittest.TestCase):
    def test_only_original_source_and_context_reach_model(self):
        state = source_state(CASE)
        self.assertEqual(state["current_message"]["text"], "原始消息")
        self.assertEqual(state["context_for_interpretation_only"][0]["text"], "上文")
        self.assertNotIn("GOLD", str(state))
        self.assertNotIn("expected", state)

    def test_low_confidence_abstention_preserves_raw_choice(self):
        prediction, detail = interpret_answers(answers(0.54), 0.55)
        self.assertEqual(prediction["relevance"], "uncertain")
        self.assertEqual(detail["relevance"]["raw_choice"], "relevant")
        self.assertTrue(detail["relevance"]["low_confidence_abstention"])
        self.assertEqual(prediction["stance"], "positive")

    def test_one_request_per_case_and_separate_coverage(self):
        decisions = FakeDecisions([answers(), answers(0.54)])
        with contextlib.redirect_stderr(io.StringIO()):
            report = evaluate([copy.deepcopy(CASE), copy.deepcopy(CASE)], decisions)
        self.assertEqual(len(decisions.calls), 2)
        self.assertEqual(set(decisions.calls[0][1]), {"relevance", "stance"})
        self.assertEqual(report["counts"]["relevance"]["eligible"], 2)
        self.assertEqual(report["counts"]["relevance"]["agreed"], 1)
        self.assertEqual(report["counts"]["relevance"]["raw_agreed"], 2)
        self.assertEqual(report["counts"]["relevance"]["coverage"], 0.5)
        self.assertEqual(report["run_usage"]["total_tokens"], 220)

    def test_medium_review_labels_excluded_from_high_confidence_denominator(self):
        case = copy.deepcopy(CASE)
        case["confidence"]["level"] = "medium"
        with contextlib.redirect_stderr(io.StringIO()):
            report = evaluate([case], FakeDecisions([answers()]))
        self.assertEqual(report["counts"]["relevance"]["eligible"], 0)
        self.assertIsNone(report["counts"]["relevance"]["agreement"])

    def test_budget_failure_stops_and_preserves_partial_report(self):
        decisions = FakeDecisions([answers(), BudgetExceeded("stop")])
        with contextlib.redirect_stderr(io.StringIO()):
            report = evaluate([copy.deepcopy(CASE)] * 3, decisions)
        self.assertFalse(report["complete"])
        self.assertEqual(report["attempted_cases"], 2)
        self.assertEqual(report["stop_reason"], "token_budget")
        self.assertEqual(report["parse_or_transport_failures"], 1)
        self.assertEqual(len(decisions.calls), 2)


if __name__ == "__main__":
    unittest.main()
