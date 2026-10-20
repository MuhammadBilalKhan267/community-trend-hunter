"""Pipeline seams with actual preparation/review policy and mocked providers.

No model server, credential file or paid API is used. Fixture vectors supply
known candidate topics; Chinese interpretation is deliberately not asserted by
these orchestration tests.
"""
from copy import deepcopy
from contextlib import closing
from datetime import date, timedelta
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from trend_hunter import pipeline
from trend_hunter import candidates as candidate_helpers
from trend_hunter.local_models import LocalGenerationError, LocalModelError, LocalModels
from trend_hunter.embedding_identity import embedding_key, save_profile
from trend_hunter.openai_generation import GenerationOutputError, GenerationServiceError, OpenAIGeneration


START = date(2026, 7, 1)
AS_OF = "2026-07-14"
DIMENSION = 128


def record(identifier, day, text, parent=None, hour=23):
    return {"id": identifier, "parent_id": parent, "platform": "Forum", "channel": "反馈",
            "user_name": f"voice-{identifier}", "title": None, "content": text,
            "created_at": f"{(START + timedelta(days=day)).isoformat()}T{hour:02d}:30:00+08:00"}


def fixture(topic_count=1, stable=False, large=False):
    rows = []
    for day in range(14):
        for index in range(10):
            rows.append(record(f"background-{day}-{index}", day,
                               f"BACKGROUND 今天分享游戏外的日常经历 {day}-{index}"))
        for topic in range(topic_count):
            count = 3 if stable else (5 if day < 7 else 35) if large else (1 if day < 7 else 4)
            for index in range(count):
                rows.append(record(f"topic-{topic}-{day}-{index}", day,
                                   f"TOPIC_{topic} 版本2.0游戏反馈说明 {day}-{index}"))
    return rows


def topic_of(value):
    match = re.search(r"TOPIC_(\d+)", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return int(match.group(1)) if match else None


def writer_topic(payload):
    sources = payload["evidence"]["sources"]
    focal = [text for source in sources for text in source.get("reviewed_focal_texts", [])]
    return topic_of(focal or sources)


class FakeModels:
    def __init__(self, *, cached_embeddings=False, embedding_cache_only=False, **_):
        self.embedding_cache_only = embedding_cache_only
        self.cached_embeddings = cached_embeddings or embedding_cache_only
        self.requests = []
        self.stats = {"embedding_calls": 0, "embedding_cache_hits": 0,
                      "chat_calls": 0, "chat_cache_hits": 0}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def health(self, required_roles=("embedding", "chat")):
        return {role: {"name": "fake-model", "digest": f"fixture-{role}"} for role in required_roles}

    def embed(self, texts):
        self.requests.append(("embed", list(texts)))
        if self.cached_embeddings:
            self.stats["embedding_cache_hits"] += len(texts)
        else:
            self.stats["embedding_calls"] += 1
        vectors = []
        for text in texts:
            topic = topic_of(text)
            vector = np.zeros(DIMENSION, dtype=np.float32)
            vector[topic if topic is not None else DIMENSION - 1] = 1
            vectors.append(vector)
        return np.asarray(vectors)

    def chat_json(self, system, payload, schema, max_tokens):
        self.requests.append(("chat", system, deepcopy(payload), deepcopy(schema), max_tokens))
        self.stats["chat_calls"] += 1
        if payload["task"] in {"draft_output", "repair_output"}:
            topic = writer_topic(payload)
            return {"title": f"玩家问题{topic}", "title_en": f"Concern {topic}",
                    "proposed_action": "Review the supplied player feedback."}
        raise AssertionError(f"Unexpected generation task {payload['task']}")

    def unload(self, model):
        self.requests.append(("unload", model))


class FakeHostedWriter:
    def __init__(self, failure=None):
        self.failure = failure
        self.requests = []
        self.stats = {"requests": 0, "cache_hits": 0, "api_calls": 0,
                      "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def chat_json(self, system, payload, schema, max_tokens):
        self.requests.append((system, deepcopy(payload), deepcopy(schema), max_tokens))
        self.stats["requests"] += 1
        self.stats["api_calls"] += 1
        if self.failure and self.stats["api_calls"] == self.failure[0]:
            raise self.failure[1]
        topic = writer_topic(payload)
        self.stats["input_tokens"] += 20
        self.stats["output_tokens"] += 5
        self.stats["total_tokens"] += 25
        return {"title": f"玩家问题{topic}", "title_en": f"Concern {topic}",
                "proposed_action": "Review the supplied player feedback."}


def confident(value):
    return {"type": "choice", "choice": value, "confidence": 1., "probabilities": {value: 1.}}


def restore_writer_ids(writer_state, original_state):
    """Compare writer handles against the separately observed public boundary."""
    identifiers = {written["id"]: original["id"]
                   for collection in ("sources", "context")
                   for written, original in zip(writer_state.get(collection, []), original_state.get(collection, []), strict=True)}

    def restore(value):
        if isinstance(value, str):
            return identifiers.get(value, value)
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, dict):
            return {identifiers.get(key, key): restore(item) for key, item in value.items()}
        return value

    return restore(writer_state)


class FakeJev:
    def __init__(self, *, rejected_topics=(), match_limit=None, opposing_index=None, **_):
        self.rejected_topics = set(rejected_topics)
        self.match_limit = match_limit
        self.opposing_index = opposing_index
        self.requests = []
        self.stats = {"api_calls": 0, "cache_hits": 0, "input_tokens": 0,
                      "output_tokens": 0, "total_tokens": 0, "calls": 0}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                      "accounted_tokens": 0, "target_exceeded": False,
                      "max_exceeded": False, "reserved_tokens": 0}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def close(self):
        return None

    def ask(self, state, questions):
        self.requests.append((deepcopy(state), deepcopy(questions)))
        self.stats["api_calls"] += 1
        self.stats["calls"] += 1
        answers = {}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0. if name in {"action_unsupported_facts", "unsupported_scope"} else 1.}
                continue
            criteria = question["criteria"]
            proposals = state.get("proposals", {}).get("concerns", [])
            prefix = re.match(r"c(\d+)_", name)
            proposal = proposals[int(prefix.group(1))] if prefix else {}
            topic = topic_of(proposal.get("definition", ""))
            if name == "sibling_relation":
                focal_topics = [topic_of(item["definition"]) for item in proposals]
                value = "same" if len(set(focal_topics)) == 1 else "distinct"
            elif name.endswith("_relevance"):
                value = "background" if topic is None else "irrelevant" if topic in self.rejected_topics else "relevant"
            elif name.endswith("_scope"):
                value = "supported"
            elif name.endswith("_release"):
                events = question["instructions"]["eligible_events"]
                material = json.dumps(state["candidate"]["targets"], ensure_ascii=False)
                value = next((key for key, event in events.items()
                              if event.get("version") not in {None, "", "-"}
                              and event["version"] in material), "none")
            elif name.endswith("_duplicate"):
                existing = next((item for item in state.get("accepted_concerns", [])
                                 if topic_of(item["definition"]) == topic), None)
                value = existing["reference"] if existing else "distinct"
            elif re.search(r"_t\d+_match$", name):
                index = int(re.search(r"_t(\d+)_match$", name).group(1))
                target = state["candidate"]["targets"][index]
                if self.match_limit is not None and index >= self.match_limit:
                    value = "other"
                else:
                    value = "match" if topic_of(target["text"]) == topic else "other"
            elif re.search(r"_t\d+_role$", name):
                index = int(re.search(r"_t(\d+)_role$", name).group(1))
                value = "disagreement" if index == self.opposing_index else "experience"
            elif name.endswith("_stance"):
                index = int(re.search(r"_t(\d+)_stance$", name).group(1))
                value = "positive" if index == self.opposing_index else "negative"
            elif re.search(r"_r\d+_match$", name):
                value = "match"
            else:
                value = "unclear" if "unclear" in criteria else next(iter(criteria))
            answers[name] = confident(value)
        return answers


def fake_clusters(vectors, _target_size):
    return np.argmax(vectors, axis=1), np.eye(DIMENSION, dtype=np.float32)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trend-hunter-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write(self, name, rows):
        path = self.directory / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def run_pipeline(self, path, releases=None, *, cached_embeddings=False,
                     rejected_topics=(), match_limit=None, opposing_index=None, **settings):
        models, reviewers = [], []

        def construct_model(**kwargs):
            instance = FakeModels(cached_embeddings=cached_embeddings, **kwargs)
            models.append(instance)
            return instance

        def construct_reviewer(**kwargs):
            instance = FakeJev(rejected_topics=rejected_topics, match_limit=match_limit,
                               opposing_index=opposing_index, **kwargs)
            reviewers.append(instance)
            return instance

        options = {"max_reviews": 30, "cluster_cache": False, "neighbor_reviews": 0}
        options.update(settings)
        cache = self.directory / "must-not-exist.sqlite3"
        with patch.object(pipeline, "LocalModels", side_effect=construct_model), \
                patch.object(pipeline, "JevDecisions", side_effect=construct_reviewer), \
                patch.object(pipeline, "cluster_vectors", side_effect=fake_clusters):
            result, diagnostics = pipeline.run(str(path), AS_OF,
                                               releases_path=str(releases) if releases else None,
                                               cache_path=str(cache),
                                               decision_cache_path=str(self.directory / "must-not-exist-jev.sqlite3"),
                                               **options)
        self.assertFalse(cache.exists())
        self.assertFalse((self.directory / "must-not-exist-jev.sqlite3").exists())
        self.assertEqual(len(models), 1)
        self.assertEqual(len(reviewers), 1)
        return result, diagnostics, models[0], reviewers[0]

    def reviews(self, reviewer):
        return [state for state, _ in reviewer.requests if state.get("task") == "review_sampled_candidate"]

    def seed_embedding_cache(self, source, cache):
        _, eligible, _ = pipeline.prepare(str(source), AS_OF)
        texts, _ = pipeline.unique_units(pipeline.make_units(eligible))
        vectors = FakeModels().embed(texts)
        profile = {"model_name": "bge-m3:latest", "digest": "fixture-encoder-digest",
                   "options": {"num_ctx": 512, "num_batch": 512}, "dimensions": DIMENSION}
        with closing(sqlite3.connect(cache)) as db, db:
            db.execute("CREATE TABLE embeddings (cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL)")
            save_profile(db, profile)
            db.executemany("INSERT INTO embeddings VALUES (?, ?, ?)", [
                (embedding_key(profile["model_name"], profile["digest"], text, profile["options"]),
                 DIMENSION, vector.astype("<f4").tobytes()) for text, vector in zip(texts, vectors, strict=True)])
        return texts, profile

    def run_real_cached_pipeline(self, source, cache, *, prepare_only=False):
        writer, reviewer = FakeHostedWriter(), FakeJev()
        with patch.object(LocalModels, "_transport", side_effect=AssertionError("Ollama transport must not run")) as ollama, \
                patch.object(LocalModels, "health", side_effect=AssertionError("Ollama health must not run")) as health, \
                patch.object(pipeline, "OpenAIGeneration", return_value=writer) as hosted, \
                patch.object(pipeline, "JevDecisions", return_value=reviewer) as decisions, \
                patch.object(pipeline, "cluster_vectors", side_effect=fake_clusters):
            result, diagnostics = pipeline.run(str(source), AS_OF, cache_path=str(cache),
                chat_provider="openai", chat_model=None if prepare_only else "explicit-hosted-model",
                prepare_only=prepare_only, cluster_cache=False, max_reviews=30, neighbor_reviews=0,
                embedding_residency_inputs=64, base_url="unused-offline-endpoint")
        ollama.assert_not_called()
        health.assert_not_called()
        return result, diagnostics, hosted, decisions

    def test_hosted_pipeline_reads_actual_cached_vectors_without_any_ollama_calls(self):
        source = self.write("real-cached-hosted.jsonl", fixture())
        cache = self.directory / "cached.sqlite3"
        texts, profile = self.seed_embedding_cache(source, cache)
        result, diagnostics, hosted, decisions = self.run_real_cached_pipeline(source, cache)
        self.assertEqual(len(result["trends"]), 1)
        self.assertEqual(diagnostics["embedding_identity"], profile)
        self.assertTrue(diagnostics["model_config"]["embedding_cache_only"])
        self.assertEqual(diagnostics["local_usage"]["embedding_cache_hits"], len(texts))
        self.assertEqual(diagnostics["local_usage"]["api_calls"], 0)
        hosted.assert_called_once()
        decisions.assert_called_once()

    def test_incomplete_hosted_embedding_cache_stops_before_either_paid_provider(self):
        source = self.write("missing-cached-hosted.jsonl", fixture())
        cache = self.directory / "incomplete.sqlite3"
        texts, profile = self.seed_embedding_cache(source, cache)
        with closing(sqlite3.connect(cache)) as db, db:
            db.execute("DELETE FROM embeddings WHERE cache_key = ?", (
                embedding_key(profile["model_name"], profile["digest"], texts[-1], profile["options"]),))
        result, diagnostics, hosted, decisions = self.run_real_cached_pipeline(source, cache)
        self.assertEqual(result["trends"], [])
        self.assertEqual(diagnostics["stop_reason"], "embedding_cache_error")
        self.assertIn("missing", diagnostics["embedding_cache_error"].lower())
        self.assertIn("ingest", diagnostics["embedding_cache_error"].lower())
        hosted.assert_not_called()
        decisions.assert_not_called()

    def test_missing_hosted_cache_file_fails_without_creating_it_or_paid_providers(self):
        source = self.write("no-cached-hosted.jsonl", fixture())
        cache = self.directory / "not-created.sqlite3"
        result, diagnostics, hosted, decisions = self.run_real_cached_pipeline(source, cache)
        self.assertEqual(result["trends"], [])
        self.assertEqual(diagnostics["stop_reason"], "embedding_cache_error")
        self.assertIn("restore", diagnostics["embedding_cache_error"].lower())
        self.assertFalse(cache.exists())
        hosted.assert_not_called()
        decisions.assert_not_called()

    def test_real_cached_preparation_needs_no_ollama_writer_or_decision_provider(self):
        source = self.write("real-cached-prepare.jsonl", fixture())
        cache = self.directory / "prepare.sqlite3"
        self.seed_embedding_cache(source, cache)
        result, diagnostics, hosted, decisions = self.run_real_cached_pipeline(source, cache, prepare_only=True)
        self.assertEqual(result["trends"], [])
        self.assertTrue(diagnostics["candidate_packets"])
        self.assertEqual(diagnostics["local_usage"]["api_calls"], 0)
        hosted.assert_not_called()
        decisions.assert_not_called()

    def test_hosted_generation_uses_cached_local_vectors_and_the_same_jev_checks(self):
        writer = FakeHostedWriter()
        with patch.object(pipeline, "OpenAIGeneration", return_value=writer) as constructor:
            result, diagnostics, local, reviewer = self.run_pipeline(
                self.write("hosted.jsonl", fixture()), cached_embeddings=True,
                chat_provider="openai", chat_model="explicit-hosted-model")
        self.assertEqual(len(result["trends"]), 1)
        self.assertEqual(constructor.call_args.kwargs["model"], "explicit-hosted-model")
        self.assertEqual(diagnostics["model_config"]["chat_provider"], "openai")
        self.assertTrue(local.embedding_cache_only)
        self.assertEqual(diagnostics["generation_usage"]["api_calls"], 1)
        self.assertEqual(diagnostics["generation_usage"]["total_tokens"], 25)
        self.assertEqual(diagnostics["local_usage"]["chat_calls"], 0)
        self.assertEqual(local.stats["embedding_calls"], 0)
        self.assertGreater(local.stats["embedding_cache_hits"], 0)
        self.assertFalse(any(request[0] in {"chat", "unload"} for request in local.requests))
        self.assertEqual(writer.requests[0][1]["task"], "draft_output")
        self.assertTrue(any("title_en" in questions for _, questions in reviewer.requests))

    def test_prepare_only_does_not_construct_hosted_writer_or_require_its_model_or_key(self):
        local = FakeModels(cached_embeddings=True)
        with patch.object(pipeline, "LocalModels", return_value=local), \
                patch.object(pipeline, "OpenAIGeneration") as hosted, \
                patch.object(pipeline, "JevDecisions") as decisions, \
                patch.object(pipeline, "cluster_vectors", side_effect=fake_clusters):
            result, diagnostics = pipeline.run(str(self.write("prepare-hosted.jsonl", fixture())), AS_OF,
                cache_path=str(self.directory / "unused.sqlite3"), chat_provider="openai",
                prepare_only=True, cluster_cache=False)
        self.assertEqual(result["trends"], [])
        self.assertTrue(diagnostics["candidate_packets"])
        hosted.assert_not_called()
        decisions.assert_not_called()
        self.assertEqual(local.stats["chat_calls"], 0)

    def test_hosted_transport_failure_preserves_earlier_checked_outputs_without_local_fallback(self):
        writer = FakeHostedWriter(failure=(2, GenerationServiceError("Hosted transport unavailable")))
        with patch.object(pipeline, "OpenAIGeneration", return_value=writer):
            result, diagnostics, local, _ = self.run_pipeline(
                self.write("hosted-failure.jsonl", fixture(topic_count=2)),
                chat_provider="openai", chat_model="explicit-hosted-model")
        self.assertEqual(len(result["trends"]), 1)
        self.assertEqual(diagnostics["stop_reason"], "generation_service_error")
        self.assertIn("Hosted transport", diagnostics["generation_error"])
        self.assertEqual(diagnostics["generation_usage"]["api_calls"], 2)
        self.assertEqual(local.stats["chat_calls"], 0)

    def test_unusable_hosted_draft_is_recorded_once_without_a_local_repair(self):
        writer = FakeHostedWriter(failure=(1, GenerationOutputError("Hosted output incomplete")))
        with patch.object(pipeline, "OpenAIGeneration", return_value=writer):
            result, diagnostics, local, _ = self.run_pipeline(
                self.write("hosted-output-failure.jsonl", fixture()),
                chat_provider="openai", chat_model="explicit-hosted-model")
        self.assertEqual(result["trends"], [])
        self.assertEqual(len(diagnostics["generation_failures"]), 1)
        self.assertEqual(writer.stats["api_calls"], 1)
        self.assertEqual(local.stats["chat_calls"], 0)

    def test_partial_hosted_response_preserves_checked_outputs_and_stops_with_sanitized_error(self):
        class PartialResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                raise IncompleteRead(b"synthetic-secret private-source", 100)

        writer = OpenAIGeneration(model="explicit-hosted-model", api_key="synthetic-secret", cache_path=":memory:")
        first = {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": json.dumps({"title": "玩家反馈", "title_en": "Player feedback",
                                                       "proposed_action": "Review the supplied feedback."})}]}],
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}}
        writer._opener = Mock()
        writer._opener.open.side_effect = [io.BytesIO(json.dumps(first).encode()), PartialResponse()]
        with patch.object(pipeline, "OpenAIGeneration", return_value=writer):
            result, diagnostics, local, _ = self.run_pipeline(
                self.write("partial-hosted-response.jsonl", fixture(topic_count=2)), cached_embeddings=True,
                chat_provider="openai", chat_model="explicit-hosted-model")
        self.assertEqual(len(result["trends"]), 1)
        self.assertEqual(diagnostics["stop_reason"], "generation_service_error")
        self.assertIn("billing may be unknown", diagnostics["generation_error"])
        self.assertNotIn("synthetic-secret", json.dumps(diagnostics))
        self.assertNotIn("private-source", json.dumps(diagnostics))
        self.assertEqual(diagnostics["generation_usage"]["api_calls"], 2)
        self.assertEqual(diagnostics["generation_usage"]["usage_unavailable_calls"], 1)
        self.assertEqual(local.stats["chat_calls"], 0)

    def test_openai_generation_requires_an_explicit_model_before_preparation(self):
        with self.assertRaisesRegex(ValueError, "explicit chat_model"):
            pipeline.run("unread-input.jsonl", AS_OF, chat_provider="openai")

    def test_future_poison_cannot_change_output_statistics_or_either_model_prompt(self):
        prefix = fixture()
        prefix.append(record("early-reaction", 13, "+1", parent="future-parent"))
        future = [record("future-parent", 14, "TOPIC_14 FUTURE_SECRET future claim")]
        future.extend(record(f"future-{index}", 14, f"TOPIC_14 FUTURE_SECRET {index}") for index in range(50))
        future.append(record(prefix[0]["id"], 14, "FUTURE_SECRET duplicate id"))
        full = self.write("full.jsonl", prefix + future)
        truncated = self.write("prefix.jsonl", prefix)
        full_releases = self.directory / "full-releases.csv"
        full_releases.write_text("date,version\n2026-07-10,2.0\n2026-07-15,FUTURE_SECRET\n", encoding="utf-8")
        prefix_releases = self.directory / "prefix-releases.csv"
        prefix_releases.write_text("date,version\n2026-07-10,2.0\n", encoding="utf-8")

        a, ad, am, aj = self.run_pipeline(full, full_releases)
        b, bd, bm, bj = self.run_pipeline(truncated, prefix_releases)
        self.assertTrue(a["trends"], "The invariant must exercise a selected trend and review prompts")
        self.assertEqual(a, b)
        self.assertEqual(ad, bd)
        self.assertEqual(am.requests, bm.requests)
        self.assertEqual(aj.requests, bj.requests)
        self.assertNotIn("FUTURE_SECRET", json.dumps(am.requests + aj.requests))
        final_states = [request[2]["evidence"] for request in am.requests
                        if request[0] == "chat" and request[2]["task"] == "draft_output"]
        self.assertTrue(all(state["as_of"] == AS_OF for state in final_states))
        self.assertNotIn("past_release_calendar", final_states[0])
        self.assertEqual([release["version"] for release in final_states[0]["source_release_associations"]], ["2.0"])

    def test_selection_finishes_initial_batch_and_emits_at_most_eight_distinct_trends(self):
        rows = fixture(topic_count=10)
        result, diagnostics, models, reviewer = self.run_pipeline(self.write("many.jsonl", rows))
        self.assertEqual(len(result["trends"]), 8)
        self.assertEqual(len({trend["title_en"] for trend in result["trends"]}), 8)
        self.assertEqual(len(self.reviews(reviewer)), 10)
        visible = {row["id"]: row for row in rows}
        for trend in result["trends"]:
            self.assertIn(trend["status"], pipeline.STATUSES)
            self.assertEqual(len(trend["voice_ids"]), len(set(trend["voice_ids"])))
            for identifier in trend["voice_ids"]:
                self.assertIn(identifier, visible)
                self.assertLessEqual(visible[identifier]["created_at"][:10], AS_OF)
            self.assertLessEqual(trend["first_seen"], min(visible[i]["created_at"][:10] for i in trend["voice_ids"]))
        self.assertEqual(len(diagnostics["selected"]), 8)
        generations = [request for request in models.requests if request[0] == "chat"]
        self.assertEqual(len(generations), 8)
        self.assertTrue(all(call[2]["task"] == "draft_output" for call in generations))
        reviews = self.reviews(reviewer)
        self.assertTrue(all(len(state["candidate"]["targets"]) <= 10 for state in reviews))
        self.assertTrue(all(state["proposals"]["origin"] == "source_anchors" for state in reviews))
        self.assertEqual(reviewer.stats["api_calls"], len(reviews) + len(generations) + diagnostics["final_duplicate_review_calls"])

    def test_rejected_candidates_trigger_backfill_without_filling_weak_slots(self):
        rows = fixture(topic_count=12)
        result, _, _, reviewer = self.run_pipeline(self.write("backfill.jsonl", rows),
                                                  rejected_topics=range(10))
        self.assertEqual({trend["title_en"] for trend in result["trends"]}, {"Concern 10", "Concern 11"})
        self.assertGreater(len(self.reviews(reviewer)), 10)
        reviewed_topics = {topic_of(state["proposals"]) for state in self.reviews(reviewer)}
        self.assertTrue({10, 11} <= reviewed_topics)

    def test_final_batch_merge_unions_sources_recomputes_history_and_backfills(self):
        original_refresh = pipeline._refresh

        def rank(concept, *args, **kwargs):
            refreshed = original_refresh(concept, *args, **kwargs)
            refreshed["score"] = 100 - topic_of(refreshed["definition"])
            return refreshed

        class FinalDuplicateJev(FakeJev):
            def ask(self, state, questions):
                answers = super().ask(state, questions)
                if state.get("task") == "review_final_topic_duplicates":
                    for name, question in questions.items():
                        evidence = question["instructions"]
                        collections = state["collections"]
                        topics = {topic_of(collections[evidence[key]]["approved_candidate"]["definition"])
                                  for key in ("left_topic_id", "right_topic_id")}
                        for side in ("left", "right"):
                            original_ids = [source["id"] for source in collections[evidence[side + "_topic_id"]]["sources"]]
                            if evidence[side + "_source_ids"] != original_ids:
                                raise AssertionError("Pair source IDs do not bind to the shared original evidence")
                        answers[name] = confident("same" if topics == {0, 1} else "distinct")
                return answers

        with patch(f"{__name__}.FakeJev", FinalDuplicateJev), \
                patch.object(pipeline, "_refresh", side_effect=rank):
            result, diagnostics, models, reviewer = self.run_pipeline(
                self.write("final-duplicates.jsonl", fixture(topic_count=9)))
        self.assertEqual(len(result["trends"]), 8)
        self.assertNotIn("Concern 1", [trend["title_en"] for trend in result["trends"]])
        self.assertIn("Concern 8", [trend["title_en"] for trend in result["trends"]])
        merged = next(entry for entry in diagnostics["selected"] if entry["trend"]["title_en"] == "Concern 0")
        self.assertEqual(merged["history"]["stats"]["raw_count"], 70)
        self.assertEqual(len(merged["sample_review"]["votes"]), 20)
        self.assertEqual(len(merged["evidence"]), 10)
        self.assertEqual({topic_of(source["content"]) for source in merged["evidence"]}, {0, 1})
        checks = diagnostics["final_duplicate_reviews"]
        self.assertEqual(diagnostics["final_duplicate_review_calls"], 2)
        self.assertEqual(len(checks[0]["merges"]), 1)
        self.assertEqual(len(checks[0]["merges"][0]["member_ids"]), 70)
        self.assertEqual(len(checks[0]["merges"][0]["unit_votes"]), 20)
        self.assertEqual(len({check["snapshot"] for check in checks}), len(checks))
        batch_requests = [questions for state, questions in reviewer.requests
                          if state.get("task") == "review_final_topic_duplicates"]
        self.assertEqual([len(questions) for questions in batch_requests], [28, 28])
        self.assertEqual(len([row for row in models.requests if row[0] == "chat"]), 8)

    def test_complete_link_merge_does_not_treat_an_unconfirmed_edge_as_transitive(self):
        self.assertEqual(pipeline._complete_link_groups(["a", "b", "c"], [["a", "b"], ["b", "c"]]),
                         [["a", "b"]])
        self.assertEqual(pipeline._complete_link_groups(["a", "b", "c"],
                         [["a", "b"], ["b", "c"], ["a", "c"]]), [["a", "b", "c"]])

    def test_topic_union_preserves_focal_judgments_context_aliases_and_event_provenance(self):
        rows, eligible, _ = pipeline.prepare(str(self.write("merge-metadata.jsonl", fixture())), AS_OF)
        by_id = {row["id"]: row for row in rows}
        sources = [identifier for identifier in by_id if identifier.startswith("topic-0-")][:3]
        context_id = next(identifier for identifier in by_id if identifier.startswith("background-"))
        event = {"date": "2026-07-05", "version": "2.0", "notes": "A release"}
        proof = {"name": "Name", "alias": "Alias", "evidence_ids": [context_id], "source_id": context_id}

        def item(key, identifiers, stance, role, context_role, possible):
            return {"id": key, "definition": key, "member_ids": set(identifiers),
                    "unit_votes": {identifier: {identifier + ":0": {
                        "match": "match", "stance": stance, "role": role, "text": by_id[identifier]["content"]}}
                        for identifier in identifiers},
                    "candidate_ids": [key + ":candidate"], "matched_response_ids": [key + ":response"],
                    "contexts": {context_id: {"id": context_id, "role": context_role, "for_ids": identifiers}},
                    "aliases": [deepcopy(proof)], "reviewed_release_associations": [deepcopy(event)],
                    "engagement": {"sampled_attribution_supported": False, "possible_response_count": possible}}

        a = item("a", sources[:2], "negative", "experience", "possible_preceding_context", 4)
        b = item("b", sources[1:], "positive", "disagreement", "explicit_parent", 7)
        before = deepcopy([a, b])
        combined = pipeline._merge_concepts([a, b], by_id, eligible, AS_OF, {"replies": {}}, [event])
        self.assertEqual([a, b], before)
        self.assertEqual(combined["member_ids"], set(sources))
        self.assertEqual(combined["history"]["stats"]["raw_count"], 3)
        self.assertEqual(len(combined["votes"]), 3)
        self.assertEqual(combined["stances"][sources[1]], "mixed")
        self.assertEqual(combined["votes"][sources[1]]["role"], "disagreement")
        self.assertEqual(len(combined["unit_votes"][sources[1]][sources[1] + ":0"]["merged_judgments"]), 2)
        self.assertEqual(combined["contexts"][context_id]["role"], "explicit_parent")
        self.assertEqual(combined["contexts"][context_id]["for_ids"], sources)
        self.assertEqual(combined["aliases"], [proof])
        self.assertEqual(len(combined["reviewed_release_associations"]), 1)
        self.assertEqual(combined["reviewed_release_associations"][0]["version"], "2.0")
        self.assertEqual(combined["engagement"]["possible_response_count"], 7)
        self.assertEqual(combined["matched_response_ids"], ["a:response", "b:response"])

    def test_failed_wording_rechecks_the_replacement_set_without_rechecking_unchanged_snapshots(self):
        original_refresh = pipeline._refresh
        original_check = pipeline.final_check

        def rank(concept, *args, **kwargs):
            refreshed = original_refresh(concept, *args, **kwargs)
            refreshed["score"] = 100 - topic_of(refreshed["definition"])
            return refreshed

        def reject_first(decisions, draft, state):
            check = original_check(decisions, draft, state)
            if topic_of(state["concern"]["definition"]) == 0:
                return {**check, "passed": False, "failed_claims": ["title"],
                        "checks": {**check["checks"], "title": 0.}}
            return check

        with patch.object(pipeline, "_refresh", side_effect=rank), \
                patch.object(pipeline, "final_check", side_effect=reject_first):
            result, diagnostics, _, _ = self.run_pipeline(
                self.write("failed-label-backfill.jsonl", fixture(topic_count=9)), draft_retries=0)
        self.assertEqual({trend["title_en"] for trend in result["trends"]},
                         {f"Concern {topic}" for topic in range(1, 9)})
        self.assertEqual(diagnostics["final_duplicate_review_calls"], 2)
        checks = diagnostics["final_duplicate_reviews"]
        self.assertNotEqual(checks[0]["concern_ids"], checks[1]["concern_ids"])
        self.assertEqual(len({check["snapshot"] for check in checks}), len(checks))
        selected = {entry["concern_id"] for entry in diagnostics["selected"]}
        self.assertEqual(selected, set(checks[-1]["concern_ids"]))

    def test_final_batch_resource_failure_preserves_prior_grounded_versions(self):
        original_review = pipeline.review_final_duplicates

        def batches(pool, reviewed, n, _ratio):
            candidates = {candidate["id"]: candidate for candidate in pool}
            order = ["cluster:127"] + [f"cluster:{topic}" for topic in range(9)]
            return [candidates[key] for key in order if key in candidates and key not in reviewed][:n]

        for exception, reason in ((pipeline.BudgetExceeded, "token_budget"),
                                  (pipeline.DecisionError, "decision_service_error")):
            with self.subTest(exception=exception.__name__):
                calls = []

                def fail_second(decisions, snapshots):
                    calls.append(deepcopy(snapshots))
                    if len(calls) == 2:
                        raise exception("Synthetic final review resource failure")
                    return original_review(decisions, snapshots)

                with patch.object(candidate_helpers, "choose_batch", side_effect=batches), \
                        patch.object(pipeline, "review_final_duplicates", side_effect=fail_second):
                    result, diagnostics, models, _ = self.run_pipeline(
                        self.write("final-budget.jsonl", fixture(topic_count=9)), review_batch=8)
                self.assertEqual(diagnostics["stop_reason"], reason)
                self.assertEqual(len(result["trends"]), 7)
                self.assertEqual(diagnostics["final_duplicate_review_calls"], 2)
                self.assertEqual(diagnostics["final_duplicate_reviews"][-1]["error"],
                                 "Synthetic final review resource failure")
                self.assertEqual(len([row for row in models.requests if row[0] == "chat"]), 7)

    def test_later_stronger_concerns_replace_earlier_choices_without_redrafting_unchanged_entries(self):
        path = self.write("reranked-backfill.jsonl", fixture(topic_count=9))
        original_refresh = pipeline._refresh
        original_check = pipeline.final_check

        def rank_concept(concept, *args, **kwargs):
            refreshed = original_refresh(concept, *args, **kwargs)
            topic = topic_of(refreshed["definition"])
            refreshed["score"] = 100 + topic if topic in {7, 8} else topic + 1
            return refreshed

        def batches(pool, reviewed, n, _ratio):
            by_id = {candidate["id"]: candidate for candidate in pool}
            order = ["cluster:127"] + [f"cluster:{topic}" for topic in range(9)]
            return [by_id[key] for key in order if key in by_id and key not in reviewed][:n]

        for rejected_draft in (None, 8):
            with self.subTest(rejected_draft=rejected_draft):
                def ground(decisions, draft, state):
                    check = original_check(decisions, draft, state)
                    if topic_of(state["concern"]["definition"]) == rejected_draft:
                        return {**check, "passed": False, "failed_claims": ["title"]}
                    return check

                with patch.object(pipeline, "_refresh", side_effect=rank_concept), \
                        patch.object(candidate_helpers, "choose_batch", side_effect=batches), \
                        patch.object(pipeline, "final_check", side_effect=ground):
                    result, diagnostics, models, reviewer = self.run_pipeline(path, review_batch=8, draft_retries=0)

                self.assertEqual(len(self.reviews(reviewer)), 10)
                expected = range(1, 9) if rejected_draft is None else range(8)
                self.assertEqual({trend["title_en"] for trend in result["trends"]},
                                 {f"Concern {topic}" for topic in expected})
                top = ["Concern 8", "Concern 7"] if rejected_draft is None else ["Concern 7", "Concern 6"]
                self.assertEqual([trend["title_en"] for trend in result["trends"][:2]], top)
                self.assertEqual(len(diagnostics["selected"]), 8)
                drafts = [request[2] for request in models.requests
                          if request[0] == "chat" and request[2]["task"] == "draft_output"]
                self.assertEqual(len(drafts), 9)
                self.assertEqual(sorted(writer_topic(draft) for draft in drafts), list(range(9)))
                grounded = [state for state, _ in reviewer.requests if state.get("task") == "ground_output"]
                self.assertEqual(len(grounded), 9)
                if rejected_draft is not None:
                    self.assertEqual(len(diagnostics["draft_failures"]), 1)

    def test_finalization_resource_failure_retains_unchanged_grounded_entries_without_more_calls(self):
        path = self.write("grounding-budget.jsonl", fixture(topic_count=9))
        original_refresh = pipeline._refresh
        original_check = pipeline.final_check
        original_state = pipeline._state_for

        def rank_concept(concept, *args, **kwargs):
            refreshed = original_refresh(concept, *args, **kwargs)
            topic = topic_of(refreshed["definition"])
            refreshed["score"] = 100 + topic if topic in {7, 8} else topic + 1
            return refreshed

        def batches(pool, reviewed, n, _ratio):
            by_id = {candidate["id"]: candidate for candidate in pool}
            order = ["cluster:127"] + [f"cluster:{topic}" for topic in range(9)]
            return [by_id[key] for key in order if key in by_id and key not in reviewed][:n]

        for exception, stop_reason in ((pipeline.BudgetExceeded, "token_budget"),
                                       (pipeline.DecisionError, "decision_service_error"),
                                       (LocalModelError, "local_model_error")):
            for changed_old_state in (False, True):
                with self.subTest(exception=exception.__name__, changed_old_state=changed_old_state):
                    failed = False

                    def state_for(concept, *args):
                        state = original_state(concept, *args)
                        if failed and changed_old_state and topic_of(concept["definition"]) == 0:
                            state["measurements"]["observation_revision"] = "changed"
                        return state

                    def ground(decisions, draft, state):
                        nonlocal failed
                        if topic_of(state["concern"]["definition"]) == 8:
                            failed = True
                            raise exception("Synthetic resource failure")
                        return original_check(decisions, draft, state)

                    class ResourceFailingModels(FakeModels):
                        def chat_json(self, system, payload, schema, max_tokens):
                            nonlocal failed
                            draft = super().chat_json(system, payload, schema, max_tokens)
                            if exception is LocalModelError and writer_topic(payload) == 8:
                                failed = True
                                raise LocalModelError("Synthetic local generation failure")
                            return draft

                    with patch.object(pipeline, "_refresh", side_effect=rank_concept), \
                            patch.object(candidate_helpers, "choose_batch", side_effect=batches), \
                            patch.object(pipeline, "_state_for", side_effect=state_for), \
                            patch(f"{__name__}.FakeModels", ResourceFailingModels), \
                            patch.object(pipeline, "final_check", side_effect=ground):
                        result, diagnostics, models, reviewer = self.run_pipeline(path, review_batch=8)

                    self.assertEqual(diagnostics["stop_reason"], stop_reason)
                    expected = range(1, 7) if changed_old_state else range(7)
                    self.assertEqual({trend["title_en"] for trend in result["trends"]},
                                     {f"Concern {topic}" for topic in expected})
                    drafts = [request[2] for request in models.requests
                              if request[0] == "chat" and request[2]["task"] == "draft_output"]
                    self.assertEqual(len(drafts), 8)
                    grounded = [state for state, _ in reviewer.requests if state.get("task") == "ground_output"]
                    self.assertEqual(len(grounded), 7)

    def test_unload_failures_are_diagnostic_and_do_not_erase_successful_results(self):
        class UnloadFailingModels(FakeModels):
            def unload(self, model):
                super().unload(model)
                raise LocalModelError("Synthetic unload failure")

        with patch(f"{__name__}.FakeModels", UnloadFailingModels):
            result, diagnostics, _, _ = self.run_pipeline(self.write("cleanup-failure.jsonl", fixture()))
        self.assertEqual(len(result["trends"]), 1)
        self.assertEqual(diagnostics["stop_reason"], "candidates_exhausted")
        self.assertEqual({error["model"] for error in diagnostics["local_cleanup_errors"]}, {"bge-m3", "qwen3:8b"})

    def test_embedding_failure_returns_diagnostics_without_clustering_partial_inputs_or_calling_jev(self):
        class EmbeddingFailingModels(FakeModels):
            def embed(self, texts):
                self.stats["embedding_calls"] += 1
                raise LocalModelError("Synthetic embedding failure")

            def unload(self, model):
                super().unload(model)
                raise LocalModelError("Synthetic unload failure")

        models = EmbeddingFailingModels()
        with patch.object(pipeline, "LocalModels", return_value=models), \
                patch.object(pipeline, "JevDecisions") as decisions, \
                patch.object(pipeline, "cluster_vectors") as clustering:
            result, diagnostics = pipeline.run(str(self.write("failed-embedding.jsonl", fixture())), AS_OF,
                                                cluster_cache=False)
        self.assertEqual(result["trends"], [])
        self.assertEqual(diagnostics["stop_reason"], "local_model_error")
        self.assertEqual(diagnostics["local_model_error"], "Synthetic embedding failure")
        self.assertEqual(diagnostics["local_cleanup_errors"][0]["message"], "Synthetic unload failure")
        decisions.assert_not_called()
        clustering.assert_not_called()

    def test_review_budget_is_configurable_above_one_hundred(self):
        rows = fixture(topic_count=105)
        result, _, _, reviewer = self.run_pipeline(self.write("review-budget.jsonl", rows),
                                                  rejected_topics=range(105), max_reviews=103)
        self.assertEqual(result["trends"], [])
        self.assertEqual(len(self.reviews(reviewer)), 103)

    def test_stable_discussion_stays_on_watchlist_despite_positive_semantic_review(self):
        result, diagnostics, models, reviewer = self.run_pipeline(self.write("stable.jsonl", fixture(stable=True)))
        self.assertEqual(result["trends"], [])
        self.assertEqual(models.stats["chat_calls"], 0)
        self.assertTrue(self.reviews(reviewer))
        self.assertTrue(diagnostics["watchlist"])

    def test_explicit_reactions_never_increase_substantive_issue_volume(self):
        rows = fixture()
        rows.append(record("parent", 12, "TOPIC_0 连接问题需要解决", hour=21))
        rows.append(record("agreement", 13, "+1", parent="parent", hour=21))
        rows.append(record("disagreement", 13, "不同意", parent="parent", hour=22))
        result, diagnostics, _, _ = self.run_pipeline(self.write("replies.jsonl", rows))
        selected = next(entry for entry in diagnostics["selected"] if entry["trend"]["title_en"] == "Concern 0")
        self.assertEqual(selected["history"]["stats"]["raw_count"], 36)
        self.assertTrue(selected["engagement"])
        self.assertEqual(selected["activity_rank"][0], -selected["history"]["score"])
        for trend in result["trends"]:
            self.assertFalse({"agreement", "disagreement"} & set(trend["voice_ids"]))

    def test_eight_accepted_topics_are_not_replaced_by_a_larger_provisional_topic(self):
        self._check_provisional_priority(accepted_count=8)

    def test_provisional_fills_only_an_exhausted_vacancy_and_stays_labeled(self):
        self._check_provisional_priority(accepted_count=7)

    def test_failed_accepted_wording_does_not_promote_provisional_quality(self):
        self._check_provisional_priority(accepted_count=8, failed_topic=1)

    def _check_provisional_priority(self, accepted_count, failed_topic=None):
        class QualityJev(FakeJev):
            def ask(self, state, questions):
                if state.get("task") == "review_sampled_candidate":
                    self.match_limit = 6 if topic_of(state.get("proposals")) == 0 else 10
                answers = super().ask(state, questions)
                if failed_topic is not None and state.get("task") == "ground_output":
                    topic = topic_of(questions["title"]["instructions"]["original_sources"])
                    if topic == failed_topic:
                        answers["title"]["noul"] = .1
                return answers

        real_refresh = pipeline._refresh

        def priority(concept, *args, **kwargs):
            refreshed = real_refresh(concept, *args, **kwargs)
            topic = topic_of(refreshed["definition"])
            refreshed["score"] = 10000 if topic == 0 else 100 - topic
            return refreshed

        with patch(f"{__name__}.FakeJev", QualityJev), patch.object(pipeline, "_refresh", side_effect=priority):
            result, diagnostics, models, _ = self.run_pipeline(
                self.write(f"quality-{accepted_count}-{failed_topic}.jsonl", fixture(topic_count=accepted_count + 1)),
                max_refinements=0)
        selected = diagnostics["selected"]
        topics = [topic_of(entry["evidence"]) for entry in selected]
        self.assertEqual(len(result["trends"]), 8)
        normal = [entry for entry in selected if entry["sample_quality"] == "accepted"]
        provisional = [entry for entry in selected if entry["sample_quality"] == "provisional"]
        if accepted_count == 8 and failed_topic is None:
            self.assertEqual(diagnostics["stop_reason"], "eight_supported_trends")
            self.assertEqual(set(topics), set(range(1, 9)))
            self.assertEqual(provisional, [])
            self.assertEqual(models.stats["chat_calls"], 8)
        else:
            self.assertEqual(diagnostics["stop_reason"], "candidates_exhausted")
            self.assertEqual(len(normal), 7)
            self.assertEqual(topics[-1], 0)
            self.assertEqual(len(provisional), 1)
            self.assertEqual(provisional[0]["sample_review"]["quality"], "provisional")
            self.assertEqual(provisional[0]["sample_review"]["match_ratio"], .6)
            self.assertIn("provisional topic interpretation", provisional[0]["trend"]["why"])
            if failed_topic is not None:
                self.assertNotIn(failed_topic, topics)
        self.assertTrue(all(entry["sample_review"]["match_ratio"] >= .8 for entry in normal))

    def test_approved_alias_proof_reaches_final_models_without_becoming_issue_support(self):
        rows = fixture()
        for row in rows:
            if topic_of(row["content"]) == 0:
                row["content"] += " 星舟的操作体验"
        proof = record("alias-proof", 0, "BACKGROUND 资料说明，凌月＝星舟。", hour=1)
        parent = record("ordinary-parent", 12, "BACKGROUND Earlier discussion context.", hour=1)
        rows.extend([proof, parent])
        target = next(row for row in rows if row["id"] == "topic-0-13-3")
        target["content"] = "TOPIC_0 星舟反馈"
        target["parent_id"] = parent["id"]
        reply = record("ordinary-reply", 13, "+1", parent=target["id"])
        reply["created_at"] = reply["created_at"].replace(":30:", ":31:")
        rows.append(reply)
        rows.append(record("future-alias", 14, "BACKGROUND 未来源＝星舟。FUTURE_ALIAS_SECRET", hour=1))

        class AliasReferencingModels(FakeModels):
            def chat_json(self, system, payload, schema, max_tokens):
                draft = super().chat_json(system, payload, schema, max_tokens)
                if payload["task"] == "draft_output":
                    draft["title"] = "凌月的操作体验反馈"
                return draft

        with patch(f"{__name__}.FakeModels", AliasReferencingModels):
            result, diagnostics, models, reviewer = self.run_pipeline(self.write("alias-handoff.jsonl", rows))
        self.assertEqual(len(result["trends"]), 1)
        self.assertEqual(diagnostics["selected"][0]["history"]["stats"]["raw_count"], 35)
        self.assertFalse({"alias-proof", "ordinary-parent", "ordinary-reply", "future-alias"}
                         & set(result["trends"][0]["voice_ids"]))
        draft_states = [request[2]["evidence"] for request in models.requests
                        if request[0] == "chat" and request[2]["task"] == "draft_output"]
        ground_material = [questions["title"]["instructions"] for state, questions in reviewer.requests
                           if state.get("task") == "ground_output"]
        self.assertEqual(len(draft_states), 1)
        self.assertEqual(len(ground_material), 1)
        self.assertTrue(all(re.fullmatch(r"s\d+", row["id"])
                            for row in draft_states[0]["sources"] + draft_states[0]["context"]))
        state = restore_writer_ids(draft_states[0], {
            "sources": ground_material[0]["original_sources"],
            "context": ground_material[0]["interpretation_context"],
        })
        for original, projected in zip(ground_material[0]["original_sources"], state["sources"], strict=True):
            self.assertEqual(projected, original)
            self.assertTrue(projected["reviewed_focal_texts"])
        self.assertEqual(ground_material[0]["interpretation_context"], state["context"])
        self.assertEqual(ground_material[0]["aliases"], state["aliases"])
        self.assertEqual(ground_material[0]["as_of"], state["as_of"])
        selected = diagnostics["selected"][0]
        self.assertNotIn("sampled_stances", state)
        self.assertNotIn("measurements", state)
        grounded_activity = next(questions["action_valid"]["instructions"] for source, questions in reviewer.requests
                                 if source.get("task") == "ground_output")
        self.assertNotIn("daily_activity", grounded_activity["measurements"])
        self.assertNotIn("platform_activity", grounded_activity["measurements"])
        self.assertEqual(grounded_activity["measurements"]["sampled_stances"], {"negative": 10})
        self.assertTrue(grounded_activity["measurements"]["platform_history"])
        self.assertTrue(selected["history"]["stats"]["daily"])
        self.assertTrue(selected["history"]["stats"]["platform_windows"])
        self.assertTrue(diagnostics["audits"][0]["packet"]["stats"]["daily_activity"])
        self.assertTrue(diagnostics["audits"][0]["packet"]["stats"]["platform_activity"])
        self.assertEqual(selected["aliases"], state["aliases"])
        self.assertEqual(selected["context"], state["context"])
        self.assertEqual(selected["draft_claims"][0]["evidence_ids"], result["trends"][0]["voice_ids"])
        self.assertNotIn("alias-proof", selected["draft_claims"][0]["evidence_ids"])
        self.assertEqual(selected["proposed_action"], "Review the supplied player feedback.")
        self.assertEqual(result["trends"][0]["why"],
                         result["trends"][0]["title_en"] + ". "
                         + pipeline._public_status_basis({"history": selected["history"]})
                         + " Suggested next step: " + selected["proposed_action"])
        self.assertNotIn("concern", state)
        public_concern = ground_material[0]["approved_concern"]
        self.assertEqual(public_concern["proposal_kind"], "source_anchor")
        self.assertTrue(public_concern["source_anchor_id"].startswith("topic-0-"))
        self.assertIn("TOPIC_0", public_concern["focal_text"])
        approved = next(request[2]["approved_concern"] for request in models.requests
                        if request[0] == "chat" and request[2]["task"] == "draft_output")
        for field in ("focal_text", "definition", "source_anchor_id", "source_anchor_index", "id", "proposal_kind"):
            self.assertNotIn(field, approved)
        self.assertNotIn("require semantic review", approved["uncertainty"])
        self.assertIn("reviewed samples", approved["uncertainty"])
        self.assertEqual(state["aliases"][0]["name"], "凌月")
        self.assertEqual(state["aliases"][0]["evidence_ids"], ["alias-proof"])
        context = {row["id"]: row for row in state["context"]}
        self.assertEqual(context["alias-proof"]["content"], proof["content"])
        self.assertEqual(context["alias-proof"]["created_at"], proof["created_at"])
        self.assertTrue(context["alias-proof"]["alias_provenance"])
        self.assertTrue({"ordinary-parent", "ordinary-reply"} <= set(context))
        self.assertLessEqual(len(context), 20)
        self.assertFalse(any("凌月" in (row.get("content") or "") for row in selected["evidence"]))
        self.assertNotIn("FUTURE_ALIAS_SECRET", json.dumps(models.requests + reviewer.requests))

    def test_final_context_keeps_associated_context_for_long_and_short_selected_sources(self):
        rows = [record("long", 3, "A detailed substantive experience " + "with supporting details " * 12),
                record("short", 3, "I got lost yesterday"),
                record("long-parent", 2, "Another discussion subject"),
                record("map-context", 2, "We are discussing the new area"),
                record("unverified-reply", 3, "An unrelated response"),
                record("verified-reply", 3, "A reviewed relevant response"),
                record("alias-proof", 0, "A source-backed nickname assertion")]
        by_id = {row["id"]: {**row, "day": row["created_at"][:10]} for row in rows}
        context = {
            "long-parent": {"id": "long-parent", "text": "Another discussion subject",
                            "role": "explicit_parent", "for_ids": ["long"]},
            "map-context": {"id": "map-context", "text": "We are discussing the new area",
                            "role": "possible_preceding_context", "for_ids": ["short"]},
            "unverified-reply": {"id": "unverified-reply", "text": "An unrelated response",
                                 "role": "explicit_response", "for_ids": ["long"]},
            "verified-reply": {"id": "verified-reply", "text": "A reviewed relevant response",
                               "role": "possible_following_response", "for_ids": ["long"]},
        }
        concept = {"id": "concern", "definition": "A source-anchored topic", "scope": "Supplied source topic",
                   "uncertainty": "Sampled interpretation", "match_ratio": 1., "matched_ids": ["long", "short"],
                   "opposing_ids": [], "matched_response_ids": ["verified-reply"], "contexts": context,
                   "history": {"stats": {"raw_count": 2}, "status": "new", "why_en": "Observed onset"},
                   "first_seen": "2026-07-04", "engagement": {}, "stances": {},
                   "aliases": [{"name": "Canonical", "alias": "Nickname", "source_id": "alias-proof",
                                "evidence_ids": ["alias-proof"], "caveat": "A contextual assertion"}]}
        calendar = [{"date": "2026-07-01", "version": "v1"}, {"date": "2026-07-10", "version": "v2"}]
        state = pipeline._state_for(concept, by_id, calendar, AS_OF)
        self.assertEqual(state["as_of"], AS_OF)
        self.assertEqual(state["past_release_calendar"], [])
        self.assertEqual(state["source_release_associations"], [])
        concept["history"]["stats"]["as_of"] = AS_OF
        self.assertEqual(pipeline._state_for(concept, by_id, calendar)["as_of"], AS_OF)
        self.assertEqual({row["id"] for row in state["context"]},
                          {"alias-proof", "long-parent", "map-context", "verified-reply"})
        self.assertEqual(state["context"][0]["id"], "alias-proof")
        self.assertEqual({row["id"] for row in state["sources"]}, {"long", "short"})
        self.assertEqual(state["measurements"]["raw_count"], 2)

    def test_final_sources_keep_confirmed_anchor_dissent_and_time_platform_diversity(self):
        rows = [record(f"source-{index:02d}", index, f"Original source content {index}") for index in range(30)]
        for index, row in enumerate(rows):
            row["platform"] = {10: "QQ", 18: "Chat", 23: "Video"}.get(index, "Forum")
            row["user_name"] = f"author-{index % 4}"
        by_id = {row["id"]: {**row, "day": row["created_at"][:10]} for row in rows}
        concept = {"id": "concern", "matched_ids": list(by_id), "opposing_ids": ["source-06"],
                   "source_anchor_id": "source-12", "first_seen": "2026-07-01", "engagement": {},
                   "history": {"stats": {}, "status": "growing", "why_en": "Computed activity"}}
        state = pipeline._state_for(concept, by_id, [], "2026-07-30")
        sources = state["sources"]
        identifiers = [source["id"] for source in sources]
        self.assertEqual(len(identifiers), 10)
        self.assertEqual(len(set(identifiers)), 10)
        self.assertTrue({"source-00", "source-29", "source-12", "source-06"} <= set(identifiers))
        self.assertEqual({source["platform"] for source in sources}, {"Forum", "QQ", "Chat", "Video"})
        day_numbers = [date.fromisoformat(source["day"]).toordinal() for source in sources]
        self.assertLessEqual(max(right - left for left, right in zip(day_numbers, day_numbers[1:])), 7)
        self.assertGreater(len({source["user_name"] for source in sources}), 1)
        self.assertEqual(sources[0]["day"], state["first_seen"])
        self.assertEqual(pipeline._state_for(concept, by_id, [], "2026-07-30")["sources"], sources)
        concept["matched_ids"].remove("source-12")
        self.assertNotIn("source-12", {source["id"] for source in pipeline._state_for(concept, by_id, [], "2026-07-30")["sources"]})

    def test_final_source_focal_provenance_excludes_other_and_uncertain_units(self):
        source = record("source", 0, "A reviewed concern; A separate subject; An unclear fragment")
        source["title"] = "The unchanged original title"
        by_id = {"source": {**source, "day": "2026-07-01"}}
        concept = {"id": "concern", "matched_ids": ["source"], "opposing_ids": [],
                   "first_seen": "2026-07-01", "engagement": {},
                   "history": {"stats": {}, "status": "new", "why_en": "Computed activity"}}
        targets = [
            {"id": "source", "unit_id": "source:0", "text": "A reviewed concern"},
            {"id": "source", "unit_id": "source:1", "text": "A separate subject"},
            {"id": "source", "unit_id": "source:2", "text": "An unclear fragment"},
        ]
        for target, match in zip(targets, ("match", "other", "unclear")):
            pipeline._record_unit_votes(concept, [target], {
                "contributions": {"source": {"match": match, "role": "experience"}},
                "stances": {"source": "negative"}})
        before = deepcopy(by_id)
        evidence = pipeline._state_for(concept, by_id, [], AS_OF)["sources"][0]
        self.assertEqual(evidence["reviewed_focal_texts"], ["A reviewed concern"])
        self.assertEqual(evidence["content"], source["content"])
        self.assertEqual(evidence["title"], source["title"])
        self.assertEqual(by_id, before)

    def test_known_topic_examples_are_current_source_deduplicated_bounded_and_cutoff_safe(self):
        rows = [record(f"known-{index}", index, f"Original full content and qualifier {index}") for index in range(6)]
        rows += [record("future", 14, "Future original"), record("other", 1, "An unrelated source"),
                 record("unclear", 2, "An unresolved source")]
        by_id = {row["id"]: {**row, "day": row["created_at"][:10]} for row in rows}
        by_id["future"]["day"] = "2026-07-01"  # The actual timestamp controls cutoff eligibility.
        ledger = {identifier: {identifier + ":0": {"match": "match", "text": f"Reviewed focal {identifier}"}}
                  for identifier in [*by_id, "missing"]}
        ledger["known-0"].update({"known-0:1": {"match": "match", "text": "An opposing assessment of the same issue"},
                                  "known-0:2": {"match": "other", "text": "Unrelated stale unit"},
                                  "known-0:3": {"match": "unclear", "text": "Unresolved unit"}})
        ledger["other"]["other:0"]["match"] = "other"
        ledger["unclear"]["unclear:0"]["match"] = "unclear"
        concept = {"id": "existing", "definition": "Original illustrative seed", "matched_ids": [*by_id, "missing", "known-0"],
                   "source_anchor_id": "known-3", "opposing_ids": [], "unit_votes": ledger}
        before = deepcopy((concept, by_id))
        represented = pipeline._accepted_for_review([concept], by_id, AS_OF)
        examples = represented[0]["reviewed_examples"]
        self.assertEqual([source["id"] for source in examples], ["known-0", "known-3", "known-5"])
        self.assertEqual(examples[0]["reviewed_focal_texts"],
                         ["Reviewed focal known-0", "An opposing assessment of the same issue"])
        self.assertEqual(examples[0]["content"], by_id["known-0"]["content"])
        self.assertEqual(examples[0]["created_at"], by_id["known-0"]["created_at"])
        self.assertEqual(examples[0]["platform"], by_id["known-0"]["platform"])
        self.assertEqual(represented[0]["definition"], concept["definition"])
        self.assertNotIn("Unrelated stale unit", json.dumps(examples))
        self.assertNotIn("Unresolved unit", json.dumps(examples))
        self.assertEqual(pipeline._accepted_for_review([concept], by_id, AS_OF), represented)
        examples[0]["reviewed_focal_texts"].append("A mutated provider copy")
        examples[0]["content"] = "A mutated original copy"
        self.assertEqual((concept, by_id), before)

    def test_existing_topic_examples_allow_paraphrase_merge_without_merging_different_matchmaking_issues(self):
        for same_issue in (True, False):
            with self.subTest(same_issue=same_issue):
                rows = fixture(topic_count=2)
                for row in rows:
                    topic = topic_of(row["content"])
                    if topic == 0:
                        row["content"] = "TOPIC_0 I repeatedly face opponents with a much higher rank."
                    elif topic == 1:
                        row["content"] = ("TOPIC_1 As a novice I keep facing expert opponents." if same_issue else
                                          "TOPIC_1 I cannot find a match despite waiting a long time.")

                class RepresentedIssueJev(FakeJev):
                    def ask(self, state, questions):
                        answers = super().ask(state, questions)
                        if state.get("task") != "review_sampled_candidate" or topic_of(state["proposals"]) != 1:
                            return answers
                        existing = next((item for item in state["accepted_concerns"]
                                         if topic_of(item["definition"]) == 0), None)
                        if existing:
                            examples = existing["reviewed_examples"]
                            self.assert_examples = examples
                            if same_issue and examples and all(source["reviewed_focal_texts"] for source in examples):
                                for name in questions:
                                    if name.endswith("_duplicate"):
                                        answers[name] = confident(existing["reference"])
                        return answers

                with patch(f"{__name__}.FakeJev", RepresentedIssueJev):
                    result, diagnostics, models, reviewer = self.run_pipeline(
                        self.write(f"represented-paraphrase-{same_issue}.jsonl", rows))
                self.assertEqual(len(result["trends"]), 1 if same_issue else 2)
                self.assertTrue(reviewer.assert_examples)
                self.assertLessEqual(len(reviewer.assert_examples), 3)
                self.assertTrue(all("higher rank" in source["content"] for source in reviewer.assert_examples))
                chats = [request for request in models.requests if request[0] == "chat"]
                self.assertEqual(len(chats), len(result["trends"]))
                self.assertEqual(reviewer.stats["api_calls"], len(self.reviews(reviewer)) + len(chats) + diagnostics["final_duplicate_review_calls"])
                if same_issue:
                    self.assertEqual(diagnostics["selected"][0]["history"]["stats"]["raw_count"], 70)

    def test_reviewed_event_associations_override_literal_version_fallback(self):
        source = record("source", 3, "Players discuss the celebration and the newly named arrival, including version 2.0.")
        by_id = {"source": {**source, "day": "2026-07-04"}}
        concept = {"id": "concern", "matched_ids": ["source"], "opposing_ids": [],
                   "history": {"stats": {"as_of": AS_OF}, "status": "new", "why_en": "Observed onset"},
                   "first_seen": "2026-07-04", "engagement": {}, "stances": {}}
        celebration = {"date": "2026-07-01", "version": "-", "notes": "Celebration starts"}
        arrival = {"date": "2026-07-10", "version": "2.0", "notes": "A named arrival"}
        future = {"date": "2026-07-15", "version": "future", "notes": "Future event"}
        calendar = [celebration, arrival, future]
        legacy = pipeline._state_for(concept, by_id, calendar, AS_OF)
        self.assertEqual([row["version"] for row in legacy["source_release_associations"]], ["2.0"])
        concept["reviewed_release_associations"] = []
        self.assertEqual(pipeline._state_for(concept, by_id, calendar, AS_OF)["past_release_calendar"], [])
        concept["reviewed_release_associations"] = [
            {**celebration, "association": "content_association_only"}, arrival, celebration, future,
            {**arrival, "notes": "Altered notes are not the approved calendar row"},
            {"date": "2026-07-04", "version": "unknown", "notes": "Not in the calendar"},
        ]
        state = pipeline._state_for(concept, by_id, calendar, AS_OF)
        self.assertEqual([row["version"] for row in state["source_release_associations"]], ["-", "2.0"])
        self.assertEqual(state["past_release_calendar"], state["source_release_associations"])
        self.assertTrue(all(row["association"] == "content_association_only" for row in state["past_release_calendar"]))
        self.assertTrue(all("does not establish causation" in row["relationship"] for row in state["past_release_calendar"]))

    def test_merged_reviews_union_supported_named_events_without_changing_topic_counts(self):
        rows = fixture()
        for row in rows:
            if topic_of(row["content"]) == 0:
                row["content"] = "TOPIC_0 关于节庆和新角色的体验反馈 " + row["id"]
        celebration = {"date": "2026-07-01", "version": "-", "notes": "Celebration starts"}
        arrival = {"date": "2026-07-10", "version": "2.0", "notes": "A named arrival"}
        future = {"date": "2026-07-15", "version": "future", "notes": "Future event"}
        calendar = self.directory / "named-events.csv"
        calendar.write_text("date,version,notes\n2026-07-01,-,Celebration starts\n"
                            "2026-07-10,2.0,A named arrival\n2026-07-15,future,Future event\n", encoding="utf-8")
        real_pool, real_review = candidate_helpers.build_candidate_pool, pipeline.review_candidate
        reviews = 0

        def repeated_candidate(*args, **kwargs):
            first = next(candidate for candidate in real_pool(*args, **kwargs)
                         if candidate["kind"] == "cluster" and int(np.argmax(candidate["center"])) == 0)
            repeated = deepcopy(first)
            repeated["id"] += ":another-view"
            return [first, repeated]

        def review_events(*args, **kwargs):
            nonlocal reviews
            result = real_review(*args, **kwargs)
            reviews += 1
            for concern in result["concerns"]:
                concern["reviewed_release_associations"] = ([celebration] if reviews == 1 else [
                    arrival, future, {**arrival, "notes": "Unrecognized changed row"}])
            return result

        with patch.object(candidate_helpers, "build_candidate_pool", side_effect=repeated_candidate), \
                patch.object(pipeline, "review_candidate", side_effect=review_events):
            result, diagnostics, models, _ = self.run_pipeline(
                self.write("named-event-reviews.jsonl", rows), calendar, review_batch=1)
        self.assertEqual(reviews, 2)
        self.assertEqual(len(result["trends"]), 1)
        selected = diagnostics["selected"][0]
        self.assertEqual(selected["history"]["stats"]["raw_count"], 35)
        self.assertEqual([row["version"] for row in selected["release_associations"]], ["-", "2.0"])
        final_state = [request[2]["evidence"] for request in models.requests if request[0] == "chat"][-1]
        self.assertEqual(final_state["source_release_associations"], selected["release_associations"])
        self.assertNotIn("past_release_calendar", final_state)
        self.assertNotIn("version 2.0", json.dumps(final_state["sources"]))
        self.assertNotIn("Future event", json.dumps(final_state))

    def test_neighbor_review_recomputes_opposing_branches_before_drafting_without_automatic_merges(self):
        for same_concern, runtime_budget in ((True, None), (False, None), (True, 9)):
            with self.subTest(same_concern=same_concern, runtime_budget=runtime_budget):
                rows = []
                for day in range(14):
                    daily = []
                    for topic in range(8):
                        count = (4 if day < 7 else 1) if topic == 0 else (1 if day < 7 else 4)
                        daily.extend(record(f"topic-{topic}-{day}-{index}", day,
                                            f"TOPIC_{topic} Independent source experience {day}-{index}")
                                     for index in range(count))
                    neighbor_topic = 0 if same_concern else 8
                    daily.extend(record(f"neighbor-{day}-{index}", day,
                                        f"TOPIC_{neighbor_topic} NEIGHBOR_BRANCH Another source experience {day}-{index}")
                                 for index in range(1 if day < 7 else 4))
                    # Hold the observed community denominator constant so the
                    # opposite branches together are a stable topic.
                    daily.extend(record(f"background-{day}-{index}", day,
                                        f"BACKGROUND Other daily discussion {day}-{index}")
                                 for index in range(200 - len(daily)))
                    rows.extend(daily)
                events = []

                class NeighborModels(FakeModels):
                    def embed(self, texts):
                        vectors = super().embed(texts)
                        for index, text in enumerate(texts):
                            if "NEIGHBOR_BRANCH" in text:
                                vectors[index] = 0
                                vectors[index, 0], vectors[index, 20] = .995, .1
                                vectors[index] /= np.linalg.norm(vectors[index])
                        return vectors

                    def chat_json(self, system, payload, schema, max_tokens):
                        events.append(("draft", writer_topic(payload)))
                        return super().chat_json(system, payload, schema, max_tokens)

                class NeighborJev(FakeJev):
                    def ask(self, state, questions):
                        if state.get("task") == "review_sampled_candidate":
                            events.append(("review", state["candidate"]["candidate_id"]))
                        return super().ask(state, questions)

                def split_close_vectors(vectors, target_size):
                    labels = np.argmax(vectors, axis=1)
                    centers = np.eye(DIMENSION, dtype=np.float32)
                    labels[vectors[:, 20] > .01] = 20
                    centers[20] = 0
                    centers[20, 0], centers[20, 20] = .995, .1
                    centers[20] /= np.linalg.norm(centers[20])
                    return labels, centers

                def batches(pool, reviewed, n, ratio):
                    by_id = {candidate["id"]: candidate for candidate in pool}
                    order = [f"cluster:{topic}" for topic in range(8)] + ["cluster:127", "cluster:20"]
                    return [by_id[key] for key in order if key in by_id and key not in reviewed][:n]

                with patch(f"{__name__}.FakeModels", NeighborModels), \
                        patch(f"{__name__}.FakeJev", NeighborJev), \
                        patch.object(candidate_helpers, "choose_batch", side_effect=batches), \
                        patch(f"{__name__}.fake_clusters", side_effect=split_close_vectors):
                    result, diagnostics, models, _ = self.run_pipeline(
                        self.write(f"neighbors-{same_concern}.jsonl", rows), review_batch=8,
                        neighbor_reviews=16 if runtime_budget else 1, max_refinements=0,
                        max_reviews=runtime_budget or 30)
                neighbor_audit = next(audit for audit in diagnostics["audits"] if audit["stage"] == "neighbor")
                self.assertEqual(neighbor_audit["candidate"], "cluster:20")
                self.assertTrue(neighbor_audit["neighbor_context"]["opposing_direction"])
                self.assertGreater(neighbor_audit["neighbor_context"]["cosine"], .9)
                self.assertEqual(diagnostics["neighbor_reconciliation"]["reviews"], 1)
                if runtime_budget:
                    self.assertEqual(diagnostics["review_attempts"], runtime_budget)
                    self.assertEqual(len(diagnostics["audits"]), runtime_budget)
                    self.assertEqual(diagnostics["stop_reason"], "runtime_review_limit")
                    self.assertEqual(diagnostics["neighbor_reconciliation"]["status"], "runtime_review_limit")
                self.assertLess(events.index(("review", "cluster:20")),
                                next(index for index, event in enumerate(events) if event[0] == "draft"))
                histories = {int(entry["trend"]["title_en"].split()[-1]): entry["history"]
                             for entry in diagnostics["selected"]}
                histories.update({topic_of(entry["definition"]): entry["history"]
                                  for entry in diagnostics["watchlist"]})
                if same_concern:
                    self.assertEqual(histories[0]["stats"]["raw_count"], 70)
                    self.assertIsNone(histories[0]["status"])
                    self.assertNotIn("Concern 0", {trend["title_en"] for trend in result["trends"]})
                    self.assertTrue(all(topic != 0 for kind, topic in events if kind == "draft"))
                else:
                    self.assertEqual(histories[0]["stats"]["raw_count"], 35)
                    self.assertEqual(histories[0]["status"], "fading")
                    self.assertEqual(histories[8]["stats"]["raw_count"], 35)
                    self.assertEqual(histories[8]["status"], "growing")

    def test_repair_receives_failed_draft_checks_and_bound_sources_without_mutating_originals(self):
        original_state, original_check, original_draft = pipeline._state_for, pipeline.final_check, pipeline.final_draft
        baseline_states, checked_states, previous, pipeline_draft_states, requirements = [], [], [], [], []

        def capture_state(*args, **kwargs):
            state = original_state(*args, **kwargs)
            baseline_states.append(state)
            return state

        def fail_first_title(decisions, draft, state):
            checked_states.append(deepcopy(state))
            check = original_check(decisions, draft, state)
            if len(checked_states) == 1:
                previous.append(deepcopy(draft))
                question = decisions.requests[-1][1]["title_en"]
                policy = {"title_en": {"task": question["instructions"]["task"], "criteria": question.get("criteria", {})}}
                requirements.append(deepcopy(policy))
                return {**check, "passed": False, "failed_claims": ["title_en"],
                        "checks": {**check["checks"], "title_en": .2}, "failure_requirements": policy}
            return check

        def capture_draft(local, approved, state):
            pipeline_draft_states.append(deepcopy(state))
            return original_draft(local, approved, state)

        class RepairModels(FakeModels):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.returned = []

            def chat_json(self, system, payload, schema, max_tokens):
                draft = super().chat_json(system, payload, schema, max_tokens)
                if not self.returned:
                    draft["title_en"] = "All players experienced this concern."
                else:
                    # A provider must not gain mutable access to baseline
                    # evidence or the object it returned on an earlier call.
                    payload["evidence"]["previous_draft_check"]["checks"]["title_en"] = 999
                    payload["evidence"]["previous_draft_check"]["failure_requirements"]["title_en"]["task"] = "Mutated retry policy copy"
                    payload["evidence"]["sources"][0]["content"] = "Mutated retry source copy"
                self.returned.append(draft)
                return draft

        with patch(f"{__name__}.FakeModels", RepairModels), \
                patch.object(pipeline, "_state_for", side_effect=capture_state), \
                patch.object(pipeline, "final_draft", side_effect=capture_draft), \
                patch.object(pipeline, "extractive_fallback", return_value=None), \
                patch.object(pipeline, "final_check", side_effect=fail_first_title):
            result, diagnostics, models, _ = self.run_pipeline(self.write("repair-handoff.jsonl", fixture()),
                                                              draft_retries=1)
        self.assertEqual(len(result["trends"]), 1)
        drafts = [restore_writer_ids(request[2]["evidence"], checked_states[index])
                  for index, request in enumerate(request for request in models.requests if request[0] == "chat")]
        self.assertEqual(len(drafts), 2)
        writer_payloads = [request[2] for request in models.requests if request[0] == "chat"]
        self.assertEqual(writer_payloads[0]["task"], "draft_output")
        self.assertNotIn("repair_instruction", writer_payloads[0])
        self.assertEqual(writer_payloads[1]["task"], "repair_output")
        self.assertIn("failure_requirements", writer_payloads[1]["repair_instruction"])
        self.assertIn("not an approved template", writer_payloads[1]["repair_instruction"])
        self.assertNotIn("previous_draft", drafts[0])
        self.assertNotIn("previous_draft_check", drafts[0])
        self.assertNotIn("previous_draft", drafts[1])
        self.assertNotIn(previous[0]["title_en"], json.dumps(drafts[1]))
        self.assertEqual(pipeline_draft_states[1]["previous_draft"], previous[0])
        self.assertEqual(drafts[1]["previous_draft_check"]["failed_claims"], ["title_en"])
        self.assertEqual(drafts[1]["previous_draft_check"]["checks"]["title_en"], .2)
        self.assertEqual(drafts[1]["previous_draft_check"]["failure_requirements"], requirements[0])
        self.assertEqual(pipeline_draft_states[1]["previous_draft_check"]["failure_requirements"], requirements[0])
        self.assertNotEqual(requirements[0]["title_en"]["task"], "Mutated retry policy copy")
        self.assertNotIn("failed_claim_material", drafts[1]["previous_draft_check"])
        self.assertNotIn("failed_claim_material", pipeline_draft_states[1]["previous_draft_check"])
        self.assertEqual(models.returned[0]["title_en"], "All players experienced this concern.")
        self.assertEqual(set(models.returned[0]), {"title", "title_en", "proposed_action"})
        self.assertTrue(all(source["content"] != "Mutated retry source copy"
                            for state in baseline_states + checked_states for source in state["sources"]))
        self.assertTrue(all("previous_draft" not in state and "previous_draft_check" not in state
                            and state["measurements"]["raw_count"] == 35 for state in baseline_states + checked_states))
        self.assertEqual(diagnostics["selected"][0]["history"]["stats"]["raw_count"], 35)
        _, _, control_models, _ = self.run_pipeline(self.directory / "repair-handoff.jsonl", draft_retries=0)
        first_request = next(request for request in models.requests if request[0] == "chat")
        control_request = next(request for request in control_models.requests if request[0] == "chat")
        self.assertEqual(first_request, control_request)

    def test_component_fallback_reuses_checked_fields_without_more_calls_or_failed_public_prose(self):
        original_check = pipeline.final_check
        for failed_field in ("action_valid", "title", "title_en"):
            with self.subTest(failed_field=failed_field):
                rejected = []

                def fail_component(decisions, draft, state):
                    check = original_check(decisions, draft, state)
                    check = {**check, "passed": False, "failed_claims": [failed_field],
                             "checks": {**check["checks"], failed_field: .2}}
                    rejected.append({"draft": deepcopy(draft), "check": deepcopy(check)})
                    return check

                with patch.object(pipeline, "final_check", side_effect=fail_component):
                    result, diagnostics, models, reviewer = self.run_pipeline(
                        self.write(f"component-fallback-{failed_field}.jsonl", fixture()), draft_retries=1)
                calls = [request for request in models.requests if request[0] == "chat"]
                reviews = self.reviews(reviewer)
                if failed_field in {"title", "title_en"}:
                    self.assertEqual(result["trends"], [])
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(reviewer.stats["api_calls"], len(reviews) + 2)
                    failure = diagnostics["draft_failures"][0]
                    self.assertEqual(len(failure["attempts"]), 2)
                    self.assertEqual(failure["attempts"][0]["draft"], rejected[0]["draft"])
                    self.assertEqual(failure["attempts"][0]["check"], rejected[0]["check"])
                    self.assertTrue(failure["evidence_ids"])
                    self.assertTrue(failure["version"])
                    self.assertTrue(diagnostics["watchlist"])
                    continue
                self.assertEqual(len(result["trends"]), 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(reviewer.stats["api_calls"], len(reviews) + 1)
                selected = diagnostics["selected"][0]
                self.assertEqual(selected["rejected_generation"], rejected[0])
                self.assertEqual(selected["grounding"]["rejected_generation"], rejected[0])
                self.assertEqual(set(result["trends"][0]), {
                    "title", "title_en", "category", "why", "status", "first_seen", "voice_ids"})
                self.assertNotIn(failed_field, selected["grounding"]["checks"])
                self.assertEqual(selected["generation_mode"], "checked_topic_summary")
                self.assertEqual(result["trends"][0]["category"], result["trends"][0]["title_en"])
                self.assertIn(result["trends"][0]["title_en"], result["trends"][0]["why"])
                self.assertEqual(selected["components"]["proposed_action"], "omitted_failed_component")
                self.assertIsNone(selected["proposed_action"])
                self.assertNotIn(rejected[0]["draft"]["proposed_action"], result["trends"][0]["why"])
                self.assertNotIn("Suggested next step:", result["trends"][0]["why"])

    def test_invalid_generation_skips_its_version_and_preserves_other_grounded_concerns(self):
        original_refresh = pipeline._refresh

        def rank(concept, *args, **kwargs):
            refreshed = original_refresh(concept, *args, **kwargs)
            refreshed["score"] = 100 - topic_of(concept["definition"])
            return refreshed

        class OneInvalidModels(FakeModels):
            def chat_json(self, system, payload, schema, max_tokens):
                draft = super().chat_json(system, payload, schema, max_tokens)
                if writer_topic(payload) == 1:
                    raise LocalGenerationError("Synthetic exhausted schema failure: category is missing")
                return draft

        with patch(f"{__name__}.FakeModels", OneInvalidModels), \
                patch.object(pipeline, "_refresh", side_effect=rank):
            result, diagnostics, models, reviewer = self.run_pipeline(
                self.write("one-invalid-generation.jsonl", fixture(topic_count=3)))
        self.assertEqual([trend["title_en"] for trend in result["trends"]], ["Concern 0", "Concern 2"])
        self.assertEqual(diagnostics["stop_reason"], "candidates_exhausted")
        self.assertNotIn("local_model_error", diagnostics)
        failures = diagnostics["generation_failures"]
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0]["version"])
        self.assertIn("category", failures[0]["reason"])
        self.assertNotIn(failures[0]["concern_id"], {entry["concern_id"] for entry in diagnostics["selected"]})
        calls = [request for request in models.requests if request[0] == "chat"]
        self.assertEqual([writer_topic(request[2]) for request in calls], [0, 1, 2])
        self.assertEqual(reviewer.stats["api_calls"], len(self.reviews(reviewer)) + 2 + diagnostics["final_duplicate_review_calls"])

    def test_two_of_ten_matches_are_rejected_but_eight_with_opposition_are_accepted(self):
        path = self.write("contamination.jsonl", fixture(large=True))
        weak, _, _, weak_reviewer = self.run_pipeline(path, match_limit=2)
        self.assertEqual(weak["trends"], [])
        self.assertTrue(any(len(state["candidate"]["targets"]) == 10 for state in self.reviews(weak_reviewer)))
        strong, diagnostics, _, _ = self.run_pipeline(path, match_limit=8, opposing_index=1)
        self.assertEqual(len(strong["trends"]), 1)
        reviewed = diagnostics["selected"][0]["sample_review"]
        self.assertEqual(reviewed["match_ratio"], .8)
        self.assertEqual(reviewed["quality"], "accepted")
        matching = {identifier for identifier, vote in reviewed["votes"].items() if vote["match"] == "match"}
        opposing = {identifier for identifier, vote in reviewed["votes"].items() if vote["role"] == "disagreement"}
        self.assertEqual(len(opposing), 1)
        self.assertTrue(opposing <= matching)
        self.assertTrue(set(strong["trends"][0]["voice_ids"]) <= matching)

    def test_refinement_accumulates_mixed_sample_instead_of_replacing_it(self):
        class ProgressiveJev(FakeJev):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.topic_reviews = 0

            def ask(self, state, questions):
                if state.get("task") == "review_sampled_candidate" and topic_of(state.get("proposals")) == 0:
                    self.match_limit = 3 if self.topic_reviews == 0 else 5
                    self.topic_reviews += 1
                return super().ask(state, questions)

        real_sample = candidate_helpers.sample_candidate

        def bounded_refinement(candidate, *args, **kwargs):
            packet = real_sample(candidate, *args, **kwargs)
            if ":refine:" in candidate["id"]:
                packet["targets"] = packet["targets"][:5]
            return packet

        with patch(f"{__name__}.FakeJev", ProgressiveJev), \
                patch.object(candidate_helpers, "sample_candidate", side_effect=bounded_refinement):
            result, diagnostics, _, reviewer = self.run_pipeline(
                self.write("accumulated-refinement.jsonl", fixture(large=True)), max_reviews=3)

        topic_reviews = [state for state in self.reviews(reviewer) if topic_of(state["proposals"]) == 0]
        self.assertEqual([len(state["candidate"]["targets"]) for state in topic_reviews], [10, 5])
        self.assertFalse({row["id"] for row in topic_reviews[0]["candidate"]["targets"]}
                         & {row["id"] for row in topic_reviews[1]["candidate"]["targets"]})
        self.assertEqual(len(result["trends"]), 1)
        selected = diagnostics["selected"][0]
        votes = selected["sample_review"]["votes"]
        self.assertEqual(len(votes), 15)
        self.assertEqual(sum(vote["match"] == "match" for vote in votes.values()), 8)
        self.assertAlmostEqual(selected["sample_review"]["match_ratio"], 8 / 15)
        # Refinement's perfect five-example sample cannot erase the original
        # seven unrelated examples or add them back to estimated topic volume.
        self.assertEqual(sum(vote["match"] == "other" for vote in votes.values()), 7)
        self.assertEqual(selected["history"]["stats"]["raw_count"], 280 - 7)
        self.assertTrue(set(result["trends"][0]["voice_ids"])
                        <= {identifier for identifier, vote in votes.items() if vote["match"] == "match"})

    def test_changed_votes_with_same_sample_count_refresh_the_output_draft(self):
        class ChangingStanceJev(FakeJev):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.topic_reviews = 0
                self.changed_id = None

            def ask(self, state, questions):
                answers = super().ask(state, questions)
                if state.get("task") == "review_sampled_candidate" and topic_of(state.get("proposals")) == 0:
                    self.topic_reviews += 1
                    if self.topic_reviews == 2:
                        self.changed_id = state["candidate"]["targets"][0]["id"]
                        answers["c0_t0_role"] = confident("disagreement")
                        answers["c0_t0_stance"] = confident("positive")
                return answers

        real_pool = candidate_helpers.build_candidate_pool

        def repeated_candidate(*args, **kwargs):
            pool = real_pool(*args, **kwargs)
            first = next(candidate for candidate in pool
                         if candidate["kind"] == "cluster" and int(np.argmax(candidate["center"])) == 0)
            repeated = deepcopy(first)
            repeated["id"] = first["id"] + ":another-view"
            return [first, repeated]

        with patch(f"{__name__}.FakeJev", ChangingStanceJev), \
                patch.object(candidate_helpers, "build_candidate_pool", side_effect=repeated_candidate):
            result, diagnostics, models, reviewer = self.run_pipeline(
                self.write("changed-votes.jsonl", fixture(large=True)), review_batch=1)

        reviews = self.reviews(reviewer)
        self.assertEqual(len(reviews), 2)
        self.assertEqual([row["id"] for row in reviews[0]["candidate"]["targets"]],
                         [row["id"] for row in reviews[1]["candidate"]["targets"]])
        drafts = [request[2] for request in models.requests
                  if request[0] == "chat" and request[2]["task"] == "draft_output"]
        self.assertEqual(len(drafts), 2)
        changed = reviewer.changed_id
        self.assertTrue(all("measurements" not in draft["evidence"] for draft in drafts))
        grounding = [questions["title"]["instructions"] for state, questions in reviewer.requests
                     if state.get("task") == "ground_output"]
        grounded_activity = [questions["action_valid"]["instructions"] for state, questions in reviewer.requests
                             if state.get("task") == "ground_output"]
        self.assertEqual(grounded_activity[0]["measurements"]["sampled_stances"], {"negative": 10})
        self.assertEqual(grounded_activity[1]["measurements"]["sampled_stances"], {"negative": 9, "positive": 1})
        self.assertNotIn(changed, grounding[0]["counterevidence_ids"])
        self.assertIn(changed, grounding[-1]["counterevidence_ids"])
        self.assertEqual(len(result["trends"]), 1)
        self.assertEqual(len(diagnostics["selected"][0]["sample_review"]["votes"]), 10)

    def test_unit_votes_keep_source_support_and_aggregate_only_matching_stances(self):
        concept = {}

        def review(unit_id, match, stance, role="experience"):
            target = {"id": "source"}
            if unit_id is not None:
                target["unit_id"] = unit_id
            pipeline._record_unit_votes(concept, [target], {
                "contributions": {"source": {"match": match, "role": role}}, "stances": {"source": stance}})

        review("source:0", "match", "negative")
        review("source:1", "other", "positive")
        self.assertEqual(concept["votes"], {"source": {"match": "match", "role": "experience"}})
        self.assertEqual(concept["stances"], {"source": "negative"})
        review("source:1", "match", "positive", "disagreement")
        self.assertEqual(concept["stances"], {"source": "mixed"})
        self.assertEqual(concept["votes"]["source"]["role"], "disagreement")
        self.assertEqual(len(concept["votes"]), 1)
        review("source:0", "match", "positive")
        self.assertEqual(concept["stances"], {"source": "positive"})
        self.assertEqual(len(concept["unit_votes"]["source"]), 2)
        review("source:0", "other", "unclear")
        review("source:1", "other", "unclear")
        self.assertEqual(concept["votes"]["source"]["match"], "other")
        self.assertNotIn("source", concept["stances"])
        concept.clear()
        review(None, "match", "negative")
        review(None, "other", "unclear")
        self.assertEqual(list(concept["unit_votes"]["source"]), ["source"])
        self.assertEqual(concept["votes"]["source"]["match"], "other")

    def test_neighbor_clause_votes_do_not_erase_another_matching_clause_of_the_same_source(self):
        for mode in ("unrelated_unit", "opposing_unit", "revised_same_unit"):
            with self.subTest(mode=mode):
                rows = fixture()
                shared_id = "topic-0-0-0"
                shared = next(row for row in rows if row["id"] == shared_id)
                other_text = ("TOPIC_0 A favorable assessment of the same subject" if mode == "opposing_unit"
                              else "TOPIC_1 A separate subject in the same source")
                shared["content"] += "; " + other_text
                real_pool, real_sample = candidate_helpers.build_candidate_pool, candidate_helpers.sample_candidate

                def repeated_candidate(*args, **kwargs):
                    first = next(candidate for candidate in real_pool(*args, **kwargs)
                                 if candidate["kind"] == "cluster" and int(np.argmax(candidate["center"])) == 0)
                    repeated = deepcopy(first)
                    repeated["id"] += ":another-view"
                    return [first, repeated]

                def alternate_clause(candidate, *args, **kwargs):
                    packet = real_sample(candidate, *args, **kwargs)
                    focal = next(target for target in packet["targets"] if target["id"] == shared_id)
                    # Keep the focal source outside the proposal anchor slots.
                    packet["targets"] = [row for row in packet["targets"] if row["id"] != shared_id] + [focal]
                    focal["unit_id"], focal["text"] = shared_id + ":0", shared["content"].split(";")[0]
                    if candidate["id"].endswith(":another-view") and mode != "revised_same_unit":
                        focal["unit_id"], focal["text"] = shared_id + ":1", other_text
                    return packet

                class ClauseJev(FakeJev):
                    def ask(self, state, questions):
                        answers = super().ask(state, questions)
                        if state.get("task") == "review_sampled_candidate" and state["candidate"]["candidate_id"].endswith(":another-view"):
                            index = next(i for i, target in enumerate(state["candidate"]["targets"]) if target["id"] == shared_id)
                            for ci, proposal in enumerate(state["proposals"]["concerns"]):
                                if topic_of(proposal["definition"]) == 0:
                                    prefix = f"c{ci}_t{index}"
                                    answers[prefix + "_match"] = confident("match" if mode == "opposing_unit" else "other")
                                    answers[prefix + "_role"] = confident("disagreement" if mode == "opposing_unit" else "other")
                                    answers[prefix + "_stance"] = confident("positive" if mode == "opposing_unit" else "unclear")
                        return answers

                with patch.object(candidate_helpers, "build_candidate_pool", side_effect=repeated_candidate), \
                        patch.object(candidate_helpers, "sample_candidate", side_effect=alternate_clause), \
                        patch(f"{__name__}.FakeJev", ClauseJev):
                    result, diagnostics, models, _ = self.run_pipeline(
                        self.write(f"focal-votes-{mode}.jsonl", rows), review_batch=1, max_refinements=0)
                self.assertEqual(len(result["trends"]), 1)
                selected = diagnostics["selected"][0]
                self.assertEqual(len(selected["sample_review"]["votes"]), 10)
                revised = mode == "revised_same_unit"
                self.assertEqual(selected["history"]["stats"]["raw_count"], 34 if revised else 35)
                self.assertEqual(selected["sample_review"]["votes"][shared_id]["match"], "other" if revised else "match")
                self.assertEqual(shared_id in result["trends"][0]["voice_ids"], not revised)
                if not revised:
                    shared_evidence = next(source for source in selected["evidence"] if source["id"] == shared_id)
                    self.assertEqual(shared_evidence["content"], shared["content"])
                    expected_focal = [shared["content"].split(";")[0]]
                    if mode == "opposing_unit":
                        expected_focal.append(other_text)
                    self.assertEqual(shared_evidence["reviewed_focal_texts"], expected_focal)
                expected_stances = ({"negative": 9} if revised else {"negative": 9, "mixed": 1}
                                    if mode == "opposing_unit" else {"negative": 10})
                self.assertEqual(selected["history"]["stats"]["sampled_stances"], expected_stances)
                if mode == "opposing_unit":
                    self.assertEqual(selected["sample_review"]["votes"][shared_id]["role"], "disagreement")
                self.assertEqual(len([request for request in models.requests if request[0] == "chat"]),
                                 1 if mode == "unrelated_unit" else 2)

    def test_cached_vectors_do_not_trigger_embedding_unload(self):
        path = self.write("cached.jsonl", fixture())
        result, _, models, _ = self.run_pipeline(path, cached_embeddings=True, embedding_residency_inputs=50)
        self.assertTrue(result["trends"])
        self.assertEqual(models.stats["embedding_calls"], 0)
        self.assertGreater(models.stats["embedding_cache_hits"], 0)
        self.assertGreater(len([request for request in models.requests if request[0] == "embed"]), 1)
        self.assertNotIn(("unload", "bge-m3"), models.requests)

    def test_embedding_residency_windows_preserve_output_evidence_and_review_inputs(self):
        path = self.write("embedding-windows.jsonl", fixture(topic_count=3))
        whole, whole_diagnostics, whole_models, whole_reviewer = self.run_pipeline(
            path, embedding_residency_inputs=0)
        windowed, windowed_diagnostics, windowed_models, windowed_reviewer = self.run_pipeline(
            path, embedding_residency_inputs=50)
        self.assertTrue(whole["trends"])
        self.assertEqual(windowed, whole)
        self.assertEqual(windowed_diagnostics["selected"], whole_diagnostics["selected"])
        self.assertEqual(windowed_diagnostics["candidate_summary"], whole_diagnostics["candidate_summary"])
        self.assertEqual(windowed_reviewer.requests, whole_reviewer.requests)
        self.assertEqual([request for request in windowed_models.requests if request[0] == "chat"],
                         [request for request in whole_models.requests if request[0] == "chat"])
        whole_inputs = [text for request in whole_models.requests if request[0] == "embed" for text in request[1]]
        windows = [request[1] for request in windowed_models.requests if request[0] == "embed"]
        self.assertEqual([text for window in windows for text in window], whole_inputs)
        self.assertTrue(all(len(window) <= 50 for window in windows))
        self.assertGreater(len(windows), 1)
        self.assertEqual(whole_models.stats["embedding_calls"], 1)
        self.assertEqual(windowed_models.stats["embedding_calls"], len(windows))
        self.assertEqual(windowed_models.requests.count(("unload", "bge-m3")), len(windows))

    def test_only_residency_windows_with_inference_unload_the_encoder(self):
        class AlternatingCacheModels(FakeModels):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.window_count = 0
                self.residency_events = []

            def embed(self, texts):
                index = self.window_count
                self.window_count += 1
                self.cached_embeddings = bool(index % 2)
                self.residency_events.append(("embed", index, not self.cached_embeddings))
                return super().embed(texts)

            def unload(self, model):
                if model == "bge-m3":
                    self.residency_events.append(("unload", model))
                return super().unload(model)

        with patch(f"{__name__}.FakeModels", AlternatingCacheModels):
            result, _, models, _ = self.run_pipeline(
                self.write("mixed-cache-windows.jsonl", fixture()), embedding_residency_inputs=50)
        self.assertTrue(result["trends"])
        self.assertEqual(models.residency_events, [
            ("embed", 0, True), ("unload", "bge-m3"),
            ("embed", 1, False),
            ("embed", 2, True), ("unload", "bge-m3"),
            ("embed", 3, False),
        ])
        self.assertEqual(models.stats["embedding_calls"], 2)
        self.assertEqual(models.stats["embedding_cache_hits"], 75)

    def test_prepared_packet_compaction_preserves_time_and_platform_series(self):
        raw_stats = {
            "raw_count": 9,
            "daily": [
                {"day": "2026-07-01", "raw_count": 2, "provisional_author_day_support": 1.5},
                {"day": "2026-07-02", "raw_count": 7, "provisional_author_day_support": 4.0},
            ],
            "platform_windows": {"QQ": {"prior_count": 2, "recent_count": 7}},
            "platform_history": {"QQ": {"first_seen": "2026-07-01", "last_seen": "2026-07-02"}},
            "boundary": {"partial_end": False},
        }
        packet = {"targets": [{"id": "source"}], "context": [], "stats": raw_stats}
        original = deepcopy(packet)
        prepared = pipeline._bounded_packet(packet)
        evaluated = pipeline._bounded_packet(prepared)
        self.assertEqual(evaluated, prepared)
        self.assertEqual(packet, original)
        self.assertEqual(evaluated["stats"]["daily_activity"], [
            ["2026-07-01", 2, 1.5], ["2026-07-02", 7, 4.0]])
        self.assertEqual(evaluated["stats"]["platform_activity"], raw_stats["platform_windows"])
        self.assertEqual(evaluated["stats"]["platform_history"], raw_stats["platform_history"])

    def test_new_vectors_are_unloaded_before_generation(self):
        result, _, models, _ = self.run_pipeline(self.write("cold.jsonl", fixture()))
        self.assertTrue(result["trends"])
        self.assertEqual(models.stats["embedding_calls"], 1)
        unload = models.requests.index(("unload", "bge-m3"))
        first_chat = next(index for index, request in enumerate(models.requests) if request[0] == "chat")
        self.assertLess(unload, first_chat)

    def test_output_validator_rejects_missing_and_future_sources(self):
        result = {"as_of": AS_OF, "trends": [{"title": "问题", "title_en": "Concern", "category": "Feedback",
                  "status": "new", "first_seen": "2026-07-01", "voice_ids": ["known"], "why": "Observed evidence."}]}
        records = [{"id": "known", "day": "2026-07-01"}, {"id": "future", "day": "2026-07-15"}]
        pipeline.validate_output(result, records)
        for identifier in ("missing", "future"):
            invalid = deepcopy(result)
            invalid["trends"][0]["voice_ids"] = [identifier]
            with self.assertRaises(ValueError):
                pipeline.validate_output(invalid, records)
