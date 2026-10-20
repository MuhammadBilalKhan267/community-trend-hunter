"""Populate the reusable, item-local embedding cache, then release model memory."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from trend_hunter.config import (ConfigurationError, DEFAULT_EMBEDDING_MODEL,
                                 DEFAULT_OLLAMA_URL, load_setting)
from trend_hunter.data import GAME_TIMEZONE, make_units, prepare
from trend_hunter.discovery import unique_units
from trend_hunter.local_models import LocalModels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="voices.jsonl")
    parser.add_argument("--as-of")
    parser.add_argument("--cache", default=".cache/models.sqlite3")
    parser.add_argument("--ollama-url", help="Override TREND_OLLAMA_URL")
    parser.add_argument("--embedding-model", help="Override TREND_EMBEDDING_MODEL")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-gpu", type=int)
    parser.add_argument("--residency-inputs", type=int, default=2048,
                        help="Release model memory after this many inspected inputs with inference; zero disables recycling")
    parser.add_argument("--report", default=".cache/embedding-ingestion.json")
    args = parser.parse_args()
    try:
        if args.ollama_url is None:
            args.ollama_url = load_setting("TREND_OLLAMA_URL", default=DEFAULT_OLLAMA_URL)
        if args.embedding_model is None:
            args.embedding_model = load_setting("TREND_EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL)
        if not args.ollama_url or not args.embedding_model or any(
                char.isspace() or ord(char) < 32 or ord(char) == 127 for char in args.embedding_model):
            raise ConfigurationError("Use a nonempty embedding model name and Ollama URL")
    except ConfigurationError as exc:
        parser.error(str(exc))
    if args.residency_inputs < 0:
        parser.error("residency-inputs must be nonnegative")
    cutoff = args.as_of
    if cutoff is None:
        with Path(args.input).open(encoding="utf-8-sig") as source:
            cutoff = max(datetime.fromisoformat(json.loads(line)["created_at"])
                         .astimezone(GAME_TIMEZONE).date().isoformat()
                         for line in source if line.strip())
    records, eligible, stats = prepare(args.input, cutoff)
    units = make_units(eligible)
    texts, _ = unique_units(units)
    options = {} if args.num_gpu is None else {"num_gpu": args.num_gpu}
    started = time.perf_counter()
    report = {"as_of": cutoff, "messages": len(records), "clauses": len(units),
              "unique_inputs": len(texts), "data": stats, "complete": False}
    with LocalModels(base_url=args.ollama_url, embedding_model=args.embedding_model,
                     cache_path=args.cache, batch_size=args.batch_size,
                     embedding_options=options, timeout=300, keep_alive="30m") as models:
        try:
            info = models.health(("embedding",))
            report["model"] = info["embedding_model"]
            print(json.dumps({"event": "start", "inputs": len(texts), "as_of": cutoff}), flush=True)
            loaded = False
            dimensions = None
            window = args.residency_inputs or len(texts) or 1
            for start in range(0, len(texts), window):
                before = models.stats["embedding_calls"]
                try:
                    result = models.embed(texts[start:start + window])
                finally:
                    # Even a failed batch can have loaded the encoder. The
                    # outer cleanup must still release it before generation.
                    loaded = models.stats["embedding_calls"] > before
                if (result.ndim != 2 or result.shape[0] != len(texts[start:start + window])
                        or result.shape[1] < 1 or not np.isfinite(result).all()):
                    raise RuntimeError("Encoder returned invalid embedding coverage")
                if dimensions is not None and result.shape[1] != dimensions:
                    raise RuntimeError("Encoder dimensions changed during ingestion")
                dimensions = result.shape[1]
                done = min(start + window, len(texts))
                elapsed = time.perf_counter() - started
                print(json.dumps({"event": "progress", "done": done, "total": len(texts),
                                  "seconds": round(elapsed, 1),
                                  "requests": models.stats["embedding_calls"],
                                  "cache_hits": models.stats["embedding_cache_hits"]}), flush=True)
                if loaded and args.residency_inputs:
                    models.unload(args.embedding_model)
                    loaded = False
            report["complete"] = True
            report["dimensions"] = dimensions
        finally:
            if models.stats["embedding_calls"] and locals().get("loaded", True):
                try:
                    models.unload(args.embedding_model)
                except Exception as exc:
                    report["unload_error"] = type(exc).__name__
            report["seconds"] = round(time.perf_counter() - started, 3)
            report["usage"] = dict(models.stats)
            destination = Path(args.report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "seconds": report["seconds"], "report": args.report}), flush=True)


if __name__ == "__main__":
    main()
