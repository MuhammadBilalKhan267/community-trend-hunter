# Community Trend Hunter

Find changing topics in timestamped community messages. BGE-M3 and K-means propose groups, Jev reviews selected evidence, and Ollama or OpenAI writes bilingual findings. Python calculates activity and selects source IDs.

[Architecture](docs/ARCHITECTURE.md) · [Algorithms](docs/ALGORITHMS.md) · [Evaluation](docs/EVALUATION.md) · [Outputs](results/README.md) · [Chinese explanations](docs/CHINESE_EXPLANATIONS.md)

![Pipeline architecture](docs/assets/architecture.svg)

## Repository layout

| Path | Purpose |
| --- | --- |
| `run.py` | Command-line entry point |
| `trend_hunter/` | Input preparation, grouping, selection, model adapters, evidence checks and output |
| `scripts/` | Embedding ingestion, bundle export/restore and optional Windows runtime setup |
| `tests/` | Unit and regression tests |
| `evaluation/`, `evaluate_decisions.py` | Labeled relevance/stance examples and an optional Jev diagnostic |
| `docs/` | Architecture, algorithm formulas, evaluation results and SVG diagrams |
| `results/` | Saved JSON outputs for the three evaluated dates |
| `embeddings/` | Lossless vector shards, integrity manifest and restore instructions |
| `voices.jsonl`, `versions.csv` | Input messages and release calendar |

`.gitignore` excludes credentials, runtime caches, generated outputs, downloaded models/tools and local archives. `.env.example` contains placeholders only.

## Setup

Install Python 3.12 or newer. Run the commands below from the repository root, which contains `run.py`.

```text
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows; `source .venv/bin/activate` on POSIX), then install dependencies:

```text
python -m pip install -r requirements.txt
```

For full runs, copy [.env.example](.env.example) to a private `.env` file and replace the `JEV_API_KEY` placeholder with your key. You can also set `JEV_API_KEY` or `TYPESAFE_API_KEY` in the process environment. The OpenAI writer also requires `OPENAI_API_KEY`. Offline cache-only preparation needs neither key.

Configuration resolves in this order: **CLI option → nonempty process environment → literal `.env` value → default**.

| Setting | CLI flag | Environment variable |
| --- | --- | --- |
| Writer provider | `--chat-provider` | `TREND_CHAT_PROVIDER` |
| Local writer model | `--chat-model` with `--chat-provider ollama` | `TREND_OLLAMA_CHAT_MODEL` |
| OpenAI writer model | `--chat-model` with `--chat-provider openai` | `TREND_OPENAI_CHAT_MODEL` |
| Embedding model | `--embedding-model` | `TREND_EMBEDDING_MODEL` |
| Decision model | `--jev-model` | `TREND_JEV_MODEL` |
| Local Ollama endpoint | `--ollama-url` | `TREND_OLLAMA_URL` |

Ollama with `qwen3:8b` remains the default writer. Provider selection is explicit; an API key does not select OpenAI. `--chat-model` overrides the model for the selected provider. OpenAI requires a configured model and never inherits the local model setting.

Jev receives selected originals and bounded context. With the OpenAI writer enabled, reviewed writing evidence also goes to OpenAI. Stored embeddings stay local. The OpenAI path needs no Ollama server or model downloads after the embedding bundle is restored.

## Embeddings

Restore the supplied lossless [embedding bundle](embeddings/README.md):

```text
python scripts/embedding_cache.py restore --bundle embeddings --cache .cache/models.sqlite3
```

The OpenAI path reads embeddings only from this cache. It validates the recorded encoder profile and every required vector before contacting Jev or OpenAI. Missing or invalid cache data stops the run with `embedding_cache_error`; it never starts local inference. See [bundle details](embeddings/README.md) for profile compatibility and recovery.

Embedding ingestion requires [Ollama](https://docs.ollama.com/) and the encoder:

```text
ollama pull bge-m3
```

For local generation, also install the writer:

```text
ollama pull qwen3:8b
```

Keep Ollama running. If needed, start `ollama serve` in a separate terminal. Local runs query the installed encoder identity; cache hits avoid BGE inference. For new or changed input, ingest missing representations:

```text
python scripts/ingest_embeddings.py --input voices.jsonl --cache .cache/models.sqlite3 --batch-size 64 --embedding-model bge-m3
```

Ingestion needs only BGE, the embedding-model setting and the Ollama URL; it needs no writer or Jev configuration. It records the validated encoder profile and unloads BGE afterward. Writer changes preserve embeddings, while a different encoder identity can require new vectors. Each analysis cutoff uses its own eligible grouping; a vector database is not required.

## Run

The examples select models through CLI flags, which override environment and `.env` settings. Keep credentials in the environment or private `.env` file.

Run with the local writer:

```text
python run.py --input voices.jsonl --as-of 2026-08-30 --out trends.json --review-out outputs/review.json --embedding-model bge-m3 --chat-provider ollama --chat-model qwen3:8b
```

To use OpenAI:

```text
python run.py --input voices.jsonl --as-of 2026-08-30 --out trends.json --review-out outputs/review.json --embedding-model bge-m3 --chat-provider openai --chat-model gpt-6.1-sol
```

[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) is the suggested hosted writer; it supports the Responses API and Structured Outputs. This backend uses the same prompts and Jev checks. Its Chinese output quality and completion within the current token limit remain unevaluated.

Change `--as-of` to `2026-06-30` or `2026-07-24` for the other recorded snapshots. Full runs consume Jev credits; see [Evaluation](docs/EVALUATION.md) for recorded runtime and usage.

A successful run exits with code 0 and prints `Wrote N trends ...; stop=...`. It writes `trends.json` and `outputs/review.json`. Exit code 2 indicates an embedding-cache, model-service or cleanup failure; inspect the review file for the error and any partial results.

The public file contains at most eight findings. The review file records sampled decisions, evidence, histories, rejected drafts and usage. Use `--max-reviews N` to limit candidate reviews. With this option omitted, review stops after enough findings, candidate exhaustion or the token budget.

After restoring the bundle, prepare candidates entirely offline without Ollama or API keys:

```text
python run.py --input voices.jsonl --as-of 2026-08-30 --out outputs/prepared.json --review-out outputs/prepared-review.json --embedding-model bge-m3 --chat-provider openai --prepare-only
```

With the default local provider, `--prepare-only` can embed cache misses through Ollama. With the OpenAI provider, it requires complete cached embeddings. Both modes skip writer and Jev calls. Run `python run.py --help` for all options.

## Approach

Author breadth and capped repetition rank groups, with separate emerging and unusual review queues. Jev checks membership from up to ten varied original examples per request; shared sentiment alone does not establish a topic. Confirmed merges trigger a fresh history count.

Relevant, scope-supported topics need at least 80% matching examples. Those with at least 50% but below 80% remain provisional and can fill exhausted-search vacancies without displacing accepted findings.

Only reviewed matching originals become citations; other cluster members supply approximate activity counts. Replies and same-channel context clarify meaning without implying agreement. Jev checks final labels and optional actions.

Time comparisons use cutoff-valid activity and platform-normalized shares. **New** means recent appearance with sufficient support; **growing** means a supported increase; **fading** means decline. **One-off** requires a short peak followed by quieter observations. **Not real** flags extreme repeated-template concentration, not proven bots. Exact gates are in [Algorithms](docs/ALGORITHMS.md).

## What did not work

Small local models misread slang and sarcasm. Generated topic proposals and separate observation paragraphs introduced unsupported meaning, so production review now starts from original sources and reuses checked titles. The current writer still makes interpretation and language errors; finite review also misses useful topics. [Evaluation](docs/EVALUATION.md) records these failures.

## End-of-August studio overview

At the August 30 cutoff, growing discussion concerns spending pressure, power perceived to depend on paid skins, and stronger new characters such as Selena. Compare progression and balance telemetry with these reports and explain monetization clearly. These are player perceptions, not verified mechanics.

Story-skipping requests and pacing complaints are increasing, alongside Xiaomi 14 performance reports. Review skip controls and pacing, and reproduce device-specific lag before attributing it to a release.

Skin enthusiasm and some existing-character viability questions are fading; that does not establish resolution. Hidden-odds allegations remain a watchlist item below the growth gate. Pity-carryover questions deserve follow-up: an offline audit found them, but the pipeline did not review their central clauses.

## Development

Codex assisted with architecture, implementation, tests, Chinese interpretation, evaluation and documentation.

## Limits and verification

- Sample agreement does not verify every assigned message. Finite review can miss topics; aggregation can change their temporal classification.
- Models can misread Chinese, confuse context with the focal issue, or produce unsuitable actions.
- Image references have no supplied pixels. Attachment counts do not establish image meaning.
- Release proximity and declining discussion do not prove cause or resolution.

See [Evaluation](docs/EVALUATION.md) for recorded local-writer results, runtime, usage and failures. Software regression tests use mocked services and run offline:

```text
python -m unittest discover -s tests -v
```

The optional [semantic diagnostic](evaluation/README.md) uses Jev to inspect relevance and stance on labeled examples. It is separate from software tests and does not benchmark topic discovery or writer quality.
