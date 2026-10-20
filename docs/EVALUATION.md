# Evaluation

The three recorded runs used local BGE-M3 embeddings, Jev 1.13.0 decisions and Qwen3 8B Q4_K_M writing through a CPU-only Ollama helper. No topic-specific prompts or thresholds changed during evaluation.

Source audits covered every cited message and fixed coverage samples chosen before review. These assistant-performed Chinese interpretations do not establish population precision or recall.

## Software tests

The latest software check passed **369 tests** in **24.180 seconds**. Tests cover input handling, cutoff isolation, grouping, selection, temporal rules, evidence binding, caching, configuration precedence, provider errors and CLI behavior. Cache-only integration tests use real SQLite vectors with Ollama connections blocked and hosted services mocked. The suite consumes no API credits. The OpenAI writer has no live quality evaluation.

```text
python -m unittest discover -s tests -v
```

Cache-only numerical preparation passed all three cutoffs using 6,304, 13,438 and 26,575 cached inputs. Ollama health checks, transport and outbound network requests were blocked; Jev and OpenAI clients were not constructed. A fresh bundle restore preserved all 27,925 vector rows byte for byte and all three shard hashes.

## Results

| Cutoff | Visible messages | Candidate groups / reviewed | Public findings | Runtime |
| --- | ---: | ---: | --- | ---: |
| June 30 | 8,925 | 79 / 78 | 7: 3 growing, 4 fading; 3 provisional | 26m 57s |
| July 24 | 21,954 | 168 / 30 | 8: 4 growing, 3 fading, 1 new | 52m 06s |
| August 30 | 48,125 | 333 / 34 | 8: 6 growing, 2 fading | 50m 36s |

The snapshots overlap. Each run allowed 100 candidate reviews, including refinement and neighbors. June used the allowance; July and August stopped after eight outputs. The table counts distinct base groups inspected, not review requests.

All three runs reused embeddings and cutoff-specific cluster snapshots. No new embedding inference or K-means fit was needed. Total runtime was **2h 09m 39s**, excluding source audits. These timings exclude cold ingestion; local writing accounted for approximately 93–99% of engine time.

## Jev usage

| Scope | Reported input + output tokens |
| --- | ---: |
| Completed runs | 6,151,667 |
| All evaluation attempts, including completed runs | **10,688,946** |

Cache hits reuse earlier decisions without new inference.

## Findings and limits

| Area | Observed behavior | Remaining limit |
| --- | --- | --- |
| Message interpretation | Most inspected samples support coherent broad topics | One second-pass review attached a topic to the wrong source example. Sample agreement does not verify unsampled members. |
| Writing | Final checks often reject drafts that change the subject | Four topics failed both writing attempts. Three losses involved changed subjects; one rejection may have been too strict. Account-sale advertising remained vaguely labeled. |
| Suggested actions | Many suggestions are relevant to the evidence | Sixteen of 23 rows have predominantly Chinese actions; five address the player instead of the studio. The semantic rubric does not fully enforce language and audience. |
| Selection | Exploration and neighbor review supplement activity ranking | Offline audits found plausible topics that received no review or only partial coverage. Their eligibility as separate findings remains untested. |
| Temporal qualification | Histories are recomputed after source unions | A broad coherent topic can fall below a change threshold even when one geometric subset qualifies. This alone does not establish a bad merge. |
| Images and conversations | Captions, attachment counts and bounded channel context survive | No image pixels are supplied. Recovery of implicit busy-channel reaction cascades is not demonstrated by these runs. |

Passing software checks and title checks does not establish semantic accuracy. Release proximity cannot establish causation, and fading discussion cannot establish resolution.

[Architecture](ARCHITECTURE.md) · [Algorithms](ALGORITHMS.md) · [Setup](../README.md)
