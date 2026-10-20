# Saved outputs

These JSON files preserve the generated results from the three recorded local-writer runs. They use cached BGE-M3 embeddings, Jev decisions and Qwen3 8B generation.

| Cutoff | File | Findings |
| --- | --- | ---: |
| June 30 | [trends-2026-06-30.json](trends-2026-06-30.json) | 7, including 3 provisional |
| July 24 | [trends-2026-07-24.json](trends-2026-07-24.json) | 8 |
| August 30 | [trends-2026-08-30.json](trends-2026-08-30.json) | 8 |

Each file contains `as_of` and up to eight trends with bilingual titles, category, status, first-seen date, source IDs and an explanation. Generated wording is unchanged, including the issues documented in [Evaluation](../docs/EVALUATION.md). The OpenAI backend was added afterward and has no live quality evaluation.

The CLI writes a fresh result to the path supplied through `--out`. Runtime reports and caches remain excluded from version control.
