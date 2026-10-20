# Semantic diagnostic

[reviewed_cases.json](reviewed_cases.json) contains Chinese messages, context, English readings and expected **relevance and stance** judgments. These assistant-reviewed development labels are not independent gold. They do not benchmark current topic discovery, final writing or population accuracy.

[evaluate_decisions.py](../evaluate_decisions.py) compares Jev decisions with those labels and reports agreement, abstentions, coverage and usage. Only original messages and context enter model inputs; expected labels and English annotations are excluded.

```text
python evaluate_decisions.py --limit 12 --out outputs/jev-evaluation.json
```

This optional command can send fixture text to Jev and consume credits. It requires a supported Jev API key; exact cached decisions are reused. Model precedence is `--model` → nonempty process `TREND_JEV_MODEL` → the file selected by `--dotenv` → the default. `--limit 0` selects all examples. Run `python evaluate_decisions.py --help` for options.

Software regression tests are separate and run offline with mocked services:

```text
python -m unittest discover -s tests -v
```

The model diagnostic is not run automatically by the test suite. See [recorded pipeline evaluation](../docs/EVALUATION.md) for the three cutoff runs, source audits and limitations.
