# Stylometry Drift Engine — Hermes Plugin

Rewrite prose to reduce identifiable stylistic writing patterns while preserving meaning exactly. Code blocks (`` ```fenced``` `` and `` `inline` ``) are preserved verbatim — only natural-language prose is transformed. Fully deterministic, fully offline, zero AI calls.

## Quick Install

```bash
hermes plugins install https://github.com/<your-org>/stylometry-drift-engine
hermes plugins enable stylometry-drift-engine
# Restart the gateway:
hermes gateway restart
```

## What It Does

The plugin provides a single tool — `stylometry_drift` — that an LLM agent calls when asked to rewrite text. It applies four deterministic transformations:

| Transformation | What it changes |
|---|---|
| **Synonym substitution** | Swaps common words for alternatives from a 250+ entry dictionary |
| **Sentence length variation** | Combines short adjacent sentences; splits long ones at conjunctions |
| **Clause reordering** | Moves subordinate clauses (e.g. "Because X, Y" → "Y because X") |
| **Punctuation shifts** | Em-dashes, semicolons, comma-to-em dash, colon-to-em dash swaps |

All transformations are seeded from the input text's SHA-256 hash, so **the same input always produces the same output** at the same intensity level.

A **history** parameter accepts prior texts from the same author. The engine extracts 3/4/5-grams from those texts and refuses to repeat any matching patterns — preventing the most obvious style fingerprint.

## Code Block Safety

The engine **never** transforms code:

- **Fenced code blocks** (` ```...``` `) — extracted before transformation, reinserted verbatim afterward
- **Inline code** (`` `...` ``) — same treatment
- **Pure code input** (a file, script, or snippet) — detected by heuristic (`looks_like_code()`) and returned unchanged with a warning

When processing mixed text (prose with embedded code examples), only the prose around the code blocks is rewritten.

## Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | yes | — | The text to rewrite. Code blocks preserved verbatim |
| `intensity` | number | no | 0.5 | Transformation aggressiveness (0.0–1.0). 0 = minimal, 1.0 = maximum |
| `history` | array[string] | no | [] | Prior texts whose n-gram patterns to avoid |

## Response

```json
{
  "original": "The quick brown fox jumps over the lazy dog. ...",
  "rewritten": "The quick brown fox jumps over the lazy dog. ...",
  "intensity": 0.7,
  "transformations_applied": true
}
```

If input looks like pure code:
```json
{
  "original": "def hello(): print('hi')",
  "rewritten": "def hello(): print('hi')",
  "intensity": 0.0,
  "transformations_applied": false,
  "warnings": ["Input detected as code — returned unchanged"]
}
```

If `text` is empty or missing, returns `{"error": "No text provided. Supply a non-empty 'text' string."}`

## Determinism Guarantee

```
rewrite(text, intensity=0.5)  →  always the same output
rewrite(text, intensity=0.3)  →  different output than intensity 0.5
```

Seeded RNG from SHA-256 of `text`. Changing `intensity` or `history` changes the seed salt, producing different output.

## Project Structure

```
stylometry-drift-engine/
├── plugin.yaml          # Hermes plugin manifest
├── __init__.py          # register(ctx) entrypoint
├── engine.py            # Core rewrite engine (deterministic, local, code-safe)
├── schemas.py           # JSON tool schema definition
├── tools.py             # Tool handler — validates args, returns JSON
├── data/
│   └── synonyms.json    # 250+ synonym entries
└── README.md
```

## Requirements

- Hermes Agent 0.16+ (config schema v21+ for opt-in plugins)
- No external dependencies — stdlib only (`hashlib`, `json`, `random`, `re`, `pathlib`)

## License

MIT
