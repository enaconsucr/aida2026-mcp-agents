# Beyond Factual Accuracy: MCP-based LLM agents for heterogeneous public-sector data

Code, benchmark, frozen logs, scoring rubric and grounding annotations for the AIDA 2026 paper
*"Beyond Factual Accuracy: Evaluating MCP-Based LLM Agents for Heterogeneous Public-Sector Data Analytics"*
(Leonardo Castellón-Rodríguez and Pedro Palos-Sánchez; Procedia Computer Science, Elsevier).

Two LLM agents (GPT-4o and Gemini Flash) answer 30 frozen questions about public-health data for Costa Rica
through a Model Context Protocol (MCP) server with three tools: World Bank (REST), OECD (SDMX) and a local corpus
of 60,278 Google Play reviews of the EDUS digital-health app.

## Two levels of reproducibility

| Level | What it reproduces | Needs API keys? |
|---|---|---|
| **1. Re-compute the published results** | Every number in the paper, from the frozen logs in this repository | No |
| **2. Re-run the experiment** | New answers from the models using the same code, benchmark and settings | Yes (OpenAI and Google) |

Level 2 results **will not match exactly**: model endpoints change (Google served `gemini-3.6-flash` while
`gemini-3.5-flash` was requested), and the live World Bank and OECD APIs may revise their data. The raw API
responses received during the paper's runs are archived in `raw_responses/` for that reason.

## Level 1: re-compute the paper's figures

```bash
pip install -r requirements-analysis.txt
python verify_paper_figures.py      # accuracy, exact answers, external failures, wrong-country calls
python analisis_inferencial_v2.py   # paired difference, bootstrap CI, Wilcoxon -> salida_inferencial/
cd annotation && python acuerdo.py annotator2_labels_v20.xlsx grounding_annotations_audited_v1_1.json blinding_key.json
```

### Where each result comes from

| Result in the paper | Value | Source file(s) | Script |
|---|---|---|---|
| Macro factual accuracy, October runs | GPT-4o 0.8607 · Gemini 0.9706 | `experiment_logs/*_scored_v2.json` | `verify_paper_figures.py` |
| Exact answers | 114/150 · 140/150 | same | `verify_paper_figures.py` |
| Paired difference (Gemini − GPT-4o) | 10.99 pp, 95% CI [3.89, 19.69], Wilcoxon W = 13, p = 0.0074 | same + `*.jsonl` | `analisis_inferencial_v2.py` |
| Median latency | 3.29 s · 6.99 s | `*.jsonl` | `analisis_inferencial_v2.py` |
| External tool failures (HTTP 429 + timeout) | 16 · 49 (43 + 6); 17 blocked duplicates for Gemini, reported separately | `*.jsonl` | `verify_paper_figures.py` |
| GPT-4o requested `USA` in the five country-unspecified chained questions | 25/25 (Gemini 0/25) | `*.jsonl` | `verify_paper_figures.py` |
| Grounding audit (September run, 10 chained questions per model) | see paper, Table 1 Panel B | `annotation/grounding_annotations_audited_v1_1.json` | — |
| Inter-annotator agreement before adjudication (Cohen's κ) | entity 0.50, evidence −0.22, causal 0.44, overall 0.18 | `annotation/` | `annotation/acuerdo.py` |

The September grounding audit uses `run_openai_20260907T024524Z` and `run_gemini_20260907T025742Z`.

## Scoring rubric: versions

- `scoring.py` (v1) is the scorer used in the original submission.
- `scoring_v2.py` (v2) fixes negation handling ("the values do **not** match" was scored as a match) and adds a
  contextual country-validity check. **All accuracy figures in the camera-ready paper use v2.**
- The only October difference between v1 and v2 is `chained_04` for GPT-4o (6/6 → 5/6 in four of five runs).
  Both outputs are kept: `*_scored.json` (v1) and `*_scored_v2.json` (v2).
- Expected values and check rules for each question are in `questions_30.json` (`ground_truth`).
- Factual checks and country validity are scored separately. Example: in `chained_06` GPT-4o passes the factual
  direction checks in 4 of 5 October runs, but `contextual_validity.country_valid_auto = 0` in all five, because it
  retrieved United States data. This is the gap between factual accuracy and grounding discussed in the paper.
- `analisis_aida_integral.py` is a descriptive companion analysis that reads the **v1** score files; use it for
  latency, tool-call and failure tables, not for accuracy.

## Level 2: re-run the experiment

```bash
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY and GOOGLE_API_KEY, then export them
python smoke_test.py            # checks the MCP server and the three tools
python run_experiment.py --questions questions_30.json --provider openai --model gpt-4o-2024-08-06
python run_experiment.py --questions questions_30.json --provider gemini --model gemini-3.5-flash
python scoring_v2.py --log experiment_logs/<new_run>.jsonl --questions questions_30.json --grounding annotation/grounding_annotations.json
```

Settings used in the paper (fixed in the code): temperature 0.0; at most 5 tool-calling rounds per question;
up to 3 attempts for transient provider errors (backoff 2 s and 4 s, ±20% jitter); duplicate tool calls blocked;
fallback synthesis if rounds are exhausted; no explicit reasoning configuration (provider defaults).
The system prompt is the fixed instruction in `run_experiment.py` plus the `data://schema` resource published by
`mcp_server.py`. GPT-4o uses Chat Completions; Gemini uses the `google-genai` SDK with automatic function calling
disabled and `thought_signature` preserved in the history. Each log records the requested and the effective model.

## Repository contents

| Path | Content |
|---|---|
| `mcp_server.py`, `mcp_client_bridge.py` | MCP server (3 tools + `data://schema`) and client bridge |
| `providers.py`, `tool_schemas.py`, `run_experiment.py` | Provider adapters (OpenAI, Gemini), tool schemas, experiment harness |
| `worldbank_tool.py`, `oecd_tool.py`, `local_reviews_tool.py` | Tool implementations |
| `train_topic_model.py`, `models/bertopic_negative_reviews_v1/`, `topic_audit_38_coded_v1.csv` | Offline BERTopic topic assignments for negative reviews (`review_id` → `topic_id`), training metadata, topic report and the human-coded taxonomy (frozen; SHA-256 checked by the tool). The fitted model files are not distributed because their vocabulary contained personal data; the tool does not need them |
| `questions_30.json`, `questions_30.freeze.json` | Frozen benchmark with ground truth and its SHA-256 |
| `data/resenas_edus_google_play.csv` | EDUS reviews: `review_id` (pseudonym), `comentario`, `calificacion`, `fecha` only |
| `experiment_logs/` | Ten October runs (5 per model), two September runs used for the grounding audit, v1 and v2 scores |
| `raw_responses/` | Raw World Bank and OECD responses received during those runs (426 files) |
| `annotation/` | Authors' labels (v1.0 frozen, v1.1 audited, change log), blind second annotator's labels, blinding key, annotation guide and cases, agreement script |
| `salida_inferencial/` | Output of `analisis_inferencial_v2.py` as reported in the paper |
| `reproducibility/` | SHA-256 manifests |

## Data, privacy and terms of use

- **EDUS reviews.** Public Google Play reviews of the EDUS app (Caja Costarricense de Seguro Social). Reviewer
  names and profile images were **removed**; only the four columns the tool needs are kept.
- **Review identifiers.** Google Play review IDs were replaced by sequential pseudonyms (`EDUS-000001`, ...), because
  an original ID can be used to open the review, and its author's profile, on Google Play. The mapping is kept by the
  authors and is not distributed.
- **Personal data written inside review texts** (names, national ID numbers, phone numbers and a birth year written
  by reviewers, names of individual health workers, and health-centre phone numbers) were replaced by `[***]` in 25
  reviews (8 of them also in the BERTopic assignments file). `reproducibility/redaction_log.csv` lists the affected `review_id`s and the category
  of data removed, without the data itself. Because the assignments file changed (redactions and pseudonymous IDs), its expected SHA-256 in
  `local_reviews_tool.py` was updated (the hash used during the experiment is kept in a comment); topic assignments
  are unchanged. The tool returns **identical results** with the published files and with the original corpus on
  all 11 distinct EDUS queries made by the models. Review texts remain the property of their authors and are shared for research reproducibility
  only; they are **not** covered by the CC BY licence below. Please do not attempt to re-identify reviewers.
- **World Bank** data: CC BY 4.0 (World Bank Open Data terms). **OECD** data: OECD terms and conditions.
- No API keys or credentials are included. Keys are read from environment variables.

## Licence

Code: MIT (`LICENSE`). Benchmark, scoring rubric, annotations and logs authored by us: CC BY 4.0.
Third-party data keep their own terms (see above).

## Citation

See `CITATION.cff`. A DOI for this release will be added here.
