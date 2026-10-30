# AIDA 2026 — inferencia exploratoria con scoring v2

## Alcance
Cinco corridas de octubre por modelo; 30 preguntas emparejadas, 300 respuestas. Septiembre excluido. Scoring v2 aplicado idénticamente. No se interpreta el p-valor como confirmatorio.

## Resultados principales

| Metric | GPT-4o | Gemini 3.6 Flash |
|---|---:|---:|
| Mean macro accuracy (v2) | 0.8607 | 0.9706 |
| Exact answers | 114/150 | 140/150 |
| Median response latency (s) | 3.29 | 6.99 |

Paired mean accuracy difference (Gemini−GPT): **10.99 pp**; 95% question-cluster bootstrap percentile CI **[3.89, 19.69] pp**, 20,000 resamples, fixed seed 20261024. Exploratory Wilcoxon W=13.000, p=0.007357; 30 question-level paired differences.

Paired mean difference of per-question median latency (Gemini−GPT): **5.19 s**, bootstrap CI **[3.82, 6.76] s**. This estimand differs from the pooled response median.

## Sensitivity

GPT-4o v2 macro accuracy excluding run 4: **91.69%** (120 responses). Main result retains run 4. This unbalanced exclusion is diagnostic, not the primary paired comparison.

## USA direction checks

Of the 25 GPT-4o responses for the five country-unspecified chained questions, **8** had at least one direction check scored correct by v2. A response may have no direction check; the definition here is 'at least one correctly passed direction-labelled check', not 'all checks passed'. See source scoring fields.

## Interpretation and limitations

These results are exploratory and conditional on the 30 frozen questions and five October runs. Bootstrap resamples questions, not individual responses. Repeated runs are not independent experimental questions. Automated factual accuracy and contextual country validity are separate dimensions; a correct trend can refer to the wrong country. Tool-selection accuracy applies only to the 17 questions per run with a defined unique expected first tool. Historical human grounding annotations should not be generalized to October runs. A 20-pp interval-width threshold was discussed post hoc and was not preregistered.

## Suggested Results text (English)

Across 30 benchmark questions, each repeated five times per model, Gemini 3.6 Flash achieved higher automatically scored factual accuracy than GPT-4o under the revised rubric. The question-paired mean difference was 10.99 percentage points (95% question-bootstrap CI 3.89 to 19.69). The analysis was exploratory and does not establish general superiority beyond the evaluated tasks. Contextual country validity was assessed separately from factual-direction checks.

## Reproducibility

Source: ten October JSONL logs; five v2 JSON score files per provider; 30 frozen question IDs. Script: `analisis_inferencial_v2.py`. Fixed bootstrap seed 20261024, B=20000. The original scoring logic remains available in `original_logic_per_question` for paired sensitivity analyses.
