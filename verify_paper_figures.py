#!/usr/bin/env python3
"""Recompute the headline figures of the AIDA 2026 paper directly from the frozen logs.

Level-1 reproducibility: no API keys or network access are needed.
Usage:  python verify_paper_figures.py
Reads:  questions_30.json, experiment_logs/run_*_202610*.jsonl and *_scored_v2.json
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
LOG = ROOT / "experiment_logs"
RUNS = {
    "openai": ["20261022T220534Z", "20261023T013236Z", "20261023T014721Z",
               "20261023T015639Z", "20261024T000301Z"],
    "gemini": ["20261022T222736Z", "20261023T013845Z", "20261023T014943Z",
               "20261023T020224Z", "20261024T190554Z"],
}
UNSPECIFIED_COUNTRY = ["chained_01", "chained_06", "chained_07", "chained_09", "chained_10"]


def main():
    for prov, stamps in RUNS.items():
        per_q, exact, n = {}, 0, 0
        fails, usa, models, effective = Counter(), Counter(), Counter(), Counter()
        for st in stamps:
            scored = json.loads((LOG / f"run_{prov}_{st}_scored_v2.json").read_text(encoding="utf-8"))
            for q in scored["per_question"]:
                per_q.setdefault(q["question_id"], []).append(q["answer_accuracy_auto"])
                exact += q["answer_checks_correct"] == q["answer_checks_total"]
                n += 1
            for line in (LOG / f"run_{prov}_{st}.jsonl").read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                models[rec.get("model")] += 1
                effective.update({pr.get("effective_model") for pr in rec.get("provider_responses", [])
                                  if isinstance(pr, dict) and pr.get("effective_model")})
                for t in rec.get("tool_calls", []):
                    if not t.get("success"):
                        err = str(t.get("error", "")).lower()
                        if t.get("blocked_as_duplicate") or "duplicate" in err:
                            fails["blocked_duplicate"] += 1
                        elif "429" in err:
                            fails["http_429"] += 1
                        elif "timeout" in err or "timed out" in err:
                            fails["timeout"] += 1
                        else:
                            fails["other"] += 1
                if rec["question_id"] in UNSPECIFIED_COUNTRY:
                    countries = {str(t.get("arguments", {}).get("country_iso3", "")).upper()
                                 for t in rec.get("tool_calls", [])}
                    usa["responses"] += 1
                    usa["requested_USA"] += "USA" in countries
        macro = sum(sum(v) / len(v) for v in per_q.values()) / len(per_q)
        print(f"\n== {prov}  ({len(stamps)} runs, {n} responses)")
        print(f"   requested model (responses): {dict(models)}")
        print(f"   effective model (responses reporting it): {dict(effective)}")
        print(f"   macro accuracy (scoring v2): {macro:.4f}")
        print(f"   exact answers:               {exact}/{n}")
        print(f"   external tool failures:      {fails['http_429'] + fails['timeout']} "
              f"(429={fails['http_429']}, timeout={fails['timeout']}); "
              f"blocked duplicates={fails['blocked_duplicate']}; other={fails['other']}")
        print(f"   USA requested in country-unspecified chained questions: "
              f"{usa['requested_USA']}/{usa['responses']}")


if __name__ == "__main__":
    main()
