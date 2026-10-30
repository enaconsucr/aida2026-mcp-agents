#!/usr/bin/env python3
"""Análisis exploratorio reproducible AIDA 2026. Solo lectura de insumos; escribe en aida_resultados/."""
import csv, json, math, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path.cwd(); LOG=ROOT/'experiment_logs'; OUT=ROOT/'aida_resultados'; OUT.mkdir(exist_ok=True)
RUNS={
'GPT-4o':['20261022T220534Z','20261023T013236Z','20261023T014721Z','20261023T015639Z','20261024T000301Z'],
'Gemini 3.6 Flash':['20261022T222736Z','20261023T013845Z','20261023T014943Z','20261023T020224Z','20261024T190554Z']}
PREF={'GPT-4o':'openai','Gemini 3.6 Flash':'gemini'}
SEED=20261024; RNG=np.random.default_rng(SEED)

def writecsv(name, rows):
    if not rows:return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def avg(v):return float(np.mean(v)) if v else float('nan')
def sd(v):return float(np.std(v,ddof=1)) if len(v)>1 else float('nan')
def median(v):return float(np.median(v)) if v else float('nan')
def iqr(v):return (float(np.percentile(v,25)),float(np.percentile(v,75))) if v else (float('nan'),float('nan'))
def ci_boot(v, B=20000):
    a=np.asarray(v,dtype=float); idx=RNG.integers(0,len(a),size=(B,len(a)))
    return tuple(np.percentile(a[idx].mean(axis=1),[2.5,97.5]))

def main():
    qs=json.loads((ROOT/'questions_30.json').read_text(encoding='utf-8')); qmap={q['id']:q for q in qs}
    rows=[]; tools=[]; checks=[]; runrows=[]; alerts=[]; ambiguous=[]
    for model,dates in RUNS.items():
        for ri,dt in enumerate(dates,1):
            stem=f'run_{PREF[model]}_{dt}'; logpath=LOG/(stem+'.jsonl'); scorepath=LOG/(stem+'_scored.json')
            if not logpath.exists() or not scorepath.exists():
                raise FileNotFoundError(f'Falta {logpath if not logpath.exists() else scorepath}. Ejecuta scoring.py para las corridas faltantes.')
            logs=[json.loads(s) for s in logpath.read_text(encoding='utf-8').splitlines() if s.strip()]
            scored=json.loads(scorepath.read_text(encoding='utf-8')); scr={x['question_id']:x for x in scored['per_question']}
            ids=[r['question_id'] for r in logs]
            if len(ids)!=30 or len(set(ids))!=30 or set(ids)!=set(qmap):raise ValueError(f'Preguntas no coinciden en {stem}')
            effective=Counter()
            for r in logs:
                qid=r['question_id']; s=scr[qid]; prs=r.get('provider_responses') or []
                eff=set(p.get('effective_model') for p in prs if isinstance(p,dict) and p.get('effective_model'))
                effective.update(eff)
                if len(eff)>1:alerts.append(f'{stem} {qid}: varias versiones efectivas: {eff}')
                version=next(iter(eff),'NO_REGISTRADO')
                if model.startswith('Gemini') and version!='gemini-3.6-flash':raise ValueError(f'Versión Gemini no homogénea: {stem} {qid} {version}')
                calls=r.get('tool_calls') or []; external=[]; blocked=0; country=[]; failures=[]
                for j,c in enumerate(calls,1):
                    arg=c.get('arguments') or {}; iso=arg.get('country_iso3');
                    if iso:country.append(str(iso).upper())
                    err=str(c.get('error') or '')
                    dup=bool(c.get('blocked_as_duplicate'))
                    if dup:blocked+=1
                    if not c.get('success') and not dup:
                        failures.append(err)
                        if re.search(r'429|rate.limit|timeout|timed.out|timed out',err,re.I):external.append(err)
                    tools.append(dict(model=model,run=ri,question_id=qid,call=j,tool=c.get('tool_name'),country_iso3=iso,success=c.get('success'),duplicate_blocked=dup,error=err,external_error=bool(external and err in external)))
                score=s.get('answer_accuracy_auto'); score=float(score) if score is not None else float('nan')
                check_list=s.get('answer_checks') or []
                for ch in check_list:
                    checks.append(dict(model=model,run=ri,question_id=qid,category=r.get('category'),name=ch.get('name'),expected=str(ch.get('expected')),correct=ch.get('correct')))
                    if re.search(r'direction|increase|decrease|higher|lower|diverg',str(ch.get('name')),re.I):
                        ambiguous.append(dict(model=model,run=ri,question_id=qid,check=ch.get('name'),expected=ch.get('expected'),correct=ch.get('correct'),final_text=(r.get('final_text') or '')[:500]))
                gt=qmap[qid].get('ground_truth') or {}; explicit_country=bool(re.search(r'Costa Rica|United States|USA|Mexico|Brazil|Chile|Colombia',qmap[qid].get('question_en',''),re.I))
                country_wrong=bool(country and any(x=='USA' for x in country) and not explicit_country)
                rows.append(dict(model=model,run=ri,run_id=stem,question_id=qid,category=r.get('category'),status=r.get('status'),requested_model=r.get('model'),effective_model=version,timestamp_utc=r.get('timestamp_utc'),accuracy=score,exact=int(score==1.0),correct_checks=s.get('answer_checks_correct'),total_checks=s.get('answer_checks_total'),latency_seconds=r.get('latency_seconds'),tool_calls=len(calls),tool_success=sum(bool(c.get('success')) for c in calls),duplicates=blocked,external_failure=int(bool(external)),other_tool_failure=int(bool(failures) and not external),country_args=';'.join(country),usa_without_explicit_country=int(country_wrong),fallback=int(bool(r.get('used_fallback_synthesis'))),provider_rounds=len(prs),final_text=r.get('final_text') or '',question=qmap[qid].get('question_en','')))
            summary=scored['summary']
            runrows.append(dict(model=model,run=ri,run_id=stem,effective_models=';'.join(sorted(effective)),n=len(logs),accuracy_macro=summary.get('answer_accuracy_macro'),accuracy_micro=summary.get('answer_accuracy_micro_components'),exact_rate=summary.get('exact_answer_rate'),tool_selection=summary.get('tool_selection_accuracy'),mcp_success=summary.get('avg_tool_execution_success_rate'),latency_median=median([float(r.get('latency_seconds')) for r in logs if r.get('latency_seconds') is not None])))
    writecsv('01_corridas.csv',runrows);writecsv('02_respuestas.csv',rows);writecsv('03_herramientas.csv',tools);writecsv('04_checks.csv',checks);writecsv('05_direcciones_para_revision.csv',ambiguous)
    grouped=defaultdict(list)
    for r in rows:grouped[(r['question_id'],r['model'])].append(r)
    pairs=[]
    for q in qs:
        qid=q['id']; a=grouped[(qid,'GPT-4o')];b=grouped[(qid,'Gemini 3.6 Flash')]
        if len(a)!=5 or len(b)!=5:raise ValueError(qid)
        ma=avg([r['accuracy'] for r in a]);mb=avg([r['accuracy'] for r in b]);pairs.append(dict(question_id=qid,category=q['category'],gpt=ma,gemini=mb,difference=mb-ma,gpt_exact=avg([r['exact'] for r in a]),gemini_exact=avg([r['exact'] for r in b])))
    writecsv('06_pares_por_pregunta.csv',pairs)
    d=[p['difference'] for p in pairs];lo,hi=ci_boot(d)
    try:
        from scipy.stats import wilcoxon
        w=wilcoxon(d,zero_method='wilcox',alternative='two-sided'); wtxt=f'W={w.statistic:.3f}, p={w.pvalue:.5g}'
    except Exception as e:wtxt=f'No calculado ({e})'
    # Variación por pregunta y repetición: descriptiva, sin suponer independencia.
    varrows=[]
    for model in RUNS:
        rr=[r for r in rows if r['model']==model]; grand=avg([r['accuracy'] for r in rr]); qmeans=[avg([r['accuracy'] for r in grouped[(q['id'],model)]]) for q in qs]
        within=avg([float(np.var([r['accuracy'] for r in grouped[(q['id'],model)]],ddof=1)) for q in qs]);between=float(np.var(qmeans,ddof=1))
        runmeans=[avg([r['accuracy'] for r in rr if r['run']==j]) for j in range(1,6)]
        varrows.append(dict(model=model,grand_mean=grand,between_question_variance=between,within_question_run_variance=within,run_mean_sd=sd(runmeans),run_mean_min=min(runmeans),run_mean_max=max(runmeans),note='Componentes descriptivos; no son estimadores ANOVA ortogonales ni efectos causales.'))
    writecsv('07_variabilidad.csv',varrows)
    bycat=[]
    for model in RUNS:
        for cat in sorted(set(r['category'] for r in rows)):
            a=[r for r in rows if r['model']==model and r['category']==cat]
            lat=[float(r['latency_seconds']) for r in a if r['latency_seconds'] is not None]
            q1,q3=iqr(lat);bycat.append(dict(model=model,category=cat,n=len(a),accuracy=avg([r['accuracy'] for r in a]),exact=avg([r['exact'] for r in a]),latency_median=median(lat),latency_q1=q1,latency_q3=q3,external_failures=sum(r['external_failure'] for r in a),fallbacks=sum(r['fallback'] for r in a)))
    writecsv('08_categorias_latencia.csv',bycat)
    issues=[]
    for model in RUNS:
        for r in [r for r in rows if r['model']==model and r['accuracy']<1]:
            issues.append(dict(model=model,run=r['run'],question_id=r['question_id'],accuracy=r['accuracy'],external_failure=r['external_failure'],usa_without_explicit_country=r['usa_without_explicit_country'],other_tool_failure=r['other_tool_failure'],note='Marcadores no exclusivos; revisar atribución causal manualmente'))
    writecsv('09_errores_para_clasificar.csv',issues)
    usa=[]
    for q in qs:
        if re.search(r'Costa Rica|United States|USA|Mexico|Brazil|Chile|Colombia',q.get('question_en',''),re.I):continue
        for model in RUNS:
            for r in grouped[(q['id'],model)]:
                usa.append(dict(model=model,run=r['run'],question_id=q['id'],country_args=r['country_args'],usa=r['usa_without_explicit_country'],accuracy=r['accuracy']))
    writecsv('10_paises_no_especificados.csv',usa)
    consistency=[]
    for q in qs:
        for model in RUNS:
            arr=grouped[(q['id'],model)]; signatures=[]
            for r in arr:
                cc=[c for c in checks if c['model']==model and c['run']==r['run'] and c['question_id']==q['id']]
                signatures.append(tuple((c['name'],c['correct']) for c in cc))
            consistency.append(dict(model=model,question_id=q['id'],same_check_outcome_across_five=len(set(signatures))==1,accuracy_sd=sd([r['accuracy'] for r in arr]),note='Consistencia de resultados de checks, NO igualdad de hechos/valores; exige extracción adicional'))
    writecsv('11_consistencia_checks.csv',consistency)
    report=[]
    report.append('# AIDA 2026 — exploratory analysis (October runs)\n')
    report.append('**Scope:** 30 fixed questions × 5 repetitions × 2 models (300 completed-answer records). Question is the paired analysis unit (n=30). All logged responses retained.\n')
    report.append('## Table 1. Overall performance (descriptive)\n')
    report.append('| Model | Macro accuracy | Exact responses | Median latency (s) |')
    report.append('|---|---:|---:|---:|')
    for model in RUNS:
        a=[r for r in rows if r['model']==model];report.append(f"| {model} | {avg([r['accuracy'] for r in a]):.3f} | {sum(r['exact'] for r in a)}/150 | {median([r['latency_seconds'] for r in a]):.2f} |")
    report.append('\n*Note:* Macro accuracy is the average of question-level scoring fractions across repeated runs. Repeated answers are not independent; descriptive counts are not a population estimate. Grounding annotations from September are NOT applied to October.\n')
    report.append('## Table 2. Paired question-level difference\n')
    report.append(f'Gemini − GPT-4o mean difference: **{avg(d):.4f}**; question-cluster bootstrap 95% percentile CI **[{lo:.4f}, {hi:.4f}]** (20,000 resamples, seed {SEED}); Wilcoxon signed rank (exploratory): **{wtxt}**.\n')
    report.append('Bootstrap samples 30 questions with replacement, retaining each question’s five runs per model through aggregation. This CI measures question-set sampling sensitivity, not generalizability to all MCP tasks.\n')
    report.append('## Table 3. Between-run variation\n')
    report.append('| Model | Run-mean SD (pp) | Min–max macro accuracy | Between-question variance | Within-question variance |')
    report.append('|---|---:|---:|---:|---:|')
    for v in varrows:report.append(f"| {v['model']} | {v['run_mean_sd']*100:.2f} | {v['run_mean_min']:.3f}–{v['run_mean_max']:.3f} | {v['between_question_variance']:.4f} | {v['within_question_run_variance']:.4f} |")
    report.append('\n*Note:* The SD refers to the five **run-level means**, not the 150 responses. Variance columns are descriptive, not a formal mixed-effects variance-component model.\n')
    report.append('## Failure attribution and sensitivity\n')
    for model in RUNS:
        a=[r for r in rows if r['model']==model];bad=[r for r in a if r['accuracy']<1]
        report.append(f"- {model}: {len(bad)} non-exact answers; among these, {sum(r['external_failure'] for r in bad)} had flagged external 429/timeout, {sum(r['usa_without_explicit_country'] for r in bad)} had USA in tool arguments without an explicit country in the prompt. These indicators can overlap and are **not** causal attributions. All answers remain in the denominator.")
    report.append('\nThe country flag only detects USA in tool arguments when selected country names are absent from the question text. It does not prove USA was incorrect: verify the expected country in the benchmark and retrieved evidence. For causal classification, manually adjudicate `09_errores_para_clasificar.csv`.\n')
    report.append('## Limitations and open decisions\n')
    report.append('- The fixed benchmark has only 30 distinct questions; five repetitions improve within-benchmark stability estimates but not task coverage.\n- No confirmatory inference or preregistered significance claims.\n- The original-versus-negation-corrected scorer cannot be compared unless **both** scoring outputs are supplied with documented version identifiers. This script consumes the currently supplied `_scored.json` only; it never invents a corrected score.\n- Full factual consistency requires extracting and comparing actual values/directions/countries from tool results and text; `11_consistencia_checks.csv` reports only agreement of check outcomes.\n- External API failures are marked, not excluded; blocked duplicate calls are separate.\n- The data do not establish actual parallel execution merely from multiple tool calls; timestamps or runtime traces are required.\n- Reasoning configuration, system prompt, and tool-iteration cap require source-code/config verification for each historical run; do not infer settings from current files alone.\n- `chained_06` and directional checks require human adjudication; inspect `04_checks.csv`, `05_direcciones_para_revision.csv`, and raw tool responses.\n- The expansion rule must be agreed independently of observed model rankings. Suggested precision-only rule: consider new runs only if the paired CI width exceeds 0.20 **and** within-question run variance is a material share of uncertainty; this is a proposed threshold, not a preregistered one.\n')
    report.append('## Suggested Results text (exploratory)\n')
    report.append(f'"Across 30 fixed questions repeated five times per model, the observed mean automatic factual accuracy was {avg([r["accuracy"] for r in rows if r["model"]=="GPT-4o"])*100:.1f}% for GPT-4o and {avg([r["accuracy"] for r in rows if r["model"]=="Gemini 3.6 Flash"])*100:.1f}% for Gemini 3.6 Flash. The paired question-level difference (Gemini minus GPT-4o) was {avg(d)*100:.1f} percentage points (exploratory question-bootstrap 95% CI {lo*100:.1f} to {hi*100:.1f} points). These estimates describe performance on the prespecified benchmark and should not be generalized to arbitrary tasks."\n')
    report.append('## Reproducibility\n')
    report.append('See `01_corridas.csv` for run IDs and effective versions, `02_respuestas.csv` for requested/effective models and UTC timestamps. Confirm temperature=0.0, five-round cap, reasoning defaults, and system prompt directly against archived source and configuration before manuscript submission.\n')
    (OUT/'INFORME_AIDA.md').write_text('\n'.join(report),encoding='utf-8')
    print('Análisis completado:',OUT.resolve());print('Diferencia pareada Gemini-GPT:',round(avg(d),4),'IC95%',tuple(round(x,4) for x in (lo,hi)));print('Wilcoxon:',wtxt);print('CSV generados: 11; informe: INFORME_AIDA.md')
if __name__=='__main__':main()
