import json,glob,re,csv
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import wilcoxon
BASE=Path(__file__).parent; OUT=BASE/'salida_inferencial';OUT.mkdir(exist_ok=True)
rows=[];logs=[]
for provider in ('openai','gemini'):
 files=sorted((BASE/'experiment_logs').glob(f'run_{provider}_*scored_v2.json'))
 assert len(files)==5,(provider,len(files))
 for run,p in enumerate(files,1):
  obj=json.loads(p.read_text()); raw=BASE/'experiment_logs'/p.name.replace('_scored_v2.json','.jsonl')
  recs=[json.loads(s) for s in raw.read_text().splitlines() if s.strip()]
  assert len(recs)==30
  for q,old,log in zip(obj['per_question'],obj['original_logic_per_question'],recs):
   assert q['question_id']==old['question_id']==log['question_id']
   c=next((c for c in obj['contextual_validity']['per_question'] if c['question_id']==q['question_id']),{})
   rows.append(dict(model=provider,run=run,question=q['question_id'],category=q['category'],accuracy=q['answer_accuracy_auto'],original=old['answer_accuracy_auto'],exact=int(q['answer_checks_correct']==q['answer_checks_total']),context=c.get('country_valid_auto'),latency=log['latency_seconds'],fallback=bool(log.get('used_fallback_synthesis')),external_fail=sum(not t.get('success') and 'duplicate' not in str(t.get('error','')).lower() for t in log.get('tool_calls',[])),calls=len(log.get('tool_calls',[]))))
   logs.append(log)
with (OUT/'observaciones_v2.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
by=defaultdict(dict)
for r in rows:by[(r['question'],r['model'])].setdefault('acc',[]).append(r['accuracy'])
questions=sorted(set(r['question'] for r in rows))
diff=np.array([np.mean(by[q,'gemini']['acc'])-np.mean(by[q,'openai']['acc']) for q in questions])
rng=np.random.default_rng(20261024);B=20000
boot=np.mean(diff[rng.integers(0,30,size=(B,30))],axis=1)
ci=np.quantile(boot,[.025,.975]);w=wilcoxon(diff,zero_method='wilcox',alternative='two-sided',method='auto')
# paired bootstrap for median question-level latency difference
lat=defaultdict(list)
for r in rows:lat[r['question'],r['model']].append(r['latency'])
ld=np.array([np.median(lat[q,'gemini'])-np.median(lat[q,'openai']) for q in questions]);lb=np.mean(ld[rng.integers(0,30,size=(B,30))],axis=1)
per_run=[]
for m in ('openai','gemini'):
 for run in range(1,6):
  z=[r for r in rows if r['model']==m and r['run']==run]
  per_run.append(dict(model=m,run=run,accuracy_macro=np.mean([r['accuracy'] for r in z]),exact=sum(r['exact'] for r in z),median_latency=np.median([r['latency'] for r in z]),external_failed_calls=sum(r['external_fail'] for r in z),fallbacks=sum(r['fallback'] for r in z)))
with (OUT/'corridas_v2.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=per_run[0]);wr.writeheader();wr.writerows(per_run)
# median per question across repetitions
med=[]
for q in questions:
 med.append(dict(question=q,openai_median=np.median(by[q,'openai']['acc']),gemini_median=np.median(by[q,'gemini']['acc']),difference=np.median(by[q,'gemini']['acc'])-np.median(by[q,'openai']['acc'])))
with (OUT/'medianas_por_pregunta.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=med[0]);wr.writeheader();wr.writerows(med)
# eight/nine directional checks on USA responses: count responses with at least one direction check correct
usa=[]
for p in sorted((BASE/'experiment_logs').glob('run_openai_*scored_v2.json')):
 obj=json.loads(p.read_text());run=int(sorted((BASE/'experiment_logs').glob('run_openai_*scored_v2.json')).index(p))+1
 for q in obj['per_question']:
  if q['question_id'] not in ('chained_01','chained_06','chained_07','chained_09','chained_10'):continue
  dc=[c for c in q['answer_checks'] if 'direction' in c['name'].lower() or c['name'].lower() in ('trend','change')]
  usa.append((run,q['question_id'],len(dc),sum(bool(c['correct']) for c in dc)))
usa_pass=sum(any_correct>0 for _,_,n,any_correct in usa)
no4=[r for r in rows if not(r['model']=='openai' and r['run']==4)]
md=f'''# AIDA 2026 — inferencia exploratoria con scoring v2\n\n## Alcance\nCinco corridas de octubre por modelo; 30 preguntas emparejadas, 300 respuestas. Septiembre excluido. Scoring v2 aplicado idénticamente. No se interpreta el p-valor como confirmatorio.\n\n## Resultados principales\n\n| Metric | GPT-4o | Gemini 3.6 Flash |\n|---|---:|---:|\n| Mean macro accuracy (v2) | {np.mean([r['accuracy'] for r in rows if r['model']=='openai']):.4f} | {np.mean([r['accuracy'] for r in rows if r['model']=='gemini']):.4f} |\n| Exact answers | {sum(r['exact'] for r in rows if r['model']=='openai')}/150 | {sum(r['exact'] for r in rows if r['model']=='gemini')}/150 |\n| Median response latency (s) | {np.median([r['latency'] for r in rows if r['model']=='openai']):.2f} | {np.median([r['latency'] for r in rows if r['model']=='gemini']):.2f} |\n\nPaired mean accuracy difference (Gemini−GPT): **{np.mean(diff)*100:.2f} pp**; 95% question-cluster bootstrap percentile CI **[{ci[0]*100:.2f}, {ci[1]*100:.2f}] pp**, 20,000 resamples, fixed seed 20261024. Exploratory Wilcoxon W={w.statistic:.3f}, p={w.pvalue:.6f}; 30 question-level paired differences.\n\nPaired mean difference of per-question median latency (Gemini−GPT): **{np.mean(ld):.2f} s**, bootstrap CI **[{np.quantile(lb,.025):.2f}, {np.quantile(lb,.975):.2f}] s**. This estimand differs from the pooled response median.\n\n## Sensitivity\n\nGPT-4o v2 macro accuracy excluding run 4: **{np.mean([r['accuracy'] for r in no4 if r['model']=='openai'])*100:.2f}%** (120 responses). Main result retains run 4. This unbalanced exclusion is diagnostic, not the primary paired comparison.\n\n## USA direction checks\n\nOf the 25 GPT-4o responses for the five country-unspecified chained questions, **{usa_pass}** had at least one direction check scored correct by v2. A response may have no direction check; the definition here is 'at least one correctly passed direction-labelled check', not 'all checks passed'. See source scoring fields.\n\n## Interpretation and limitations\n\nThese results are exploratory and conditional on the 30 frozen questions and five October runs. Bootstrap resamples questions, not individual responses. Repeated runs are not independent experimental questions. Automated factual accuracy and contextual country validity are separate dimensions; a correct trend can refer to the wrong country. Tool-selection accuracy applies only to the 17 questions per run with a defined unique expected first tool. Historical human grounding annotations should not be generalized to October runs. A 20-pp interval-width threshold was discussed post hoc and was not preregistered.\n\n## Suggested Results text (English)\n\nAcross 30 benchmark questions, each repeated five times per model, Gemini 3.6 Flash achieved higher automatically scored factual accuracy than GPT-4o under the revised rubric. The question-paired mean difference was {np.mean(diff)*100:.2f} percentage points (95% question-bootstrap CI {ci[0]*100:.2f} to {ci[1]*100:.2f}). The analysis was exploratory and does not establish general superiority beyond the evaluated tasks. Contextual country validity was assessed separately from factual-direction checks.\n\n## Reproducibility\n\nSource: ten October JSONL logs; five v2 JSON score files per provider; 30 frozen question IDs. Script: `analisis_inferencial_v2.py`. Fixed bootstrap seed 20261024, B=20000. The original scoring logic remains available in `original_logic_per_question` for paired sensitivity analyses.\n'''
(OUT/'INFORME_INFERENCIAL_V2.md').write_text(md)
print(md[:2300]);print('USA rows',usa)
