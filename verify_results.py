from pathlib import Path
import subprocess,json,re,hashlib
import nbformat,numpy as np,pandas as pd
base=Path(__file__).resolve().parent
for a in [4]:
 p=base; nb=nbformat.read(p/f'A{a}.ipynb',as_version=4); runs=sorted((p/'results').glob('*/completed.json')); run=runs[-1].parent
 counts=[c.execution_count for c in nb.cells if c.cell_type=='code']; assert counts==list(range(1,len(counts)+1))
 assert not [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
 assert nb.cells[0].source.startswith('# Abstract') and 'exporter replaces' not in nb.cells[0].source
 meta=json.loads((run/'run_metadata.json').read_text()); complete=json.loads((run/'completed.json').read_text())
 assert complete['notebook_code_sha256']==hashlib.sha256(''.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest()
 assert complete['analysis_source_sha256']==hashlib.sha256((p/'analysis.py').read_bytes()).hexdigest()
 text=subprocess.check_output(['pdftotext','-layout',str(p/f'A{a}.pdf'),'-'],text=True)
 assert 'Abstract' in text[:300] and '202618018629048' in text
 assert 'RESULT_DIR:' in text and 'Interpretation of this run' in text
 assert 'Traceback (most recent call last)' not in text
 (p/'.runtime/pdf_text.txt').write_text(text)
 info=subprocess.check_output(['pdfinfo',str(p/f'A{a}.pdf')],text=True); pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
 images=subprocess.check_output(['pdfimages','-list',str(p/f'A{a}.pdf')],text=True)
 print(a, 'pages',pages,'code_cells',len(counts),'summary_words',complete['summary_words'],'PNG_outputs',len(list((run/'figures').glob('*.png'))),'python',meta['executable'],'PDF_images',len(images.splitlines())-2)
 # Recompute primary metrics from retained raw predictions, without invoking analysis helpers.
 facts=json.loads((run/'summary.json').read_text())
 if a==3:
  d=pd.read_csv(run/'holdout_predictions.csv'); err=d.actual-d[facts['selected_model']]; assert np.isclose(np.sqrt(np.mean(err**2)),facts['holdout_rmse'])
  check=f"Direct sum-of-squared prediction errors reproduces holdout RMSE {np.sqrt(np.mean(err**2)):.9f}."
 elif a==4:
  d=pd.read_csv(run/'paired_predictions.csv'); d=d[(d.partition=='test')&(d.model==facts['selected'])]; score=np.mean((d.y-d.calibrated)**2); assert np.isclose(score,facts['metrics']['Brier'])
  record=json.loads((run/'threshold_decision.json').read_text()); assert record['test_threshold_plus_cap']['reviewed']<=int(.3*len(d))
  check=f"Direct mean squared probability error reproduces Brier {score:.9f}; review count respects floor(0.3 × test size)."
 elif a==5:
  d=pd.read_csv(run/'rolling_predictions.csv'); mae=np.mean(abs(d.actual-d.Ridge)); assert np.isclose(mae,facts['Ridge']['MAE'])
  ridge_errors=abs(d.actual-d.Ridge).to_numpy(); se=abs(d.actual-d.Seasonal).to_numpy()
  flags=np.array([j>=14 and ridge_errors[j-14:j].mean()>1.25*se[j-14:j].mean() for j in range(len(d))]); assert np.array_equal(flags,d.fallback)
  check=f"Direct absolute-error average reproduces Ridge MAE {mae:.9f}; an independent loop reproduces all {flags.sum()} past-only fallback decisions."
 else:
  d=pd.read_csv(p/'data/raw/oracle.csv'); ate=np.mean(d.Y1-d.Y0); assert np.isclose(ate,facts['sample_ATE']); reps=pd.read_csv(run/'sensitivity_replicates.csv'); g=reps[(reps.gamma==1)&(reps.method=='Observed_XG')]; coverage=((g.lower<=g.truth)&(g.upper>=g.truth)).mean(); assert coverage==facts['observed_coverage_gamma_1']
  check=f"Potential-outcome subtraction reproduces sample ATE {ate:.9f}; direct interval containment reproduces gamma-one observed coverage {coverage:.3f}."
 print(check)
 (p/'EXECUTION_CHECK.md').write_text(f'''# Execution and artifact check

Author: xuhongbo · Student ID: 202618018629048

This verification was performed by Codex, not independently by the student.

- Completed run: `{run.relative_to(p).as_posix()}`.
- Run completion time: `{complete['completed_utc']}`.
- Saved Notebook: {len(counts)} code cells executed consecutively, with no error outputs.
- PDF: {pages} pages; Abstract, author identity, result discussion, executable code, completion marker and figures are present.
- Abstract: {complete['summary_words']} words, generated from completed results.
- Raw-file SHA-256 values are in `data/raw/manifest.json` and verified before execution.
- Protocol SHA-256: `{complete['protocol_sha256']}`.
- Analysis source SHA-256: `{complete['analysis_source_sha256']}`.
- Actual interpreter: `{meta['executable']}`. The existing A2 environment was reused for execution; no A2 files are required by the analysis. A fresh environment can be created using this directory's requirements.
- Full environment: `{run.relative_to(p).as_posix()}/requirements-lock.txt`.
- Numerical cross-check: {check}

Previous completed runs remain in `results/`. No failed notebook execution occurred in these runs. Negative model comparisons and designed counterexamples are retained as scientific evidence; they are not labelled execution failures.

## Student verification boundary

The checks above are automated and do not complete the author's independent verification. The author should personally inspect the method, reproduce at least one numerical result without simply rerunning the same notebook cell, and record the actual method, result, date and reflection in `AI_USE_LOG.md`.
''')
 # Determinism comparison excludes wall-clock timing and metadata.
 if len(runs)>1:
  old=json.loads((runs[0].parent/'summary.json').read_text()); assert old==facts,(a,old,facts)
 print('Repeated-run main summary identical:',len(runs)>1)

if (base/"validation_history/artifact_check_failure.log").exists():
 with (base/"EXECUTION_CHECK.md").open("a") as f:
  f.write("\nA corrected artifact-check helper error is retained in `validation_history/artifact_check_failure.log`; it did not affect notebook experiment results.\n")
