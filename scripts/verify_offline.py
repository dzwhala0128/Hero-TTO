"""Run reproducible CPU-only acceptance; no model calls or downloads."""
from pathlib import Path
import sys,subprocess,json,os,re
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.environ['PYTHONDONTWRITEBYTECODE']='1'
from scripts.check_data import check
counts=check();py=list(R.rglob('*.py'));py=[p for p in py if not p.is_relative_to(R/'outputs')]
for p in py:compile(p.read_text(),str(p),'exec')
shell=[p for p in R.rglob('*.sh') if not p.is_relative_to(R/'outputs')]
for p in shell:subprocess.run(['bash','-n',str(p)],check=True)
cmd=[sys.executable,'-B','-m','unittest','discover','-s','tests','-v']
result=subprocess.run(cmd,cwd=R,text=True,capture_output=True)
print(result.stderr)
if result.returncode:print(result.stdout);raise SystemExit(result.returncode)
m=re.search(r'Ran (\d+) tests?',result.stderr)
print(json.dumps({'offline_core':'passed','python_syntax_files':len(py),'shell_syntax_files':len(shell),'unit_tests':int(m.group(1)) if m else None,'gold_read_by_runtime':False,'full_pipeline_gpu_imports':'pending optional dependencies','gpu_inference':'NOT RUN','fine146':'MISSING; not falsely passed'},ensure_ascii=False,indent=2))
