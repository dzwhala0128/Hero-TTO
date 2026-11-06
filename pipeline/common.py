from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ['TEXTGRAD_DISABLE_LOGGING']='1'
sys.path.insert(0,str(ROOT/'textgrad_native'))
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(name):return read(ROOT/'configs'/name)
def safe_output(value):
 p=(ROOT/value).resolve()
 if not p.is_relative_to(ROOT/'outputs'):raise ValueError('Output must be inside package outputs/')
 return p
