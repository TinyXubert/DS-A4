"""Execute and retain outputs, including failures. Run from this directory."""
from pathlib import Path
import os,datetime,traceback,json
base=Path(__file__).resolve().parent
os.chdir(base)
for var,sub in [('TMPDIR','tmp'),('JUPYTER_RUNTIME_DIR','jupyter'),('JUPYTER_CONFIG_DIR','jupyter'),('IPYTHONDIR','ipython')]:
    p=base/'.runtime'/sub; p.mkdir(parents=True,exist_ok=True); os.environ.setdefault(var,str(p))
import nbformat
from nbclient import NotebookClient
number=4
path=base/f'A{number}.ipynb'; nb=nbformat.read(path,as_version=4)
for c in nb.cells:
    if c.cell_type=='code': c.outputs=[]; c.execution_count=None
try:
    NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(base)}}).execute()
except BaseException:
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    nbformat.write(nb,base/f'failed_{stamp}.ipynb')
    (base/f'failed_{stamp}.log').write_text(traceback.format_exc())
    raise
else:
    nbformat.write(nb,path)
    print('Executed:',path)
