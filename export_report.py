"""Export saved, successfully executed notebook to Markdown, HTML and optional PDF."""
from pathlib import Path
import argparse,json,re,os,hashlib
import nbformat
from nbconvert import HTMLExporter,MarkdownExporter
base=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(); parser.add_argument('--pdf',action='store_true'); args=parser.parse_args()
a=json.loads((base/'protocol.json').read_text())['assignment']; path=base/f'A{a}.ipynb'
nb=nbformat.read(path,as_version=4)
counts=[c.execution_count for c in nb.cells if c.cell_type=='code']
assert counts and all(x is not None for x in counts) and counts==sorted(set(counts)), 'Execute all cells in order first.'
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
text='\n'.join(o.get('text','') for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='stream')
run=base/re.findall(r'RESULT_DIR:\s*(results/\S+)',text)[-1]
assert (run/'completed.json').exists()
completed=json.loads((run/'completed.json').read_text())
for filename,key in [('protocol.json','protocol_sha256'),('data/raw/manifest.json','manifest_sha256'),('analysis.py','analysis_source_sha256')]:
    assert hashlib.sha256((base/filename).read_bytes()).hexdigest()==completed[key], 'Changed since execution: '+filename
code=''.join(c.source for c in nb.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==completed['notebook_code_sha256'], 'Notebook code changed since execution; rerun first.'
abstract=(run/'ABSTRACT.md').read_text()
for c in nb.cells:
    if c.get('id')=='abstract': c.source=abstract
# Keep the author's current verification/reflection entry in every report export.
log_path=base/'AI_USE_LOG.md'
log_text=log_path.read_text(encoding='utf-8')
headings=list(re.finditer(r'(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$',log_text))
sections=[]
for i,heading in enumerate(headings):
    title=heading.group(2).strip().lower()
    if not (title.startswith('student independent verification') or title.startswith('reflection')):
        continue
    end=len(log_text)
    for later in headings[i+1:]:
        if len(later.group(1))<=len(heading.group(1)):
            end=later.start(); break
    sections.append(log_text[heading.start():end].strip())
assert sections, 'No student verification/reflection section found in AI_USE_LOG.md.'
cell_id='student-verification-reflection'
nb.cells=[c for c in nb.cells if c.get('id')!=cell_id]
nb.cells.append(nbformat.v4.new_markdown_cell(
    '<div style="break-before: page"></div>\n\n'
    + '\n\n'.join(sections) + '\n',
    id=cell_id,
    metadata={'source_file':'AI_USE_LOG.md','source_sha256':hashlib.sha256(log_path.read_bytes()).hexdigest(),
              'content_policy':'Verbatim author-provided verification/reflection; no execution or completion claim added.'}))
nbformat.write(nb,path)
(base/'ABSTRACT.md').write_text(abstract)
(base/'SUBMISSION_SUMMARY.md').write_text(abstract+'\nAuthor: xuhongbo · Student ID: 202618018629048\n\nAssignment: DS-A'+str(a)+'\n\nUse the summary above in the platform record alongside the Google Drive link and a fixed GitHub commit/tag URL after publishing. No remote publication is performed by this exporter.\n')
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
css='<style>@media print {pre,code{white-space:pre-wrap!important;overflow-wrap:anywhere!important} .jp-OutputArea-output{overflow:visible!important} .jp-Cell{break-inside:auto!important} img{max-width:100%!important} table{font-size:9px!important} } body{font-family:Arial,sans-serif}</style>'
html=html.replace('</head>',css+'</head>'); (base/f'A{a}.html').write_text(html)
md,res=MarkdownExporter().from_notebook_node(nb,resources={'output_files_dir':'report_assets','unique_key':f'A{a}'})
(base/f'A{a}_REPORT.md').write_text(md)
for name,data in res.get('outputs',{}).items():
    p=base/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
if args.pdf:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(); page.route('https://**/*',lambda route:route.abort()); page.route('http://**/*',lambda route:route.abort())
        page.goto((base/f'A{a}.html').as_uri(),wait_until='load'); page.evaluate('document.fonts.ready'); page.emulate_media(media='print')
        page.pdf(path=str(base/f'A{a}.pdf'),format='A4',print_background=True,margin={'top':'12mm','bottom':'12mm','left':'10mm','right':'10mm'})
        browser.close()
print('Exported assignment',a,'from',run.name)
