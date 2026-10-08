"""End-to-end local HTTP smoke and honest, labelled demo examples.

Evaluation-only labels are read here to document example eligibility. The API
and dashboard never import this module or load labels.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import pandas as pd
from ..common import write_json


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def wait_url(url,process):
    until=time.monotonic()+40
    while time.monotonic()<until:
        if process.poll() is not None:raise RuntimeError(f'Server exited with {process.returncode}')
        try:
            with urllib.request.urlopen(url,timeout=2) as response:
                return response.status,response.read()
        except (OSError,TimeoutError):time.sleep(.2)
    raise TimeoutError('Local HTTP service failed to become ready')


def validate_app(cfg,config_path):
    out=Path(cfg['output_dir'])
    os.environ['GRID_CONFIG']=str(Path(config_path).resolve())
    from app.scoring import Scorer,timestamp_for_interval
    service=Scorer()
    began=time.perf_counter()
    base=service.score({},write_audit=False)
    truth=pd.read_csv(Path(cfg['artifacts_dir'])/'demo_truth_evaluation_only.csv').set_index('meter_id')
    eligible_v=[r for r in base['meters'] if truth.loc[r['meter_id'],'archetype']=='vacant' and not r['inspection_flag']]
    eligible_t=[r for r in base['meters'] if bool(truth.loc[r['meter_id'],'is_theft']) and r['inspection_flag']]
    choose=lambda rows:sorted(rows,key=lambda r:r['meter_id'])[0] if rows else None
    vacant=choose(eligible_v);theft=choose(eligible_t)
    examples={'model':base['model'],'source':'first configured seed, first two held-out DTs; deterministic first eligible pseudonym',
        'purpose':'Illustrative examples only; not estimates of performance or a best-seed selection',
        'end_interval':base['as_of_interval'],'vacancy_not_flagged':vacant,'theft_flagged':theft,
        'eligible_vacancy_count':len(eligible_v),'eligible_flagged_theft_count':len(eligible_t),
        'dt_estimates':base['dts'],'baseline_score_seconds':base['runtime_seconds'],
        'candidate_kwh_warning':'Posterior times expected-load allocation of DT residual, not verified theft or revenue'}
    scenario_rows=[]
    for name in ['none','vacancy','theft','upstream_hooking','ami_dropout']:
        r=base if name=='none' else service.score({'scenario':name},write_audit=False)
        scenario_rows.append({'scenario':name,'flag_count':sum(x['inspection_flag'] for x in r['meters']),
            'total_unexplained_kwh':sum(x['unexplained_kwh'] for x in r['dts']),
            'seconds':r['runtime_seconds']})
    examples['scenario_smoke']=scenario_rows
    write_json(out/'demo_examples.json',examples)
    results=json.loads((out/'results.json').read_text())
    results['demo_examples']=examples
    write_json(out/'results.json',results)

    api_port=free_port();ui_port=free_port()
    children=[]
    with (out/'api_server.log').open('w',encoding='utf-8') as api_log,(out/'dashboard_server.log').open('w',encoding='utf-8') as ui_log:
        try:
            api=subprocess.Popen([sys.executable,'-m','uvicorn','app.api:app','--host','127.0.0.1','--port',str(api_port)],stdout=api_log,stderr=subprocess.STDOUT)
            children.append(api)
            ui=subprocess.Popen([sys.executable,'-m','streamlit','run','app/dashboard.py','--server.address','127.0.0.1','--server.port',str(ui_port),'--server.headless','true'],stdout=ui_log,stderr=subprocess.STDOUT)
            children.append(ui)
            code,body=wait_url(f'http://127.0.0.1:{api_port}/health',api)
            health=json.loads(body)
            ui_code,_=wait_url(f'http://127.0.0.1:{ui_port}/_stcore/health',ui)
            end=service.context.kwh.shape[1]
            batch={'dt_readings':[{'dt_id':0,'ts':timestamp_for_interval(end-1),
                                  'kwh_in':float(service.context.dt_kwh[0,end-1]+1)}]}
            request=urllib.request.Request(f'http://127.0.0.1:{api_port}/score',
                data=json.dumps(batch).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=30) as response:
                post_code=response.status;scored=json.load(response)
            before=next(x for x in base['dts'] if x['dt_id']==0)['unexplained_kwh']
            after=next(x for x in scored['dts'] if x['dt_id']==0)['unexplained_kwh']
            assert .99<after-before<1.01,'HTTP batch update did not reach the scorer'
            assert len(scored['meters'])==len(service.context.meters)
            validation={'status':'passed','api_health_http':code,'score_http':post_code,
                'streamlit_health_http':ui_code,'meter_count':len(scored['meters']),
                'observed_dt_residual_increment_kwh':after-before,'api_health':health,
                'seconds':time.perf_counter()-began,
                'scope':'Real local HTTP servers, batch scoring; all dashboard pages and replay button separately checked by Streamlit AppTest',
                'visual_limit':'No pixel screenshot test or external hosting performed'}
            write_json(out/'app_validation.json',validation)
        finally:
            for child in children:
                child.terminate()
            for child in children:
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    return validation
