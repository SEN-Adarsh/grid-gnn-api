from __future__ import annotations
from datetime import datetime
from functools import lru_cache
from typing import Literal
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,ConfigDict,Field
from .scoring import Scorer

class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)

class Reading(StrictModel):
    meter_id:str
    ts:datetime
    kwh:float|None=None
    kvarh:float=0.
    voltage_v:float|None=Field(default=None,ge=0,le=400)
    event_flags:int=Field(default=0,ge=0,le=255)
    is_missing:bool=False

class DTReading(StrictModel):
    dt_id:int=Field(ge=0)
    ts:datetime
    kwh_in:float
    kvarh_in:float|None=None
    v_lv:float|None=Field(default=None,ge=0,le=400)

class ScoreRequest(StrictModel):
    readings:list[Reading]=Field(default_factory=list,max_length=20000)
    dt_readings:list[DTReading]=Field(default_factory=list,max_length=20000)
    model:Literal['M0','M1','M2','M3','M4']|None=None
    as_of_interval:int|None=Field(default=None,ge=1)
    scenario:Literal['none','vacancy','theft','upstream_hooking','ami_dropout']='none'
    meter_id:str|None=None
    severity:float=Field(default=.7,gt=0,lt=1)
    tamper_event:bool=False
    imputation:bool=True

IES_V0_4 = {
    'architecture': 'federated_stateless_interface',
    'building_blocks_used': [
        'BB1_Identity (Meter & DT Unique Asset IDs)',
        'BB3_Interaction_Protocols (Hash-bound /score Request-Ack)',
        'BB4_Energy_Data_Exchange (15-min AMI & DT Telemetry Schema)',
        'BB5_Energy_Credentials (Cryptographic Audit Log & Model SHA256)'],
    'data_residency': 'local_discom_retained_no_central_db',
}

app=FastAPI(title='Grid-GNN | Inspection Decision Support',version='0.1.0',
    description='Local synthetic PoC. Known demo assets have stored baseline history. No enforcement actions.')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['*'])

@lru_cache(maxsize=1)
def scorer():return Scorer()

@app.get('/health')
def health():
    service=scorer()
    obs=service.context
    total=int(obs.kwh.shape[1])
    minimum=int((obs.cfg['baseline_days']+7)*96)
    return {'status':'ok','profile_source':service.selection.get('profile_source','synthetic_fallback'),
            'model':service.selection['model'],
            'models':sorted(service.models),
            'models_note':service.selection.get('training_data',''),
            'scoring_window':{'min':minimum,'max':total},
            'config_hash':service.selection['config_hash'],'automated_enforcement':False,
            'ies_v0_4_alignment':IES_V0_4}

@app.get('/ies/manifest')
def ies_manifest():
    """Machine-readable India Energy Stack (IES) v0.4 alignment manifest."""
    return {
        'standard': 'India Energy Stack (IES) v0.4',
        'alignment': IES_V0_4,
        'building_blocks': [
            {'id': 'BB1', 'name': 'Identity',
             'implementation': 'meter_id and dt_id in every request/response are federated asset & consumer identifiers for cross-referencing MDM, GIS and billing without a central database'},
            {'id': 'BB3', 'name': 'Interaction Protocols',
             'implementation': 'stateless POST /score request/acknowledgement flow; every response bound to config_hash and model_sha256'},
            {'id': 'BB4', 'name': 'Energy Data Exchange',
             'implementation': '15-minute interval AMI (readings[]) and DT telemetry (dt_readings[]) schema; raw consumer data stays inside the host utility'},
            {'id': 'BB5', 'name': 'Energy Credentials',
             'implementation': 'append-only audit_log.jsonl entries carry model/config provenance for regulatory defence (Electricity Act s.126/s.135)'},
        ],
        'data_exchange_schema': {
            'request': {
                'readings': ['meter_id', 'ts', 'kwh', 'kvarh', 'voltage_v', 'event_flags', 'is_missing'],
                'dt_readings': ['dt_id', 'ts', 'kwh_in', 'kvarh_in', 'v_lv'],
                'control': ['model', 'as_of_interval', 'scenario', 'severity', 'tamper_event', 'imputation']},
            'response': ['meters[] (rank, probability_simulated_theft, reason_codes, recommended_action, assigned_owner, target_sla, rdss_mis_code)',
                         'dts[] (4-way loss_attribution, work order)', 'config_hash', 'model_sha256'],
            'interval': '15 minutes (Asia/Kolkata)'},
        'central_database': False,
        'automated_enforcement': False,
    }

@app.post('/score')
def score(body:ScoreRequest):
    try:return scorer().score(body.model_dump(mode='json'))
    except (ValueError,KeyError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
