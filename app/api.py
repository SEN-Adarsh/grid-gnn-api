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
    'deployment_stage': 'Stage 1: Validation Reference Implementation (IES v0.4 Section 6.2 Accelerator, Section 7 Scaling Up, Pages 16-17)',
    'building_blocks_used': [
        'BB1_Identity (Meter & DT Unique Asset IDs)',
        'BB2_Registries_Directories (DER Visibility, A1.2)',
        'BB3_Interaction_Protocols (Hash-bound /score Request-Ack)',
        'BB4_Energy_Data_Exchange_Public (Policy Pack: physics + precision thresholds, A1.6)',
        'BB4_Energy_Data_Exchange_Permissioned (15-min AMI & DT Telemetry Schema)',
        'BB5_Energy_Credentials (Cryptographic Audit Log & Model SHA256, A1.7)'],
    'data_residency': 'local_discom_retained_no_central_db',
    'guiding_principles': {
        'data_federation': 'Raw AMI/SCADA stays in host DISCOM MDMS/ADMS; stateless interface only (Page 10)',
        'ai_readiness': 'Standardised 15-min interval arrays & verifiable audit receipts (Page 10)',
        'dpdp_npsdp_compliance': 'Zero PII retention; purpose-bound NTL & loss attribution scoring (Pages 5, 20)',
        'inclusivity': 'Zero-label cold-start default (M0/M1) for newly metered RDSS feeders (Page 10)',
    },
    'ies_scenarios_addressed': ['Scenario 2: Inconsistent asset/GIS identifiers across systems (Page 7)', 'Scenario 4: DISCOM trapped in integration loops (Pages 7-8)'],
    'ies_risks_addressed': ['Risk 2: Uneven AMI deployment (Pages 20-21)', 'Risk 5: Quality at scale / GIS mapping inconsistency (Pages 20-21)'],
    'ies_annexures': ['A1.2 DER Visibility Registry (Page 22)', 'A1.4 Consumer Energy Passport (Page 23)', 'A1.6 Policy Pack (Pages 23-24)', 'A1.7 DISCOM-Regulator Filing Object (Pages 13, 24)', 'A2.1 Supported Host Systems (Page 25)'],
    'supported_host_systems': ['MDMS (AMI)', 'GIS', 'ADMS/SCADA', 'CIS/Billing', 'RDSS MIS Portal (Annexure A2.1, Page 25)'],
    'openforge_api_setu': 'APIs published via API Setu; code hosted on OpenForge (Section 11, Page 20)',
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
        'building_blocks_matrix_6col': [
            {'ies_column': 'WHO - Identity & Addressing', 'role': 'Core Dependency', 'mapping': 'meter_id and dt_id are federated asset & consumer identifiers (A1.4 Consumer Energy Passport, Page 23); prevents cross-system GIS-to-MDMS indexing mismatches (Scenario 2, Page 7)'},
            {'ies_column': 'WHERE - Registries & Directories', 'role': 'Supporting Role', 'mapping': 'Discovers authorised DISCOM MDMS/GIS endpoints and DER Visibility Registries (A1.2, Page 22) so prosumer reverse flows (negative_fraction) are recognised, not misclassified'},
            {'ies_column': 'HOW - Interaction Protocols', 'role': 'Core Dependency', 'mapping': 'Stateless POST /score and GET /health OpenAPI request/acknowledgement flow; ready for API Setu / OpenForge (Section 11, Page 20)'},
            {'ies_column': 'WHAT (Public) - Public Energy Data Exchange', 'role': 'Supporting Role', 'mapping': 'Policy Pack (A1.6, Pages 23-24): AC power-flow conservation (<=1e-5 tolerance, pandapower agreement 6.74e-12 V) + 70%/50% precision thresholds with abstention; config_hash 6e60f7bd16aff2ef'},
            {'ies_column': 'WHAT (Private) - Permissioned Energy Data Exchange', 'role': 'Core Dependency', 'mapping': 'Purpose-bound 15-min AMI interval arrays (readings[]) and DT telemetry (dt_readings[]) under Data Federation + NPSDP/DPDP Act 2023 (Pages 5, 10, 20); zero PII retained'},
            {'ies_column': 'PROOF - Energy Credentials', 'role': 'Core Dependency', 'mapping': 'Tamper-evident audit_log.jsonl stamped with config_hash + model_sha256, exportable as IES A1.7 DISCOM-Regulator Filing Object (Pages 13, 24) for SERC loss-truing and Electricity Act s.126/s.135'},
        ],
        'building_blocks': [
            {'id': 'BB1', 'name': 'Identity',
             'implementation': 'meter_id and dt_id in every request/response are federated asset & consumer identifiers for cross-referencing MDM, GIS and billing without a central database'},
            {'id': 'BB2', 'name': 'Registries & Directories',
             'implementation': 'DER Visibility Registry (A1.2, Page 22) for prosumer reverse-flow recognition; DISCOM MDMS/GIS endpoint discovery'},
            {'id': 'BB3', 'name': 'Interaction Protocols',
             'implementation': 'stateless POST /score request/acknowledgement flow; every response bound to config_hash and model_sha256'},
            {'id': 'BB4', 'name': 'Energy Data Exchange (Public + Permissioned)',
             'implementation': 'Public: Policy Pack (A1.6) physics + precision thresholds. Permissioned: 15-min interval AMI (readings[]) and DT telemetry (dt_readings[]) schema; raw consumer data stays inside the host utility'},
            {'id': 'BB5', 'name': 'Energy Credentials',
             'implementation': 'append-only audit_log.jsonl entries carry model/config provenance for regulatory defence (Electricity Act s.126/s.135); exportable as A1.7 Filing Object'},
        ],
        'data_exchange_schema': {
            'request': {
                'readings': ['meter_id', 'ts', 'kwh', 'kvarh', 'voltage_v', 'event_flags', 'is_missing'],
                'dt_readings': ['dt_id', 'ts', 'kwh_in', 'kvarh_in', 'v_lv'],
                'control': ['model', 'as_of_interval', 'scenario', 'severity', 'tamper_event', 'imputation']},
            'response': ['meters[] (rank, probability_simulated_theft, reason_codes, recommended_action, assigned_owner, target_sla, rdss_mis_code)',
                         'dts[] (4-way loss_attribution, work order)', 'config_hash', 'model_sha256'],
            'interval': '15 minutes (Asia/Kolkata)'},
        'deployment_stage': 'Stage 1: Validation Reference Implementation (IES v0.4 Section 6.2, Section 7, Pages 16-17)',
        'ies_scenarios_addressed': ['Scenario 2: Inconsistent asset/GIS identifiers (Page 7)', 'Scenario 4: DISCOM trapped in integration loops (Pages 7-8)'],
        'ies_risks_addressed': ['Risk 2: Uneven AMI deployment (Pages 20-21)', 'Risk 5: Quality at scale / GIS mapping inconsistency (Pages 20-21)'],
        'ies_annexures': ['A1.2 DER Visibility (Page 22)', 'A1.4 Consumer Energy Passport (Page 23)', 'A1.6 Policy Pack (Pages 23-24)', 'A1.7 DISCOM-Regulator Filing Object (Pages 13, 24)', 'A2.1 Supported Host Systems (Page 25)'],
        'central_database': False,
        'automated_enforcement': False,
    }

@app.post('/score')
def score(body:ScoreRequest):
    try:return scorer().score(body.model_dump(mode='json'))
    except (ValueError,KeyError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
