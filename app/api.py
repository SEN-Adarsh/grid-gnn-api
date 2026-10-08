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

app=FastAPI(title='Grid-GNN | Inspection Decision Support',version='0.1.0',
    description='Local synthetic PoC. Known demo assets have stored baseline history. No enforcement actions.')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['*'])

@lru_cache(maxsize=1)
def scorer():return Scorer()

@app.get('/health')
def health():
    service=scorer()
    return {'status':'ok','profile_source':'synthetic_fallback','model':service.selection['model'],
            'models':sorted(service.models),
            'models_note':service.selection.get('training_data',''),
            'config_hash':service.selection['config_hash'],'automated_enforcement':False}

@app.post('/score')
def score(body:ScoreRequest):
    try:return scorer().score(body.model_dump(mode='json'))
    except (ValueError,KeyError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
