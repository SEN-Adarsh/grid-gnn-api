from __future__ import annotations
import copy
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import joblib
import numpy as np
import pandas as pd
import torch
from src.common import load_config,jsonable
from src.features.build import build_features,causal_impute,reason_codes
from src.models.explanations import tree_group_shap
from src.sim.physics import meter_power_to_nodes,sweep,transformer_loss

ROOT=Path(__file__).resolve().parents[1]
START=pd.Timestamp('2026-06-01',tz='Asia/Kolkata')
START_DATETIME=START.to_pydatetime()
_LOCK=threading.Lock()


def timestamp_for_interval(index):
    return (START_DATETIME+timedelta(minutes=15*int(index))).isoformat()


def configured_paths():
    cfg=load_config(ROOT/os.environ.get('GRID_CONFIG','configs/final.yaml'))
    return cfg,ROOT/cfg['artifacts_dir'],ROOT/cfg['output_dir']


def scenario(obs,name,meter_index,severity=.7,as_of=None,tamper_event=False):
    """Public-input, reversible counterfactuals. Never reads hidden labels."""
    if name=='none':return obs,[]
    changed=copy.deepcopy(obs)
    end=as_of or obs.kwh.shape[1]
    begin=max(obs.cfg['baseline_days']*96,end-7*96)
    part=slice(begin,end)
    g=int(obs.meters.dt_id.iloc[meter_index]);m=obs.cfg['meters_per_dt']
    cfg=obs.cfg
    if name in ('vacancy','theft'):
        changed.kwh[meter_index,part]*=1-severity
        changed.kvarh[meter_index,part]*=1-severity
        changed.events[meter_index,part]&=np.uint8(254)
        if name=='theft' and tamper_event:
            available=np.flatnonzero(~changed.missing[meter_index,part])
            if len(available):changed.events[meter_index,begin+available[-1]]|=1
        if name=='vacancy':
            ids=slice(g*m,(g+1)*m);topo=obs.topologies[g]
            dt_idx=obs.meters.dt_id.to_numpy()
            old=causal_impute(obs.kwh[:,:end],obs.dt_voltage[:,:end],dt_idx)[ids,part]/.25
            new=old.copy();new[meter_index-g*m]*=1-severity
            qratio=np.tan(np.arccos(cfg['power_factor']))
            before=meter_power_to_nodes(topo,old,obs.meters.phase.to_numpy()[ids],qratio)
            after=meter_power_to_nodes(topo,new,obs.meters.phase.to_numpy()[ids],qratio)
            vb,lb,_=sweep(topo,*before,obs.dt_voltage[g,part],cfg['solver_iterations'])
            va,la,_=sweep(topo,*after,obs.dt_voltage[g,part],cfg['solver_iterations'])
            pb,qb=before[0].sum((0,2)),before[1].sum((0,2))
            pa,qa=after[0].sum((0,2)),after[1].sum((0,2))
            loss_delta=la-lb+transformer_loss(pa,qa,cfg)-transformer_loss(pb,qb,cfg)
            changed.dt_kwh[g,part]+=(pa-pb+loss_delta)*.25
            changed.dt_kvarh[g,part]+=(qa-qb)*.25
            for j,node in enumerate(topo.meter_nodes):
                phase=int(obs.meters.phase.iloc[g*m+j])
                changed.voltage[g*m+j,part]+=va[node,:,phase]-vb[node,:,phase]
    elif name=='upstream_hooking':
        # Extra energy at the LV bus before consumer branches; no meter label.
        active=obs.dt_voltage[g,part]>50
        changed.dt_kwh[g,part]+=severity*.25*active
        changed.dt_kvarh[g,part]+=severity*.25*np.tan(np.arccos(cfg['power_factor']))*active
    elif name=='ami_dropout':
        changed.kwh[meter_index,part]=np.nan
        changed.kvarh[meter_index,part]=np.nan
        changed.voltage[meter_index,part]=np.nan
        changed.missing[meter_index,part]=True
        changed.events[meter_index,part]=0
    else:
        raise ValueError('Unknown scenario')
    return changed,[f'Synthetic counterfactual: {name}; stored source telemetry remains unchanged.',
                    'Counterfactual demonstrations are separate from the held-out benchmark.']


class Scorer:
    def __init__(self):
        self.cfg,self.artifacts,self.results_dir=configured_paths()
        self.context=joblib.load(self.artifacts/'demo_context.joblib')
        self.models=joblib.load(self.artifacts/'models.joblib')
        self.selection=json.loads((self.artifacts/'selected_model.json').read_text())
        self.model_sha256=hashlib.sha256((self.artifacts/'models.joblib').read_bytes()).hexdigest()
        torch.set_num_threads(self.cfg['torch_threads'])
        self.ids={str(v):i for i,v in enumerate(self.context.meters.meter_id)}

    def timestamp_index(self,value):
        timestamp=pd.Timestamp(value)
        if timestamp.tzinfo is None:raise ValueError('Timestamps must include a timezone')
        step=(timestamp.to_pydatetime()-START_DATETIME).total_seconds()/900
        if abs(step-round(step))>1e-7:raise ValueError('Timestamp must align with a quarter-hour interval')
        return int(round(step))

    def score(self,request,write_audit=True):
        started=time.perf_counter()
        name=request.get('model') or self.selection['model']
        if name not in self.models:raise ValueError('Unknown model')
        obs=copy.deepcopy(self.context)
        total=obs.kwh.shape[1]
        end=int(request.get('as_of_interval') or total)
        minimum=(obs.cfg['baseline_days']+7)*96
        if not minimum<=end<=total:raise ValueError(f'Scoring requires a prefix between {minimum} and {total} stored intervals')
        for row in request.get('readings',[]):
            if row['meter_id'] not in self.ids:raise ValueError('Unknown meter; supply and validate network metadata before onboarding')
            i=self.ids[row['meter_id']];t=self.timestamp_index(row['ts'])
            if not 0<=t<end:raise ValueError('A reading is outside the requested scoring prefix')
            missing=bool(row.get('is_missing',False))
            if not missing and (row.get('kwh') is None or row.get('voltage_v') is None):
                raise ValueError('An observed reading requires kwh and voltage_v')
            obs.missing[i,t]=missing
            obs.kwh[i,t]=np.nan if missing else row['kwh']
            obs.kvarh[i,t]=np.nan if missing else row.get('kvarh',0.)
            obs.voltage[i,t]=np.nan if missing else row['voltage_v']
            obs.events[i,t]=0 if missing else row.get('event_flags',0)
        for row in request.get('dt_readings',[]):
            g=int(row['dt_id']);t=self.timestamp_index(row['ts'])
            if not 0<=g<obs.cfg['n_dts'] or not 0<=t<end:
                raise ValueError('DT reading is outside the known network or requested prefix')
            obs.dt_kwh[g,t]=row['kwh_in']
            if row.get('v_lv') is not None:obs.dt_voltage[g,t]=row['v_lv']
            if row.get('kvarh_in') is not None:obs.dt_kvarh[g,t]=row['kvarh_in']
        target=request.get('meter_id') or obs.meters.meter_id.iloc[0]
        if target not in self.ids:raise ValueError('Unknown scenario meter')
        obs,notes=scenario(obs,request.get('scenario','none'),self.ids[target],
                           request.get('severity',.7),end,request.get('tamper_event',False))
        f=build_features(obs,imputation=request.get('imputation',True),as_of_interval=end)
        model=self.models[name]
        score,prob=model.predict(f)
        op=model.thresholds[str(self.cfg['target_precision'])]
        flag=prob>=op['threshold']
        weights=prob*f.expected_load*flag
        allocations=np.zeros(len(prob))
        dt_rows=[]
        for g in range(obs.cfg['n_dts']):
            ix=f.dt_index==g;denom=weights[ix].sum()
            if denom>0:allocations[ix]=f.dt_estimate[g]*weights[ix]/denom
            dt_rows.append({'dt_id':g,'unexplained_kwh':float(f.dt_estimate[g]),
                'candidate_allocated_kwh':float(allocations[ix].sum()),
                'unassigned_kwh':float(f.dt_estimate[g]-allocations[ix].sum()),
                'input_kwh':float(f.dt_input_window[g]),
                'technical_loss_estimate_kwh':float(f.estimated_loss[g,f.start:end].sum()),
                'missing_fraction':float(obs.missing[ix,f.start:end].mean()),
                'localisation':'DT only for unmetered upstream load; no culprit inference'})
        order=np.lexsort((obs.meters.meter_id.to_numpy(),-score))
        ranks=np.empty(len(order),int);ranks[order]=np.arange(1,len(order)+1)
        meters=[]
        for i,row in obs.meters.iterrows():
            meters.append({'meter_id':row.meter_id,'dt_id':int(row.dt_id),'rank':int(ranks[i]),
                'probability_simulated_theft':float(prob[i]),'inspection_flag':bool(flag[i]),
                'candidate_kwh_allocation':float(allocations[i]),'category':row.category,
                'area_type':row.area_type,'missing_fraction':float(f.own[i,10]),
                'recorded_to_baseline_ratio':float(f.own[i,1]),
                'reason_codes':reason_codes(f,i)})
        meters.sort(key=lambda r:r['rank'])
        history=[];packets=[]
        m=obs.cfg['meters_per_dt']
        for g in range(obs.cfg['n_dts']):
            measured=np.nansum(obs.kwh[g*m:(g+1)*m,:end],axis=0)
            imputed=f.imputed[g*m:(g+1)*m].sum(0)
            miss=obs.missing[g*m:(g+1)*m,:end].mean(0)
            for begin in range(0,end,96):
                sl=slice(begin,min(end,begin+96))
                history.append({'dt_id':g,'ts':timestamp_for_interval(begin),
                    'input_kwh':float(obs.dt_kwh[g,sl].sum()),'observed_consumer_kwh':float(measured[sl].sum()),
                    'imputed_consumer_kwh':float(imputed[sl].sum()),'estimated_technical_kwh':float(f.estimated_loss[g,sl].sum()),
                    'residual_kwh':float(f.residual[g,sl].sum()),'missing_fraction':float(miss[sl].mean())})
            for t in range(max(0,end-96),end):
                packets.append({'dt_id':g,'ts':timestamp_for_interval(t),
                    'input_kwh':float(obs.dt_kwh[g,t]),'observed_consumer_kwh':float(measured[t]),
                    'imputed_consumer_kwh':float(imputed[t]),'estimated_technical_kwh':float(f.estimated_loss[g,t]),
                    'missing_fraction':float(miss[t])})
        explanation=None
        if name in ('M2','M3'):
            values=f.own if name=='M2' else f.full
            explanation=tree_group_shap(model.model,values[self.ids[target]],model.explanation_reference)
            explanation={**jsonable(explanation),'method':'exact group Shapley against the training median; raw probability, not causal'}
        if not op['validation_target_met']:
            notes.append('The validation precision target was unattainable for this fitted model; inspection flags are disabled.')
        notes += ['Synthetic profiles and reference topology; probabilities are not validated for Indian field use.',
            'Candidate kWh allocation is a heuristic share of unexplained DT energy, not established theft or recovered billing.',
            'A field inspector must verify every flag. No disconnection or penalty is automated.']
        response=jsonable({'model':name,'profile_source':'synthetic_fallback','config_hash':self.selection['config_hash'],
            'model_sha256':self.model_sha256,
            'as_of_interval':end,'as_of_time':timestamp_for_interval(end-1),
            'telemetry_mode':'replayed telemetry','scenario':request.get('scenario','none'),
            'operating_point':op,'meters':meters,'dts':sorted(dt_rows,key=lambda r:-r['unexplained_kwh']),
            'daily_balance':history,'interval_packets':packets,'selected_meter_explanation':explanation,
            'gnn_global_permutation_importance':self.selection.get('gnn_global_permutation_importance',[]) if name=='M4' else [],
            'notes':notes,'runtime_seconds':time.perf_counter()-started})
        if write_audit:
            event={'at_utc':datetime.now(timezone.utc).isoformat(),'model':name,
                'model_sha256':self.model_sha256,
                'config_hash':self.selection['config_hash'],'as_of_interval':end,'scenario':response['scenario'],
                'input_sha256':hashlib.sha256(json.dumps(jsonable(request),sort_keys=True,default=str).encode()).hexdigest(),
                'flagged_meters':[{'meter_id':r['meter_id'],'probability':r['probability_simulated_theft'],
                                  'reason_codes':r['reason_codes']} for r in meters if r['inspection_flag']],
                'automated_action':'none; decision support only'}
            self.results_dir.mkdir(exist_ok=True,parents=True)
            with _LOCK,(self.results_dir/'audit_log.jsonl').open('a') as out:
                out.write(json.dumps(event)+'\n')
        return response
