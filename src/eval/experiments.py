from __future__ import annotations
import copy
from dataclasses import replace
import numpy as np
import pandas as pd
from ..sim.simulator import Observations,HiddenLabels,simulate
from ..features.build import build_features,graph_structure
from ..models.gnn import fit_gnn
from ..models.baselines import fit_baselines
from .metrics import evaluation_frame,bootstrap_summary,measure


def subset_observations(obs,hidden,dt_ids):
    dt_ids=np.asarray(dt_ids)
    ids=np.flatnonzero(np.isin(obs.meters.dt_id,dt_ids))
    mapping={int(old):new for new,old in enumerate(dt_ids)}
    meters=obs.meters.iloc[ids].reset_index(drop=True).copy()
    meters['dt_id']=meters.dt_id.map(mapping)
    meters['mapped_dt_id']=meters.dt_id
    meters['feeder_id']=[f'DT{int(g):03d}-LV' for g in meters.dt_id]
    meters['pole_id']=[f'DT{int(g):03d}-P{str(p).rsplit("-P",1)[-1]}' for g,p in zip(meters.dt_id,meters.pole_id)]
    config={**obs.cfg,'n_dts':len(dt_ids)}
    sub=Observations(meters,obs.kwh[ids],obs.kvarh[ids],obs.voltage[ids],obs.events[ids],obs.missing[ids],
        obs.dt_kwh[dt_ids],obs.dt_kvarh[dt_ids],obs.dt_voltage[dt_ids],obs.dt_currents[dt_ids],
        obs.temperature,[obs.topologies[g] for g in dt_ids],obs.split[dt_ids],config,obs.seed)
    lab=hidden.labels_meter.iloc[ids].reset_index(drop=True).copy()
    truth=HiddenLabels(lab,hidden.ntl[dt_ids],hidden.technical[dt_ids],hidden.visible[dt_ids],
        hidden.invisible[dt_ids],hidden.theft_kwh[ids],np.array([mapping.get(int(g),-1) for g in hidden.true_dt[ids]]),
        hidden.clean_balance_error)
    return sub,truth


def stress_specs():
    specs=[]
    for v in [0.,.05,.15,.30]:
        specs.append(('dropout',v,{'dropout':v}))
    for resolution in [.1,1.]:
        for v in [.1,.5,1.,2.]:
            specs.append((f'voltage_sigma_resolution_{resolution}',v,
                          {'voltage_noise_v':v,'meter_voltage_resolution_v':resolution}))
    for v in [0.,.05,.10,.20]:
        specs.append(('mapping_error',v,{'mapping_error':v}))
    for v in [0,5,15]:
        specs.append(('dt_skew_minutes',v,{'dt_skew_minutes':v}))
    for v in [.03,.05,.08]:
        specs.append(('prevalence',v,{'prevalence':v}))
    specs.extend([('parameter_shift','B',{'parameter_set':'B'}),
                  ('unseen_topology','cigre_commercial',{'topology_source':'cigre_commercial'}),
                  ('events','off',{'events':False}),('events','on',{'events':True})])
    return specs


def run_stress_seed(cfg,seed,models):
    records=[];snr=[];cache={}
    for axis,value,overrides in stress_specs():
        change={**cfg,**overrides}
        latent_key=(change['prevalence'],change['parameter_set'],change['topology_source'])
        if latent_key not in cache:
            ideal={**change,'dropout':0.,'mapping_error':0.,'voltage_noise_v':0.,
                'meter_voltage_resolution_v':.000001,'dt_voltage_noise_v':0.,
                'energy_noise_fraction':0.,'dt_accuracy_fraction':0.,'dt_skew_minutes':0.,'events':True}
            base,truth=simulate(ideal,seed)
            cache[latent_key]=subset_observations(base,truth,np.flatnonzero(base.split=='test'))
        obs,hidden=measure_stress(cache[latent_key][0],cache[latent_key][1],change,seed)
        f=build_features(obs)
        frame,dt,_=evaluation_frame(obs,hidden,f,models,obs.cfg)
        records.append({'axis':axis,'value':value,'frame':frame,'dt':dt})
        y=hidden.labels_meter.is_theft.to_numpy().astype(bool)
        snr.append({'seed':seed,'axis':axis,'value':value,'median_snr_theft':float(np.median(f.snr[y])),
            'median_snr_honest':float(np.median(f.snr[~y])),'per_meter_snr':f.snr.tolist(),
            'meter_ids':obs.meters.meter_id.tolist()})
    return records,snr


def measure_stress(base,truth,cfg,seed):
    """Matched measurement draws across stress settings, fixed latent physics."""
    rng=np.random.default_rng(seed+8123)
    obs=copy.deepcopy(base)
    hidden=copy.deepcopy(truth)
    obs.cfg={**cfg,'n_dts':base.cfg['n_dts']}
    n,t=obs.kwh.shape;d=obs.cfg['n_dts'];m=obs.cfg['meters_per_dt']
    obs.kwh*=1+rng.normal(0,cfg['energy_noise_fraction'],(n,1))
    obs.voltage+=rng.normal(0,cfg['voltage_noise_v'],(n,t))
    res=cfg['meter_voltage_resolution_v']
    obs.voltage=np.round(obs.voltage/res)*res
    outage=base.dt_voltage<50
    obs.voltage[outage[obs.meters.dt_id]]=0
    obs.dt_voltage+=rng.normal(0,cfg['dt_voltage_noise_v'],(d,t))
    obs.dt_voltage[outage]=0
    for g in range(d):
        shift=rng.uniform(-1,1)*cfg['dt_skew_minutes']/cfg['interval_minutes']
        obs.dt_kwh[g]=np.interp(np.arange(t)+shift,np.arange(t),obs.dt_kwh[g])
        obs.dt_kwh[g]*=1+rng.uniform(-1,1)*cfg['dt_accuracy_fraction']
    missing=rng.random((n,t))<cfg['dropout']
    for i in range(n):
        burst=rng.random()<cfg['burst_dropout_fraction']
        at=int(rng.integers(cfg['baseline_days']*96,t-96))
        duration=int(rng.integers(24,3*96))
        if burst and cfg['dropout']>0:
            missing[i,at:at+duration]=True
    obs.missing=missing
    for arr in [obs.kwh,obs.kvarh,obs.voltage]:
        arr[missing]=np.nan
    obs.events[missing]=0
    if not cfg['events']:
        obs.events[:]=0
    # Wrong declared DT/slot assignment; every physical meter stays entirely
    # in the held-out partition. The scorer never sees this permutation.
    order=np.arange(n)
    selected=rng.choice(n,int(round(n*cfg['mapping_error'])),replace=False)
    if len(selected)>1:
        order[selected]=np.roll(selected,1)
    slots=obs.meters[['dt_id','mapped_dt_id','feeder_id','pole_id']].copy()
    obs.meters=obs.meters.iloc[order].reset_index(drop=True)
    obs.meters[['dt_id','mapped_dt_id','feeder_id','pole_id']]=slots
    for name in ['kwh','kvarh','voltage','events','missing']:
        setattr(obs,name,getattr(obs,name)[order])
    hidden.labels_meter=hidden.labels_meter.iloc[order].reset_index(drop=True)
    hidden.theft_kwh=hidden.theft_kwh[order]
    hidden.true_dt=hidden.true_dt[order]
    return obs,hidden


def ablations_for_seed(cfg,seed,obs,hidden,f,base_model):
    y=hidden.labels_meter.is_theft.to_numpy().astype(int)
    target=hidden.ntl[:,f.start:f.end].sum(1)
    frame,dt,_=evaluation_frame(obs,hidden,f,{'M4':base_model},cfg)
    logs={}
    for name in ['M4_no_graph','M4_no_temporal','M4_no_voltage','M4_no_dt_residual','M4_no_imputation','M4_shuffled']:
        af=f
        if name=='M4_no_imputation':
            af=build_features(obs,imputation=False)
        elif name=='M4_shuffled':
            af=copy.copy(f);af.adjacency,af.node_indices=graph_structure(obs,shuffle=True)
        model,log=fit_gnn(af,y,target,obs.split,cfg,seed,variant=name,selected_params=base_model.params)
        sf,_,_=evaluation_frame(obs,hidden,af,{name:model},cfg)
        for col in sf:
            if col.startswith(name+'_'):
                frame[col]=sf[col]
        logs[name]=log
    return frame,dt,logs


def temporal_check(cfg,seed,obs,hidden,selected_name):
    endpoint=cfg['days']-cfg['assessment_days']
    f=build_features(obs,endpoint=endpoint)
    later=build_features(obs)
    y=hidden.labels_meter.is_theft.to_numpy().astype(int)
    # A label becomes positive only once its injected attack has started.
    y_early=y*(hidden.labels_meter.theft_start.to_numpy()<endpoint*96)
    if selected_name=='M4':
        target=hidden.ntl[:,f.start:f.end].sum(1)
        model,log=fit_gnn(f,y_early,target,obs.split,cfg,seed)
    else:
        models,log=fit_baselines(f,y_early,obs.split,cfg,seed)
        model=models[selected_name]
    frame,dt,_=evaluation_frame(obs,hidden,later,{selected_name:model},cfg)
    return frame,dt,{'model':selected_name,'fit_end_day':endpoint,'test_end_day':cfg['days'],
        'train_and_val_dts_only':True,'test_dts_never_trained':True,'log':log}
