from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

METRICS=['precision_at_k','recall_at_k','pr_auc','vacant_fpr','dt_top1','dt_top3',
         'dt_mape','dt_bias_kwh','precision','recall','ece','inspections_80',
         'theft_kwh_recall_at_k','inspection_ratio_vs_m0']


def evaluation_frame(obs,hidden,f,models,cfg):
    ix=obs.split[f.dt_index]=='test'
    y=hidden.labels_meter.is_theft.to_numpy().astype(int)
    frame=pd.DataFrame({'meter_id':obs.meters.meter_id,'dt_id':f.dt_index,'y':y,
        'vacant':hidden.labels_meter.archetype.isin(['vacant','seasonal']),
        'archetype':hidden.labels_meter.archetype,'category':obs.meters.category,
        'area_type':obs.meters.area_type,'baseline_kwh_day':np.expm1(f.own[:,0]),
        'theft_kwh':hidden.theft_kwh[:,f.start:f.end].sum(1)})
    times={}
    import time
    for name,model in models.items():
        then=time.perf_counter()
        score,prob=model.predict(f)
        times[name]=(time.perf_counter()-then)/cfg['n_dts']
        frame[f'{name}_score']=score
        frame[f'{name}_prob']=prob
        frame[f'{name}_flag']=prob>=model.thresholds[str(cfg['target_precision'])]['threshold']
        frame[f'{name}_flag50']=prob>=model.thresholds[str(cfg['secondary_precision'])]['threshold']
    dt=pd.DataFrame({'dt_id':np.arange(cfg['n_dts']),
        'ntl_kwh':hidden.ntl[:,f.start:f.end].sum(1),
        'visible_kwh':hidden.visible[:,f.start:f.end].sum(1),
        'invisible_kwh':hidden.invisible[:,f.start:f.end].sum(1),
        'estimate_kwh':f.dt_estimate,
        'input_kwh':f.dt_input_window})
    return frame[ix].reset_index(drop=True),dt[obs.split=='test'].reset_index(drop=True),times


def calibration_error(y,prob):
    bins=np.minimum((np.clip(prob,0,1)*10).astype(int),9)
    value=0.
    for b in range(10):
        ix=bins==b
        if ix.any():
            value+=ix.mean()*abs(y[ix].mean()-prob[ix].mean())
    return float(value)


def measure(frame,dt,name):
    y=frame.y.to_numpy().astype(bool)
    s=frame[f'{name}_score'].to_numpy()
    prob=frame[f'{name}_prob'].to_numpy()
    flag=frame[f'{name}_flag'].to_numpy().astype(bool)
    # Stable ties by pseudonym; tied scores are not ranked by labels.
    order=np.lexsort((frame.meter_id.to_numpy(),-s))
    k=max(1,int(np.ceil(.05*len(frame))))
    top=order[:k]
    vacant=frame.vacant.to_numpy().astype(bool)
    weights=frame.theft_kwh.to_numpy()
    total_kwh=weights.sum()
    inspections=(int(np.searchsorted(np.cumsum(weights[order]),.8*total_kwh))+1) if total_kwh>0 else np.nan
    baseline_inspections=np.nan
    if 'M0_score' in frame and total_kwh>0:
        bo=np.lexsort((frame.meter_id.to_numpy(),-frame.M0_score.to_numpy()))
        baseline_inspections=int(np.searchsorted(np.cumsum(weights[bo]),.8*total_kwh))+1
    # All methods use the same metrology-based DT estimator. Its metrics must
    # therefore be identical; learned node ranking gets no credit for it.
    true=dt.ntl_kwh.to_numpy()
    est=dt.estimate_kwh.to_numpy()
    dt_order=np.argsort(-est,kind='stable')
    peak=int(np.argmax(true))
    positive=true>1.
    return {
        'precision_at_k':float(y[top].mean()),'recall_at_k':float(y[top].sum()/max(y.sum(),1)),
        'pr_auc':float(average_precision_score(y,s)) if y.any() else np.nan,
        'vacant_fpr':float(flag[vacant].mean()) if vacant.any() else np.nan,
        'dt_top1':float(peak in dt_order[:1]),'dt_top3':float(peak in dt_order[:3]),
        'dt_mape':float(np.mean(np.abs(est[positive]-true[positive])/true[positive])) if positive.any() else np.nan,
        'dt_bias_kwh':float(np.mean(est-true)),
        'precision':float(y[flag].mean()) if flag.any() else np.nan,
        'recall':float(y[flag].sum()/max(y.sum(),1)),
        'ece':calibration_error(y,prob),'inspections_80':float(inspections),
        'theft_kwh_recall_at_k':float(weights[top].sum()/total_kwh) if total_kwh else np.nan,
        'inspection_ratio_vs_m0':float(inspections/baseline_inspections),
        'flag_count':int(flag.sum()),'k':k,'meters':len(frame),'thefts':int(y.sum())}


def bootstrap_summary(runs,names,reps=1000,seed=9841):
    """Hierarchical bootstrap: resample independent seeds, then whole DTs.

    DT draws are paired across models. Precompute each seed's cluster draws,
    then resample seed/cluster-draw pairs for the mean-of-seed estimand.
    """
    rng=np.random.default_rng(seed)
    rcount=len(runs)
    cluster_reps=min(150,reps)
    point=np.empty((rcount,len(names),len(METRICS)))
    draws=np.empty((rcount,cluster_reps,len(names),len(METRICS)))
    for ri,(frame,dt) in enumerate(runs):
        for mi,name in enumerate(names):
            values=measure(frame,dt,name)
            point[ri,mi]=[values[k] for k in METRICS]
        groups=[frame[frame.dt_id==g] for g in dt.dt_id]
        for b in range(cluster_reps):
            ids=rng.integers(0,len(groups),len(groups))
            bf=pd.concat([groups[j] for j in ids],ignore_index=True)
            bd=dt.iloc[ids]
            for mi,name in enumerate(names):
                values=measure(bf,bd,name)
                draws[ri,b,mi]=[values[k] for k in METRICS]
    boots=np.empty((reps,len(names),len(METRICS)))
    for b in range(reps):
        ss=rng.integers(0,rcount,rcount)
        bb=rng.integers(0,cluster_reps,rcount)
        with np.errstate(invalid='ignore'):
            boots[b]=np.nanmean(draws[ss,bb],axis=0)
    def interval(mean,samples):
        finite=samples[np.isfinite(samples)]
        return {'mean':float(mean),'ci_low':float(np.quantile(finite,.025)) if len(finite) else None,
                'ci_high':float(np.quantile(finite,.975)) if len(finite) else None,
                'valid_bootstrap_draws':int(len(finite))}
    headline={}
    for mi,name in enumerate(names):
        headline[name]={k:interval(np.nanmean(point[:,mi,j]),boots[:,mi,j]) for j,k in enumerate(METRICS)}
    paired={}
    pairs=[('M4','M3'),('M3','M2'),('M3','M0')]
    if 'M4' in names:
        pairs += [('M4',n) for n in names if n.startswith('M4_')]
    for a,b in pairs:
        if a not in names or b not in names:
            continue
        ai,bi=names.index(a),names.index(b)
        paired[a+'-'+b]={k:interval(np.nanmean(point[:,ai,j]-point[:,bi,j]),boots[:,ai,j]-boots[:,bi,j])
                           for j,k in enumerate(METRICS)}
    return {'headline':headline,'paired':paired,'ci_method':
        'Hierarchical seed and DT-cluster percentile bootstrap; same resampling draws across models; mean of seed metrics.',
        'bootstrap_reps':reps,'cluster_draws_per_seed':cluster_reps,'n_seeds':rcount}


def fairness_table(frame,name):
    rows=[]
    band=pd.cut(frame.baseline_kwh_day,[-np.inf,4,10,np.inf],labels=['low','medium','high'])
    df=frame.assign(consumption_band=band.astype(str))
    overall=df[f'{name}_flag'].mean()
    for attribute in ['category','area_type','consumption_band','archetype']:
        for group,part in df.groupby(attribute,observed=True):
            y=part.y.to_numpy().astype(bool)
            flag=part[f'{name}_flag'].to_numpy().astype(bool)
            rows.append({'attribute':attribute,'group':group,'n':len(part),'positive_n':int(y.sum()),
                'precision':float(y[flag].mean()) if flag.any() else None,
                'recall':float(flag[y].mean()) if y.any() else None,
                'fpr':float(flag[~y].mean()) if (~y).any() else None,
                'flag_rate':float(flag.mean()),
                'flag_rate_disparity_ratio':float(flag.mean()/overall) if overall else None})
    return rows
