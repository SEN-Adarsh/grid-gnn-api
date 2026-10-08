from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..common import write_json

COLORS={'M0':'#758396','M1':'#a5afbb','M2':'#e6aa39','M3':'#0b8c81','M4':'#2463eb'}


def claims_register(out):
    out=Path(out)
    results=json.loads((out/'results.json').read_text())
    config_hash=results['config_hash']
    rows=[]
    def walk(value,path,source):
        if isinstance(value,dict):
            if {'mean','ci_low','ci_high'}<=set(value):
                rows.append({'claim_text':path,'metric':path,'value':value['mean'],
                    'ci_low':value['ci_low'],'ci_high':value['ci_high'],'source_file':source,'config_hash':config_hash})
                return
            for k,v in value.items():walk(v,f'{path}.{k}' if path else k,source)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,f'{path}[{i}]',source)
        elif value is not None:
            rows.append({'claim_text':path,'metric':path,'value':value,'ci_low':'','ci_high':'',
                'source_file':source,'config_hash':config_hash})
    for p in sorted(out.glob('*.json')):
        if p.name in ('results_checkpoint.json','protocol.json') or p.name.startswith('seed_'):
            continue
        walk(json.loads(p.read_text()),'',str(p))
    with (out/'claims_register.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['claim_text','metric','value','ci_low','ci_high','source_file','config_hash'])
        writer.writeheader();writer.writerows(rows)
    return len(rows)


def figures(results,out):
    out=Path(out);dest=out/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
    names=list(COLORS)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,['precision_at_k','pr_auc','vacant_fpr'],
                           ['Precision at inspection budget','PR-AUC','Vacancy / seasonal false flags']):
        data=[results['headline'][n][key] for n in names]
        values=np.array([r['mean'] for r in data])
        error=np.array([[r['mean']-r['ci_low'] for r in data],[r['ci_high']-r['mean'] for r in data]])
        ax.bar(names,values,color=[COLORS[n] for n in names],width=.65)
        ax.errorbar(range(5),values,yerr=np.maximum(error,0),fmt='none',color='#172235',capsize=4,lw=1)
        ax.set_ylim(0,1);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Synthetic grid benchmark · mean and DT-cluster confidence intervals\nM0 abstains; M1 rarely flags. Zero vacancy FPR is not proof of protection.',fontsize=12)
    fig.savefig(dest/'benchmark.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    pairs=['M4-M3','M3-M2','M3-M0']
    for i,p in enumerate(pairs):
        r=results['paired'][p]['pr_auc']
        ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],
                    fmt='o',capsize=5,color='#2463eb',markersize=7)
    ax.axvline(0,color='#8492a5',ls='--');ax.set_yticks(range(len(pairs)),pairs)
    ax.set_xlabel('PR-AUC difference');ax.set_title('Does added graph complexity help?')
    fig.savefig(dest/'paired_differences.png',dpi=180);plt.close(fig)
    axes_specs=[('dropout','Missing telemetry'),('mapping_error','Wrong declared mapping'),
                ('dt_skew_minutes','DT clock skew (minutes)'),('voltage_sigma_resolution_1.0','Voltage noise, coarse resolution')]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(axis,title) in zip(axes.flat,axes_specs):
        rows=sorted([s for s in results['stress'] if s['axis']==axis],key=lambda s:float(s['value']))
        for name in ['M2','M3','M4']:
            x=[float(s['value']) for s in rows]
            y=[s['headline'][name]['pr_auc']['mean'] for s in rows]
            low=[s['headline'][name]['pr_auc']['ci_low'] for s in rows]
            high=[s['headline'][name]['pr_auc']['ci_high'] for s in rows]
            ax.plot(x,y,'o-',label=name,color=COLORS[name]);ax.fill_between(x,low,high,color=COLORS[name],alpha=.10)
        ax.set_title(title);ax.set_ylabel('PR-AUC');ax.set_ylim(0,1);ax.legend(frameon=False,ncols=3,fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('Fixed-model stress tests · synthetic telemetry',fontsize=14)
    fig.savefig(dest/'stress_curves.png',dpi=180);plt.close(fig)
    frames=[pd.read_csv(out/f'seed_{results["config"]["seed_start"]+i}_meters.csv') for i in range(results['config']['seeds'])]
    frame=pd.concat(frames,ignore_index=True)
    fig,ax=plt.subplots(figsize=(5.5,4.5),layout='constrained')
    ax.plot([0,1],[0,1],color='#8492a5',ls='--',label='Ideal')
    bins=np.linspace(0,1,11)
    for name in ['M2','M3','M4']:
        p=frame[name+'_prob'].to_numpy();y=frame.y.to_numpy()
        ids=np.minimum((p*10).astype(int),9)
        means=[];observed=[]
        for b in range(10):
            if (ids==b).sum():
                means.append(p[ids==b].mean());observed.append(y[ids==b].mean())
        ax.plot(means,observed,'o-',color=COLORS[name],label=name)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Validation-calibrated probability',ylabel='Test positive fraction',
           title='Calibration on held-out DTs')
    ax.legend(frameon=False);fig.savefig(dest/'reliability.png',dpi=180);plt.close(fig)
    ab=results['ablations']['paired']
    names=[p for p in ab if p.startswith('M4-M4_')]
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for i,n in enumerate(names):
        r=ab[n]['pr_auc']
        ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],fmt='o',capsize=4,color='#0b8c81')
    ax.axvline(0,color='#8492a5',ls='--')
    ax.set_yticks(range(len(names)),[n.replace('M4-M4_','Full minus ').replace('_',' ') for n in names])
    ax.set_xlabel('PR-AUC difference');ax.set_title('Ablations · limited independent seeds')
    fig.savefig(dest/'ablations.png',dpi=180);plt.close(fig)


def generate_evidence(cfg):
    out=Path(cfg['output_dir'])
    results=json.loads((out/'results.json').read_text())
    figures(results,out)
    claims_register(out)
    return results
