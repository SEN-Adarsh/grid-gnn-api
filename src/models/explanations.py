"""Exact group Shapley values for tree predictions with a fixed reference.

This is grouped, reference-based interventional SHAP, not a causal explanation
and not the specialised TreeSHAP algorithm. Feature dependence is ignored.
"""
from itertools import combinations
from math import factorial
import numpy as np
from ..features.build import OWN_NAMES


def groups(n_features):
    candidates={
        'consumption_history':[i for i in range(min(n_features,len(OWN_NAMES))) if i not in (10,11,12)],
        'data_quality':[10,12],
        'event_flags':[11],
        'peer_context':list(range(len(OWN_NAMES),len(OWN_NAMES)+5)),
        'dt_energy_balance':list(range(len(OWN_NAMES)+5,len(OWN_NAMES)+9)),
        'voltage':list(range(len(OWN_NAMES)+9,len(OWN_NAMES)+12))}
    return {name:[i for i in ids if i<n_features] for name,ids in candidates.items() if any(i<n_features for i in ids)}


def tree_group_shap(model,x,reference):
    x=np.atleast_2d(x).astype(np.float32)
    ref=np.asarray(reference,dtype=np.float32)
    items=list(groups(x.shape[1]).items());g=len(items);n=len(x)
    inputs=[]
    for mask in range(1<<g):
        z=np.broadcast_to(ref,x.shape).copy()
        for j,(_,cols) in enumerate(items):
            if mask&(1<<j):z[:,cols]=x[:,cols]
        inputs.append(z)
    values=np.asarray(model.predict_proba(np.concatenate(inputs,axis=0))[:,1],dtype=np.float64).reshape(1<<g,n)
    phi=np.zeros((n,g))
    for j in range(g):
        for mask in range(1<<g):
            if mask&(1<<j):continue
            size=mask.bit_count()
            weight=factorial(size)*factorial(g-size-1)/factorial(g)
            phi[:,j]+=weight*(values[mask|(1<<j)]-values[mask])
    return {'group_names':[name for name,_ in items],'values':phi,
            'reference_probability':values[0],'prediction':values[-1],
            'additivity_max_error':float(np.max(np.abs(values[0]+phi.sum(1)-values[-1])))}
