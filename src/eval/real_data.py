from __future__ import annotations
import hashlib
import shutil
import struct
import zlib
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import train_test_split
from ..common import hash_file


def obtain_sgcc(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    paths=[]
    integrity=root/'integrity.json'
    if not integrity.exists() or not (root/'data.csv').exists():
        for name in ['data.z01','data.z02','data.zip']:
            file=root/name
            if not file.exists():
                url='https://raw.githubusercontent.com/henryRDlab/ElectricityTheftDetection/master/'+name
                with urllib.request.urlopen(url,timeout=60) as r,file.open('wb') as f:
                    shutil.copyfileobj(r,f)
            paths.append(file)
        # This release is a genuine spanned ZIP. The platform's `zip -s 0`
        # conversion produced a corrupt stream, so decode the sole DEFLATE
        # entry across the original, byte-verified parts. Validate CRC and size
        # BEFORE making any extracted CSV available to modelling.
        raw=b''.join((root/name).read_bytes() for name in ['data.z01','data.z02','data.zip'])
        pos=raw.find(b'PK\x03\x04')
        if pos<0:raise ValueError('Missing SGCC local ZIP header')
        header=struct.unpack_from('<IHHHHHIIIHH',raw,pos)
        _,_,flags,method,_,_,crc,compressed_size,uncompressed_size,name_len,extra_len=header
        name=raw[pos+30:pos+30+name_len]
        if name!=b'data.csv' or method!=8 or flags&9 or uncompressed_size>512*1024**2:
            raise ValueError('Unexpected SGCC spanned ZIP structure')
        start=pos+30+name_len+extra_len
        decoder=zlib.decompressobj(-15)
        data=decoder.decompress(raw[start:start+compressed_size],uncompressed_size+1)
        data+=decoder.flush()
        if not decoder.eof or len(data)!=uncompressed_size or zlib.crc32(data)&0xffffffff!=crc:
            raise ValueError('SGCC DEFLATE length/CRC verification failed; refuse partial extraction')
        temporary=root/'data.csv.partial'
        temporary.write_bytes(data)
        temporary.replace(root/'data.csv')
        import json
        integrity.write_text(json.dumps({'sha256':hash_file(root/'data.csv'),'bytes':(root/'data.csv').stat().st_size,'crc_verified':True}), encoding='utf-8')
    else:
        import json
        checked=json.loads(integrity.read_text())
        if hash_file(root/'data.csv')!=checked['sha256']:
            raise ValueError('SGCC extracted checksum changed')
    return root/'data.csv'


def sgcc_features(values):
    missing=~np.isfinite(values)
    counts=np.maximum((~missing).sum(1),1)
    means=np.nansum(values,axis=1)/counts
    x=np.where(missing,means[:,None],values)
    scale=np.maximum(means,.1)
    feats=[np.log1p(np.maximum(means,0)),x.std(1)/scale,missing.mean(1),(x==0).mean(1),
           np.quantile(x,.1,axis=1)/scale,np.quantile(x,.9,axis=1)/scale,
           np.abs(np.diff(x,axis=1)).mean(1)/scale]
    for chunk in np.array_split(np.arange(x.shape[1]),8):
        feats += [x[:,chunk].mean(1)/scale,x[:,chunk].std(1)/scale,missing[:,chunk].mean(1)]
    return np.nan_to_num(np.column_stack(feats)).astype('float32')


def sgcc_check(cfg):
    began=time.perf_counter()
    try:
        path=obtain_sgcc(cfg['sgcc_dir'])
        df=pd.read_csv(path)
        if df.CONS_NO.duplicated().any():
            raise ValueError('Duplicate customers: must group/deduplicate before splitting')
        date_cols=sorted([c for c in df.columns if c not in ('CONS_NO','FLAG')],key=pd.Timestamp)
        y=df.FLAG.to_numpy().astype(int)
        if set(np.unique(y))!={0,1}:
            raise ValueError('Unexpected SGCC labels')
        x=sgcc_features(df[date_cols].to_numpy(dtype=np.float32))
        ix=np.arange(len(y))
        train,rest=train_test_split(ix,test_size=.4,stratify=y,random_state=1208)
        val,test=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=1208)
        best=-np.inf;chosen=None;trials=[]
        for depth,leaf in [(12,3),(None,3)]:
            model=RandomForestClassifier(n_estimators=120,max_depth=depth,min_samples_leaf=leaf,
                class_weight='balanced_subsample',random_state=1208,n_jobs=2,max_features=.8)
            model.fit(x[train],y[train])
            model.n_jobs=1
            ap=average_precision_score(y[val],model.predict_proba(x[val])[:,1])
            trials.append({'max_depth':depth,'validation_pr_auc':float(ap)})
            if ap>best:
                best=ap;chosen=model
        score=chosen.predict_proba(x[test])[:,1]
        rng=np.random.default_rng(740)
        metrics={'pr_auc':[],'roc_auc':[]}
        for _ in range(cfg['bootstrap_reps']):
            sample=rng.integers(0,len(test),len(test))
            yy=y[test][sample];ss=score[sample]
            metrics['pr_auc'].append(average_precision_score(yy,ss))
            metrics['roc_auc'].append(roc_auc_score(yy,ss))
        estimates={}
        for key,fun in [('pr_auc',average_precision_score),('roc_auc',roc_auc_score)]:
            estimates[key]={'mean':float(fun(y[test],score)),
                'ci_low':float(np.quantile(metrics[key],.025)),'ci_high':float(np.quantile(metrics[key],.975))}
        return {'status':'completed','source':'SGCC author GitHub repository','country':'China',
            'customer_rows_observed':len(df),'date_columns_observed':len(date_cols),
            'positive_labels_observed':int(y.sum()),'positive_fraction_observed':float(y.mean()),
            'advertised_customer_rows':42372,'advertised_date_columns':1035,
            'source_discrepancy':'Customer count matches README; observed date columns omit one calendar day from the advertised span.',
            'missing_calendar_dates':[v.strftime('%Y-%m-%d') for v in pd.date_range(pd.Timestamp(date_cols[0]),pd.Timestamp(date_cols[-1])).difference(pd.DatetimeIndex(date_cols))],
            'file_sha256':hash_file(path),'file_bytes':path.stat().st_size,
            'split':'customer-stratified disjoint train/validation/test; dates explicitly sorted',
            'train_customers':len(train),'validation_customers':len(val),'test_customers':len(test),
            'unique_customer_ids':int(df.CONS_NO.nunique()),'duplicate_customer_ids':0,
            'first_date':date_cols[0],'last_date':date_cols[-1],
            'metrics':estimates,'trials':trials,'model':'RandomForest on customer temporal summaries',
            'bootstrap_unit':'customer','seconds':time.perf_counter()-began,
            'limitations':['Daily, non-Indian consumption without topology','Noisy labels; label timing not known',
                          'Research access is public; dataset redistribution licence not explicit',
                          'One customer split; this does not validate the grid model']}
    except Exception as exc:
        return {'status':'skipped','reason':f'{type(exc).__name__}: {exc}','seconds':time.perf_counter()-began}
