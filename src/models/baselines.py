from __future__ import annotations
import time
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from ..features.build import m0_scores


class Calibrator:
    """Platt calibration on validation only; never alters raw ranking."""
    def fit(self, raw, y):
        self.center = float(np.mean(raw))
        self.scale = max(float(np.std(raw)), 1e-6)
        self.model = LogisticRegression(C=1., max_iter=500, random_state=0)
        self.model.fit(((raw-self.center)/self.scale)[:,None],y)
        return self

    def predict(self,raw):
        return self.model.predict_proba(((raw-self.center)/self.scale)[:,None])[:,1]


def operating_threshold(y, prob, target, min_flags):
    choices = []
    for threshold in np.unique(prob):
        flagged = prob >= threshold
        count = int(flagged.sum())
        if count >= min_flags and y[flagged].mean() >= target:
            choices.append((int(y[flagged].sum()),count,float(threshold)))
    if not choices:
        return {'threshold':1.000001,'validation_target_met':False,
                'target_precision':target,'validation_flags':0,
                'validation_precision':None,'validation_recall':0.}
    # Maximise captured positives subject to target precision, then fewer flags.
    tp,count,threshold = sorted(choices,key=lambda z:(-z[0],z[1],-z[2]))[0]
    return {'threshold':threshold,'validation_target_met':True,
            'target_precision':target,'validation_flags':count,
            'validation_precision':tp/count,'validation_recall':tp/max(y.sum(),1)}


class Baseline:
    def __init__(self,name,model=None):
        self.name,self.model = name,model
        self.calibrator = None
        self.thresholds = {}

    def raw(self,f):
        if self.name == 'M0':
            return m0_scores(f)
        x = f.own if self.name in ('M1','M2') else f.full
        if self.name == 'M1':
            return -self.model.score_samples(self.scaler.transform(x))
        return self.model.predict_proba(x)[:,1]

    def predict(self,f):
        raw = self.raw(f)
        return raw,self.calibrator.predict(raw)


def finish_calibration(model,f,y,val,cfg):
    raw = model.raw(f)
    model.calibrator = Calibrator().fit(raw[val],y[val])
    probs = model.calibrator.predict(raw)
    for p in [cfg['target_precision'],cfg['secondary_precision']]:
        model.thresholds[str(p)] = operating_threshold(y[val],probs[val],p,cfg['min_validation_flags'])
    return model


def fit_baselines(f,y,split,cfg,seed):
    train = split[f.dt_index] == 'train'
    val = split[f.dt_index] == 'val'
    fitted,logs = {},{}
    for name in ['M0','M1','M2','M3']:
        start=time.perf_counter()
        item=Baseline(name)
        log={'search_trials':0,'validation_trials':[]}
        if name == 'M1':
            item.scaler=StandardScaler().fit(f.own[train])
            item.model=IsolationForest(n_estimators=cfg['rf_trees'],max_samples=min(256,int(train.sum())),
                contamination='auto',random_state=seed,n_jobs=2)
            item.model.fit(item.scaler.transform(f.own[train]))
            item.model.n_jobs=1
        if name in ('M2','M3'):
            x=f.own if name == 'M2' else f.full
            item.explanation_reference=np.median(x[train],axis=0)
            best=-np.inf
            trials=[{'max_depth':8,'min_samples_leaf':3},
                    {'max_depth':None,'min_samples_leaf':2}][:cfg['search_trials']]
            for h in trials:
                rf=RandomForestClassifier(n_estimators=cfg['rf_trees'],class_weight='balanced_subsample',
                    random_state=seed,n_jobs=2,max_features=.8,**h)
                rf.fit(x[train],y[train])
                # Serial probability reduction makes floating-point summation
                # and threshold ties bit-for-bit deterministic.
                rf.n_jobs=1
                ap=average_precision_score(y[val],rf.predict_proba(x[val])[:,1])
                log['validation_trials'].append({**h,'pr_auc':float(ap)})
                if ap>best:
                    item.model=rf
                    best=ap
            log['search_trials']=len(trials)
        finish_calibration(item,f,y,val,cfg)
        log['fit_seconds']=time.perf_counter()-start
        log['operating_points']=item.thresholds
        fitted[name]=item
        logs[name]=log
    return fitted,logs
