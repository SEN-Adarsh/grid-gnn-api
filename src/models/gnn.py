from __future__ import annotations
import copy
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from .baselines import finish_calibration
from ..features.build import OWN_NAMES


class TemporalGraphNet(nn.Module):
    def __init__(self,n_features,n_channels,hidden=32):
        super().__init__()
        self.temporal=nn.Sequential(nn.Conv1d(n_channels,12,5,padding=2),nn.ReLU(),
                                   nn.Conv1d(12,12,3,padding=1),nn.ReLU(),
                                   nn.AdaptiveAvgPool1d(1))
        self.input=nn.Linear(n_features+12,hidden)
        self.gcn1=nn.Linear(hidden,hidden)
        self.gcn2=nn.Linear(hidden,hidden)
        self.head=nn.Linear(hidden*2,1)
        self.dt_head=nn.Sequential(nn.Linear(hidden*2,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,x,seq,adj,locations,no_temporal=False):
        d,n,_=x.shape
        m=seq.shape[1]
        temporal=self.temporal(seq.reshape(d*m,seq.shape[2],seq.shape[3])).reshape(d,m,12)
        if no_temporal:
            temporal=temporal*0
        tempnodes=torch.zeros((d,n,12),dtype=x.dtype,device=x.device)
        tempnodes.scatter_(1,locations[:,:,None].expand(-1,-1,12),temporal)
        h0=torch.relu(self.input(torch.cat([x,tempnodes],dim=-1)))
        h=torch.relu(self.gcn1(torch.bmm(adj,h0)))
        h=torch.relu(self.gcn2(torch.bmm(adj,h))+h0)
        both=torch.cat([h0,h],dim=-1)
        meter=both.gather(1,locations[:,:,None].expand(-1,-1,both.shape[-1]))
        # Positive DT head is auxiliary; public residual remains the production
        # kWh estimator and its errors are separately reported.
        dt=torch.nn.functional.softplus(self.dt_head(meter.mean(1))).squeeze(-1)
        return self.head(meter).squeeze(-1),dt


class GNNModel:
    def __init__(self,variant='M4'):
        self.name=variant
        self.variant=variant
        self.thresholds={}

    def feature_values(self,f):
        x=f.full.copy()
        seq=f.sequence.copy()
        if self.variant=='M4_no_graph':
            # Strict graph-free control: remove engineered peer/DT/electrical
            # inputs as well as message passing. Retain temporal own-meter data.
            x[:,len(OWN_NAMES):]=0
            seq[:,3]=0
        if self.variant=='M4_no_voltage':
            x[:,-3:]=0
            seq[:,3]=0
        if self.variant=='M4_no_dt_residual':
            x[:,len(OWN_NAMES)+5:len(OWN_NAMES)+8]=0
        return x,seq

    def prepare(self,f):
        raw,seq=self.feature_values(f)
        x=np.clip(self.scaler.transform(raw),-8,8).astype('float32')
        seq=np.clip((seq-self.seq_mean[None,:,None])/self.seq_std[None,:,None],-8,8).astype('float32')
        d,m=f.node_indices.shape
        n=f.adjacency.shape[1]
        nodes=np.zeros((d,n,x.shape[1]),np.float32)
        for g in range(d):
            nodes[g,f.node_indices[g]]=x[g*m:(g+1)*m]
            # Root receives public DT features, not average meter labels.
            nodes[g,0,len(OWN_NAMES)+5:len(OWN_NAMES)+9]=x[g*m,len(OWN_NAMES)+5:len(OWN_NAMES)+9]
        adj=f.adjacency
        if self.variant=='M4_no_graph':
            adj=np.broadcast_to(np.eye(n,dtype=np.float32),(d,n,n)).copy()
        return tuple(torch.from_numpy(a) for a in [nodes,seq.reshape(d,m,*seq.shape[1:]),adj,f.node_indices])

    def raw(self,f):
        self.net.eval()
        with torch.no_grad():
            logits,dt=self.net(*self.prepare(f),no_temporal=self.variant=='M4_no_temporal')
        return logits.numpy().ravel()

    def predict(self,f):
        raw=self.raw(f)
        return raw,self.calibrator.predict(raw)

    def dt_prediction(self,f):
        self.net.eval()
        with torch.no_grad():
            _,value=self.net(*self.prepare(f),no_temporal=self.variant=='M4_no_temporal')
        return value.numpy()*f.dt_input_window


def fit_gnn(f,y,dt_target,split,cfg,seed,variant='M4',selected_params=None):
    start=time.perf_counter()
    torch.set_num_threads(cfg['torch_threads'])
    torch.use_deterministic_algorithms(True)
    train_d=np.flatnonzero(split=='train')
    val_d=np.flatnonzero(split=='val')
    train=split[f.dt_index]=='train'
    val=split[f.dt_index]=='val'
    d,m=f.node_indices.shape
    item=GNNModel(variant)
    x,seq=item.feature_values(f)
    item.scaler=StandardScaler().fit(x[train])
    item.seq_mean=seq[train].mean((0,2))
    item.seq_std=np.maximum(seq[train].std((0,2)),.1)
    tensors=item.prepare(f)
    labels=torch.tensor(y.reshape(d,m),dtype=torch.float32)
    targets=torch.tensor(np.maximum(dt_target,0)/np.maximum(f.dt_input_window,1),dtype=torch.float32)
    imbalance=float((1-y[train]).sum()/max(y[train].sum(),1))
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(imbalance))
    trials=([selected_params] if selected_params else [
        {'hidden':32,'lr':.003,'weight_decay':.001},
        {'hidden':48,'lr':.0015,'weight_decay':.003}][:cfg['search_trials']])
    best_global=-np.inf
    logs=[]
    for hp in trials:
        torch.manual_seed(seed)
        net=TemporalGraphNet(x.shape[1],seq.shape[1],hp['hidden'])
        optimizer=torch.optim.AdamW(net.parameters(),lr=hp['lr'],weight_decay=hp['weight_decay'])
        best=-np.inf;state=None;stale=0;best_epoch=0
        for epoch in range(cfg['gnn_epochs']):
            net.train();optimizer.zero_grad()
            logits,pred_dt=net(*(a[train_d] for a in tensors),no_temporal=variant=='M4_no_temporal')
            loss=criterion(logits,labels[train_d])+.1*nn.functional.smooth_l1_loss(pred_dt,targets[train_d])
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            net.eval()
            with torch.no_grad():
                vs,_=net(*(a[val_d] for a in tensors),no_temporal=variant=='M4_no_temporal')
            ap=float(average_precision_score(y.reshape(d,m)[val_d].ravel(),vs.numpy().ravel()))
            if ap>best+1e-5:
                best=ap;state=copy.deepcopy(net.state_dict());stale=0;best_epoch=epoch+1
            else:
                stale+=1
            if stale>=cfg['gnn_patience']:
                break
        logs.append({'params':hp,'validation_pr_auc':best,'best_epoch':best_epoch,'epochs_executed':epoch+1})
        if best>best_global:
            best_global=best
            item.net=TemporalGraphNet(x.shape[1],seq.shape[1],hp['hidden'])
            item.net.load_state_dict(state)
            item.params=hp
    finish_calibration(item,f,y,val,cfg)
    return item,{'fit_seconds':time.perf_counter()-start,'search_trials':len(trials),
                'validation_trials':logs,'operating_points':item.thresholds,'variant':variant}


def permutation_groups(model,f,y,mask,seed=802):
    """Validation-only sensitivity, not causal explanations or SHAP."""
    rng=np.random.default_rng(seed)
    base=average_precision_score(y[mask],model.raw(f)[mask])
    groups={'temporal_statistics':list(range(len(OWN_NAMES))),
            'peer_aggregates':list(range(len(OWN_NAMES),len(OWN_NAMES)+5)),
            'dt_residual':list(range(len(OWN_NAMES)+5,len(OWN_NAMES)+9)),
            'voltage':list(range(len(f.names)-3,len(f.names)))}
    out=[]
    for name,cols in groups.items():
        changed=copy.copy(f)
        values=f.full.copy()
        ids=np.flatnonzero(mask)
        permutation=rng.permutation(ids)
        values[np.ix_(ids,cols)]=values[np.ix_(permutation,cols)]
        changed.own=values[:,:len(OWN_NAMES)]
        changed.graph=values[:,len(OWN_NAMES):]
        out.append({'group':name,'validation_pr_auc_decrease':float(base-average_precision_score(y[mask],model.raw(changed)[mask]))})
    return out
