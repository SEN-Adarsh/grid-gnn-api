from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..sim.physics import sweep, meter_power_to_nodes, transformer_loss


OWN_NAMES = ['log_baseline_kwh_day', 'recent_baseline_ratio', 'week_baseline_ratio',
             'recent_cv', 'zero_fraction', 'max_zero_run_days', 'constant_fraction',
             'night_fraction', 'day_fraction', 'weekend_ratio', 'missing_fraction',
             'event_rate_day', 'poweroff_fraction', 'negative_fraction',
             'recent_slope', 'temperature_adjusted_ratio', 'category_commercial',
             'category_agricultural', 'log_sanctioned_load']
GRAPH_NAMES = ['peer_ratio_mean', 'peer_ratio_std', 'peer_ratio_q25', 'peer_ratio_q75',
               'ratio_relative_to_peers', 'dt_residual_fraction', 'dt_week_residual_fraction',
               'dt_residual_persistence', 'dt_missing_fraction',
               'voltage_gap_change_v', 'voltage_gap_snr', 'voltage_gap_std_v']
VOLTAGE_NAMES = GRAPH_NAMES[-3:]
FORBIDDEN = ('labels', 'archetype', 'is_theft', 'physical_dt', 'true_', 'theft_params', 'tech_loss_kwh')


@dataclass
class Features:
    own: np.ndarray
    graph: np.ndarray
    sequence: np.ndarray
    adjacency: np.ndarray
    node_indices: np.ndarray
    dt_index: np.ndarray
    names: list
    dt_estimate: np.ndarray
    residual: np.ndarray
    estimated_loss: np.ndarray
    imputed: np.ndarray
    expected_load: np.ndarray
    dt_input_window: np.ndarray
    snr: np.ndarray
    endpoint: float
    start: int
    missing_policy: str

    @property
    def full(self):
        return np.concatenate([self.own, self.graph], axis=1)

    @property
    def end(self):
        return int(round(self.endpoint*96))


def causal_impute(kwh, dt_voltage, dt_index, enabled=True):
    """Missing != zero. Prior same-quarter-hour observations from prior days.

    Strictly causal; preceding available days only. If no slot history exists,
    use the expanding per-meter mean of preceding days; missing readings on
    the initial day are zero with the mask retained. Current DT outage
    overrides the expectation to zero. Zero-fill is an explicit ablation only.
    """
    kwh = np.asarray(kwh)
    if not enabled:
        return np.nan_to_num(kwh, nan=0).astype('float32')
    n, steps = kwh.shape
    out = kwh.copy()
    history_sum = np.zeros((n, 96), np.float64)
    history_count = np.zeros((n, 96), np.int32)
    overall_sum = np.zeros(n)
    overall_count = np.zeros(n)
    # Moving window over preceding days, no use of subsequent days.
    days=(steps+95)//96
    byday = np.pad(kwh,((0,0),(0,days*96-steps)),constant_values=np.nan).reshape(n,days,96)
    for day in range(days):
        if day:
            prev = byday[:,day-1]
            history_sum += np.nan_to_num(prev)
            history_count += np.isfinite(prev)
            overall_sum += np.nansum(prev, axis=1)
            overall_count += np.isfinite(prev).sum(1)
        if day > 7:
            old = byday[:,day-8]
            history_sum -= np.nan_to_num(old)
            history_count -= np.isfinite(old)
        fallback = overall_sum / np.maximum(overall_count, 1)
        guess = np.divide(history_sum, np.maximum(history_count, 1))
        guess = np.where(history_count > 0, guess, fallback[:,None])
        part = slice(day*96,(day+1)*96)
        width=kwh[:,part].shape[1]
        value = np.where(np.isfinite(kwh[:,part]), kwh[:,part], guess[:,:width])
        missing_outage = ~np.isfinite(kwh[:,part]) & (dt_voltage[dt_index,part] < 50)
        value[missing_outage] = 0
        out[:,part] = value
    return out.astype('float32')


def max_run(mask):
    current = np.zeros(mask.shape[0], np.int32)
    longest = current.copy()
    for col in mask.T:
        current = np.where(col, current+1, 0)
        longest = np.maximum(longest, current)
    return longest


def feature_contract(names):
    if len(names) != len(set(names)):
        raise ValueError('Duplicate feature names')
    for name in names:
        if any(x in name.lower() for x in FORBIDDEN):
            raise ValueError(f'Hidden label in features: {name}')


def graph_structure(obs, shuffle=False):
    d, m = obs.cfg['n_dts'], obs.cfg['meters_per_dt']
    nmax = max(t.size for t in obs.topologies)
    adj = np.zeros((d,nmax,nmax), np.float32)
    locations = np.zeros((d,m), np.int64)
    rng = np.random.default_rng(obs.seed+902)
    for g, topo in enumerate(obs.topologies):
        a = np.eye(nmax, dtype=np.float32)
        perm = np.arange(topo.size)
        if shuffle:
            # Permute node identities while preserving weights, degree multiset
            # and graph connectivity. Node data and labels are NOT permuted.
            perm = rng.permutation(topo.size)
        for node in range(1, topo.size):
            i,j = perm[node],perm[topo.parent[node]]
            a[i,j] = a[j,i] = 1/(1+topo.r[node]/.01)
        scale = 1/np.sqrt(a.sum(1))
        adj[g] = a * scale[:,None] * scale[None,:]
        locations[g] = topo.meter_nodes
    return adj, locations


def build_features(obs, endpoint=None, imputation=True, shuffled=False, as_of_interval=None):
    cfg = obs.cfg
    d,m = cfg['n_dts'], cfg['meters_per_dt']
    endpoint = cfg['days'] if endpoint is None else endpoint
    t = endpoint*96 if as_of_interval is None else int(as_of_interval)
    if t<7*96 or t>cfg['days']*96:
        raise ValueError('Scoring prefix is outside the stored history')
    endpoint=(t+95)//96
    partial=t%96!=0
    padding=endpoint*96-t
    baseline_days = min(cfg['baseline_days'], endpoint-1)
    b = baseline_days*96
    start = max(b, t-cfg['assessment_days']*96)
    current = slice(start,t)
    recent_week = slice(max(b,t-7*96),t)
    dt_idx = obs.meters.dt_id.to_numpy()
    kwh = obs.kwh[:,:t]
    imp = causal_impute(kwh, obs.dt_voltage[:,:t], dt_idx, imputation)
    baseline = np.maximum(np.mean(np.maximum(imp[:,:b],0),axis=1), .001)
    recent = imp[:,current]
    ratio = np.mean(recent,axis=1) / baseline
    week_ratio = np.mean(imp[:,recent_week],axis=1) / baseline
    hour = np.arange(t)%96/4
    day = np.arange(t)//96
    def daily(values,how='mean'):
        if not partial:
            cube=values.reshape(values.shape[0],endpoint,96)
            return cube.sum(2) if how=='sum' else cube.mean(2)
        padded=np.pad(values,((0,0),(0,padding)))
        cube=padded.reshape(values.shape[0],endpoint,96)
        counts=np.full(endpoint,96.);counts[-1]=96-padding
        # Partial daily sums are extrapolated for the temporal encoder only.
        return cube.sum(2)/counts[None,:]*(96 if how=='sum' else 1)
    day_energy = daily(imp,'sum')
    seasonal_adjust = 1 + .015*(obs.temperature[current].mean()-obs.temperature[:b].mean())
    own = np.column_stack([
        np.log1p(baseline*96), ratio, week_ratio,
        recent.std(1)/(np.abs(recent.mean(1))+.002),
        (np.abs(recent)<1e-5).mean(1), max_run(np.abs(recent)<1e-5)/96,
        (np.abs(np.diff(recent,axis=1))<1e-6).mean(1),
        np.mean(recent[:,(hour[current]>=22)|(hour[current]<6)],axis=1)/baseline,
        np.mean(recent[:,(hour[current]>=10)&(hour[current]<17)],axis=1)/baseline,
        np.mean(recent[:,day[current]%7>=5],axis=1)/(np.abs(np.mean(recent[:,day[current]%7<5],axis=1))+.001),
        obs.missing[:,current].mean(1), ((obs.events[:,current]&1)>0).sum(1)/((t-start)/96),
        ((obs.events[:,current]&2)>0).mean(1), (recent<0).mean(1),
        (day_energy[:,-3:].mean(1)-day_energy[:,-7:-4].mean(1))/(baseline*96),
        ratio/max(seasonal_adjust,.2),
        (obs.meters.category=='small_commercial').to_numpy(),
        (obs.meters.category=='agricultural').to_numpy(),
        np.log1p(obs.meters.sanctioned_load_kw.to_numpy())])
    estimated_loss = np.zeros((d,t), np.float32)
    expected_v = np.zeros((d*m,t), np.float32)
    qratio = np.tan(np.arccos(cfg['power_factor']))
    for g in range(d):
        idx = slice(g*m,(g+1)*m)
        topo = obs.topologies[g]
        # Only declared phase, declared topology, public readings and public
        # transformer nameplate assumptions enter estimated losses.
        power = imp[idx]/(cfg['interval_minutes']/60)
        pp,qq = meter_power_to_nodes(topo,power,obs.meters.phase.to_numpy()[idx],qratio)
        root = np.maximum(obs.dt_voltage[g,:t],0)
        vv,line_loss,_ = sweep(topo,pp,qq,root,cfg['solver_iterations'])
        dt_loss = transformer_loss(pp.sum((0,2)),qq.sum((0,2)),cfg)*(root>50)
        estimated_loss[g] = (line_loss+dt_loss)*(cfg['interval_minutes']/60)*cfg['loss_estimator_multiplier']
        for j,node in enumerate(topo.meter_nodes):
            expected_v[g*m+j] = vv[node,:,obs.meters.phase.iloc[g*m+j]]
    sums = imp.reshape(d,m,t).sum(1)
    residual = obs.dt_kwh[:,:t]-sums-estimated_loss
    inputs = np.maximum(obs.dt_kwh[:,current].sum(1),1)
    dt_estimate = np.maximum(residual[:,current].sum(1),0)
    resid_fraction = residual[:,current].sum(1)/inputs
    week_resid = residual[:,recent_week].sum(1)/np.maximum(obs.dt_kwh[:,recent_week].sum(1),1)
    daily_resid = daily(residual,'sum')
    persistence = (daily_resid[:,-min(7,endpoint):] > 0).mean(1)
    missing_dt = obs.missing[:,:t].reshape(d,m,t).mean(1)[:,current].mean(1)
    gap = expected_v-obs.voltage[:,:t]
    gap[obs.dt_voltage[dt_idx,:t]<50] = np.nan
    gap_base = np.nanmean(gap[:,:b],axis=1)
    gap_std = np.nanstd(gap[:,:b],axis=1)
    delta = np.nanmean(gap[:,current],axis=1)-gap_base
    snr = np.abs(delta)/np.maximum(gap_std,.05)
    peer = np.zeros((d*m,5),np.float32)
    cats = obs.meters.category.to_numpy()
    for g in range(d):
        for c in np.unique(cats[g*m:(g+1)*m]):
            ix = np.flatnonzero((dt_idx==g)&(cats==c))
            # Leave-self-out aggregates prevent a target defining its own peer.
            for i in ix:
                other = ix[ix!=i]
                if not len(other):
                    other = np.flatnonzero((dt_idx==g)&(np.arange(d*m)!=i))
                values = ratio[other]
                mu = values.mean()
                peer[i] = [mu,values.std(),np.quantile(values,.25),np.quantile(values,.75),ratio[i]-mu]
    graph = np.column_stack([peer,resid_fraction[dt_idx],week_resid[dt_idx],
                              persistence[dt_idx],missing_dt[dt_idx],delta,snr,
                              np.nanstd(gap[:,current],axis=1)])
    daily_missing = daily(obs.missing[:,:t])
    daily_events = daily(((obs.events[:,:t]&1)>0),'sum')
    valid_gap = np.nan_to_num(gap-gap_base[:,None])
    daily_gap = daily(valid_gap)
    night_energy = daily(imp*((hour>=22)|(hour<6)),'sum')
    sequence = np.stack([day_energy/(baseline[:,None]*96),daily_missing,daily_events,
                          daily_gap,night_energy/(baseline[:,None]*96)],axis=1)
    # Fixed daily timeline length is retained within a fitted model; padded
    # early-window sequences contain a missing-mask channel of one.
    if endpoint < cfg['days']:
        pad = cfg['days']-endpoint
        sequence = np.pad(sequence,((0,0),(0,0),(0,pad)))
        sequence[:,1,endpoint:] = 1
    adj,loc = graph_structure(obs,shuffled)
    names = OWN_NAMES+GRAPH_NAMES
    feature_contract(names)
    return Features(np.nan_to_num(own).astype('float32'),np.nan_to_num(graph).astype('float32'),
        np.nan_to_num(sequence).astype('float32'),adj,loc,dt_idx,names,dt_estimate,
        residual,estimated_loss,imp,baseline*(t-start),inputs,
        np.nan_to_num(snr),t/96,start,'causal_same_slot' if imputation else 'zero_fill')


def m0_scores(features):
    # A deliberately strong untrained baseline: unexplained DT fraction times
    # estimated own-history missing energy, normalised by expected energy.
    ratio = features.own[:,1]
    deficit = np.clip(1-ratio,0,1)
    residual = np.maximum(features.graph[:,5],0)
    return residual * (.05 + deficit) + .001*deficit


def reason_codes(features, row):
    own,graph = features.own[row],features.graph[row]
    return [
        f"Recorded energy is {own[1]:.2f} times the own-history expectation.",
        f"Declared DT unexplained balance is {graph[5]:+.1%} after estimated loss and imputation.",
        f"Matched peers average {graph[0]:.2f} times their history; missing intervals {own[10]:.1%}.",
        f"Tamper-like event rate {own[11]:.2f} per day; events are imperfect evidence.",
        "Vacancy, meter faults, mapping and clock errors remain alternative explanations; inspect before acting."]
