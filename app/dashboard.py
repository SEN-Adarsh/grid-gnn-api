from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from app.scoring import Scorer,ROOT

st.set_page_config(page_title='Grid-GNN | NTL Workbench',page_icon='⚡',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1450px}
[data-testid="stMetric"]{background:#f2f6fb;border:1px solid #e5ebf3;padding:18px;border-radius:12px}
[data-testid="stSidebar"]{background:#101e32}
[data-testid="stSidebar"] *{color:#e9f0fc}
h1,h2,h3{letter-spacing:-.035em}
.badge{display:inline-block;font-size:12px;letter-spacing:.06em;color:#2463eb;background:#eaf1ff;border-radius:20px;padding:6px 12px}
.quiet{color:#67758a;font-size:14px}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def service():return Scorer()

@st.cache_data(show_spinner=False,max_entries=20)
def evidence():return json.loads((service().results_dir/'results.json').read_text())

@st.cache_data(show_spinner=False,max_entries=12)
def scored(model,end,scenario_name,target,severity,imputation,tamper):
    return service().score({'model':model,'as_of_interval':end,'scenario':scenario_name,
        'meter_id':target,'severity':severity,'imputation':imputation,'tamper_event':tamper})

svc=service();result=evidence()
with st.sidebar:
    st.markdown('### ⚡ Grid-GNN')
    st.caption('CRYSTAL PARADIGM · NTL WORKBENCH')
    page=st.radio('Workspace',['Inspect','Replay & scenarios','Evidence','Responsible AI'],label_visibility='collapsed')
    st.divider()
    model=st.selectbox('Scoring model',['M0','M1','M2','M3','M4'],index=['M0','M1','M2','M3','M4'].index(result['selected_model']))
    descriptions={'M0':'Energy-balance ordering','M1':'Isolation Forest','M2':'Temporal RandomForest',
                  'M3':'RandomForest + graph summaries','M4':'Temporal graph neural network'}
    st.caption(descriptions[model])
    st.caption('Default selected on validation data. The test benchmark does not establish a GNN advantage.')
    st.divider()
    st.caption('Decision support only. Every flag requires field verification. No automatic disconnection or penalty.')
    st.caption('SYNTHETIC PROFILES · REFERENCE GRID')


def network_chart(response,selected):
    obs=svc.context
    risk={r['meter_id']:r for r in response['meters']}
    m=obs.cfg['meters_per_dt']
    ex=[];ey=[];px=[];py=[];rootx=[];rooty=[];rootlabels=[]
    for g,topo in enumerate(obs.topologies):
        xy={}
        for j,node in enumerate(topo.meter_nodes):
            row=obs.meters.iloc[g*m+j];xy[node]=(row.lon,row.lat)
        children={i:[] for i in range(topo.size)}
        for node in range(1,topo.size):children[topo.parent[node]].append(node)
        for node in range(topo.pole_count,-1,-1):
            coords=[xy[c] for c in children[node] if c in xy]
            if coords:xy[node]=(np.mean([v[0] for v in coords]),np.mean([v[1] for v in coords]))
            else:xy[node]=(obs.meters.iloc[g*m].lon,obs.meters.iloc[g*m].lat)
            if node: px.append(xy[node][0]);py.append(xy[node][1])
        rootx.append(xy[0][0]);rooty.append(xy[0][1]);rootlabels.append(f'DT {g}')
        for node in range(1,topo.size):
            a,b=xy[topo.parent[node]],xy[node]
            ex.extend([a[0],b[0],None]);ey.extend([a[1],b[1],None])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ex,y=ey,mode='lines',line=dict(color='#c9d5e5',width=1),hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter(x=px,y=py,mode='markers',marker=dict(size=4,color='#99a8ba'),name='Poles',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=obs.meters.lon,y=obs.meters.lat,mode='markers',name='Meters',
        marker=dict(size=[13 if i==selected else 8 for i in obs.meters.meter_id],
            color=[risk[i]['probability_simulated_theft'] for i in obs.meters.meter_id],
            colorscale=[[0,'#44b5ac'],[.45,'#e8b349'],[1,'#c94442']],cmin=0,cmax=1,
            colorbar=dict(title='Probability',thickness=12),line=dict(width=.7,color='white')),
        text=[f"{i}<br>Probability {risk[i]['probability_simulated_theft']:.2f}<br>Review flag: {risk[i]['inspection_flag']}" for i in obs.meters.meter_id],hovertemplate='%{text}<extra></extra>'))
    fig.add_trace(go.Scatter(x=rootx,y=rooty,mode='markers+text',text=rootlabels,textposition='top center',
        marker=dict(size=15,symbol='square',color='#2463eb'),name='DTs',hoverinfo='text'))
    fig.update_layout(height=440,margin=dict(l=5,r=5,t=20,b=5),paper_bgcolor='white',plot_bgcolor='white',
        xaxis=dict(visible=False),yaxis=dict(visible=False),legend=dict(orientation='h',y=-.03),hovermode='closest')
    return fig


def balance_chart(response,g,interval=False):
    df=pd.DataFrame(response['interval_packets'] if interval else response['daily_balance'])
    df=df[df.dt_id==g]
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    for col,title,color in [('input_kwh','DT input','#2463eb'),('imputed_consumer_kwh','Consumers, with imputation','#0b8c81'),
                             ('estimated_technical_kwh' if not interval else 'estimated_technical_kwh','Estimated technical loss','#a7b2c1')]:
        fig.add_trace(go.Scatter(x=df.ts,y=df[col],name=title,line=dict(color=color,width=2)),secondary_y=False)
    fig.add_trace(go.Scatter(x=df.ts,y=df.missing_fraction,name='Missing share',fill='tozeroy',
        line=dict(width=0,color='rgba(227,160,54,.2)'),fillcolor='rgba(227,160,54,.17)'),secondary_y=True)
    fig.update_yaxes(title_text='kWh',secondary_y=False)
    fig.update_yaxes(title_text='Missing share',range=[0,1],secondary_y=True)
    fig.update_layout(height=320,margin=dict(l=15,r=20,t=10,b=15),legend=dict(orientation='h',y=1.15),
                      paper_bgcolor='white',plot_bgcolor='white',hovermode='x unified')
    return fig


def render_inspection(response,selected,show_map=True):
    flagged=[r for r in response['meters'] if r['inspection_flag']]
    cols=st.columns(4)
    cols[0].metric('Meters in replay',len(response['meters']))
    cols[1].metric('Field reviews suggested',len(flagged))
    cols[2].metric('Unexplained DT energy',f"{sum(r['unexplained_kwh'] for r in response['dts']):,.1f} kWh")
    cols[3].metric('Mean missing share',f"{np.mean([r['missing_fraction'] for r in response['dts']]):.1%}")
    st.caption('Energy covers the current assessment window. Unexplained energy can include metering and mapping errors or upstream load.')
    if not response['operating_point']['validation_target_met']:
        st.info('This model did not meet the validation precision target; review flags are disabled. The ranking remains visible.')
    if show_map:
        left,right=st.columns([1.6,1])
        with left:
            st.subheader('Network overview')
            st.caption('Synthetic coordinates. Colours represent model probability, not confirmed theft.')
            st.plotly_chart(network_chart(response,selected),use_container_width=True,key='network_'+page)
        with right:
            row=next(r for r in response['meters'] if r['meter_id']==selected)
            st.subheader(selected)
            st.markdown('**Field review suggested**' if row['inspection_flag'] else '**Not flagged at this operating point**')
            st.metric('Synthetic-case probability',f"{row['probability_simulated_theft']:.1%}")
            for reason in row['reason_codes']:st.write('• '+reason)
            st.caption('Reason codes summarize evidence; they cannot establish intent or occupancy.')
            if response['selected_meter_explanation']:
                exp=response['selected_meter_explanation']
                st.dataframe(pd.DataFrame({'Feature group':exp['group_names'],'SHAP contribution':exp['values'][0]}),hide_index=True)
                st.caption(exp['method'])
            if response['gnn_global_permutation_importance']:
                st.dataframe(pd.DataFrame(response['gnn_global_permutation_importance']),hide_index=True)
                st.caption('Global validation permutation importance. This is not a causal explanation of this individual meter.')
    st.subheader('Inspection ordering')
    table=pd.DataFrame(response['meters'])
    table['reason']=table.reason_codes.map(lambda x:' '.join(x[:3]))
    shown=table[['rank','meter_id','dt_id','inspection_flag','probability_simulated_theft','candidate_kwh_allocation','missing_fraction','reason']]
    st.dataframe(shown,hide_index=True,width="stretch",
        column_config={'probability_simulated_theft':st.column_config.ProgressColumn('Probability',min_value=0,max_value=1,format='%.2f'),
                       'candidate_kwh_allocation':st.column_config.NumberColumn('Candidate kWh share',format='%.1f')})
    st.caption('Candidate kWh is a posterior × expected-load allocation of the DT residual. It is not a verified theft amount or recoverable revenue.')
    g=st.selectbox('DT detail',[r['dt_id'] for r in response['dts']],key='dt_detail_'+page)
    st.plotly_chart(balance_chart(response,g),use_container_width=True,key='balance_'+page)
    st.download_button('Download inspection ordering',shown.to_csv(index=False),'inspection_ordering.csv','text/csv')


if page=='Inspect':
    st.markdown('<span class="badge">REPLAYED TELEMETRY · DECISION SUPPORT</span>',unsafe_allow_html=True)
    st.title('Find the evidence before the visit')
    st.write('Inspect meter changes alongside DT energy balance, telemetry quality, and alternative explanations.')
    target=st.selectbox('Inspect a meter',svc.context.meters.meter_id.tolist(),key='inspect_meter')
    response=scored(model,svc.context.kwh.shape[1],'none',target,.7,True,False)
    st.session_state['last_response']=response
    render_inspection(response,target)

elif page=='Replay & scenarios':
    st.markdown('<span class="badge">REPLAYED TELEMETRY · SYNTHETIC INTERVENTIONS</span>',unsafe_allow_html=True)
    st.title('One interval at a time')
    st.write('Advance stored telemetry through the same scorer. Compare honest absence, under-recording, upstream load, and missing packets.')
    minimum=svc.context.cfg['baseline_days']+7;maximum=svc.context.cfg['days']
    if 'cursor' not in st.session_state:st.session_state.cursor=(maximum-3)*96
    def seek():st.session_state.cursor=st.session_state.seek_day*96
    def advance(amount):st.session_state.cursor=min(svc.context.kwh.shape[1],st.session_state.cursor+amount)
    st.slider('Seek to day',minimum,maximum,maximum-3,key='seek_day',on_change=seek)
    a,b,c=st.columns([1,1,2])
    a.button('Next interval',on_click=advance,args=(1,),width="stretch")
    b.button('Next day',on_click=advance,args=(96,),width="stretch")
    playing=c.toggle('Play stored intervals',value=False)
    c.caption('Replay advances one stored interval per refresh; it is not a live DISCOM feed.')
    controls=st.columns(3)
    scen=controls[0].selectbox('Inject scenario',['none','vacancy','theft','upstream_hooking','ami_dropout'])
    target=controls[1].selectbox('Target meter / associated DT',svc.context.meters.meter_id.tolist(),key='scenario_meter')
    severity=controls[2].slider('Suppression fraction / hooking kW',.1,.9,.7,.1)
    imputation=st.checkbox('Use causal imputation',value=True)
    tamper=st.checkbox('Inject a tamper-like event with theft',value=False)
    st.caption('Interventions affect the final replay week. Vacancy changes physical draw and voltage; theft changes the recorded energy. Dropout is masked, not a measured zero.')
    @st.fragment(run_every=1. if playing else None)
    def replay_panel():
        end=st.session_state.cursor
        response=scored(model,end,scen,target,severity,imputation,tamper)
        st.session_state['last_response']=response
        st.markdown(f"**Replay cursor:** {response['as_of_time']} · interval {end}")
        g=int(svc.context.meters.loc[svc.context.meters.meter_id==target,'dt_id'].iloc[0])
        st.plotly_chart(balance_chart(response,g,interval=True),use_container_width=True,key='interval_replay')
        render_inspection(response,target,show_map=True)
        if playing and end<svc.context.kwh.shape[1]:st.session_state.cursor=end+1
    replay_panel()

elif page=='Evidence':
    st.markdown('<span class="badge">HELD-OUT DTs · REPRODUCIBLE RESULTS</span>',unsafe_allow_html=True)
    st.title('What the benchmark supports')
    st.info('Graph-advantage thesis: '+result['thesis_verdict']+'. Architecture preference does not override measured results.')
    st.write(f"{result['config']['n_dts']*result['config']['meters_per_dt']:,} synthetic meters per seed; {result['config']['n_dts']} DTs; {result['config']['days']} days; {result['config']['seeds']} independent seeds.")
    def value(v,percent=False):
        if v['mean'] is None:return 'not defined'
        if percent:return f"{v['mean']:.1%} [{v['ci_low']:.1%}, {v['ci_high']:.1%}]"
        return f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    rows=[]
    for n in descriptions:
        h=result['headline'][n]
        rows.append({'Model':n,'Method':descriptions[n],'Precision@budget':value(h['precision_at_k']),
            'Recall@budget':value(h['recall_at_k']),'PR-AUC':value(h['pr_auc']),
            'Vacancy FPR':value(h['vacant_fpr'],True)})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.caption('Brackets are bootstrap confidence intervals resampling seeds and whole DTs. Zero FPR can reflect abstention; inspect operating-point coverage.')
    st.image(str(svc.results_dir/'figures/paired_differences.png'))
    st.image(str(svc.results_dir/'figures/stress_curves.png'))
    a,b=st.columns(2)
    with a:
        st.subheader('Voltage pathway')
        st.write('Supported incremental benefit: '+('yes' if result['voltage_verdict']['supported'] else 'no'))
        st.write(value(result['voltage_verdict']['pr_auc_delta']))
    with b:
        st.subheader('Localisation boundary')
        st.write(f"Meter-visible NTL: {result['ntl_visibility']['meter_visible_fraction']:.1%}; invisible upstream NTL: {result['ntl_visibility']['invisible_fraction']:.1%}.")
        st.write(result['ntl_visibility']['localisation_limit'])
    st.subheader('Independent real-data check')
    real=result['real_data']['sgcc']
    if real['status']=='completed':
        st.write(f"SGCC: {real['customer_rows_observed']:,} customers, daily Chinese data, no topology. PR-AUC {value(real['metrics']['pr_auc'])}.")
        st.caption('This checks a temporal baseline only; it does not validate grid localisation or India transfer.')
    else:st.write(real)
    st.write('Real LCL / SimBench check: '+result['real_data']['lcl_simbench']['status']+' — '+result['real_data']['lcl_simbench']['reason'])
    st.download_button('Download results.json',(svc.results_dir/'results.json').read_bytes(),'results.json','application/json')
    st.download_button('Download claims register',(svc.results_dir/'claims_register.csv').read_bytes(),'claims_register.csv','text/csv')

else:
    st.markdown('<span class="badge">HUMAN REVIEW · TRANSPARENT LIMITS</span>',unsafe_allow_html=True)
    st.title('Protect honest customers')
    st.write('A flag is a request to review evidence. Vacancy, seasonal absence, meter failure and bad telemetry can resemble theft.')
    a,b,c=st.columns(3)
    with a:
        st.subheader('Human authority')
        st.write('Field inspectors verify every flag. No automatic disconnection, penalty, billing change or accusation is supported.')
    with b:
        st.subheader('Privacy & retention')
        st.write('Pseudonymised IDs and minimum fields. Keep identity lookup with the DISCOM. Apply access controls and a documented retention limit before field use.')
    with c:
        st.subheader('Appeal & feedback')
        st.write('Consumers use the DISCOM grievance channel. A named reviewer must reconsider the evidence and record corrections; the model cannot decide an appeal.')
    st.subheader('Who gets falsely flagged?')
    attribute=st.selectbox('Audit slice',['category','area_type','consumption_band','archetype'])
    audit=pd.DataFrame(result['fairness'][model]);audit=audit[audit.attribute==attribute]
    st.dataframe(audit,hide_index=True,width="stretch")
    st.caption('Simulated subgroup audit with observed counts. It is not demographic fairness certification, and small groups can be unstable.')
    st.subheader('Operating policy')
    st.write('Thresholds target precision on validation DTs and may abstain. The target is not a guarantee of test or field precision. Review vacancy evidence and data quality before escalating a case.')
    st.write('Monitor balance distributions, missingness, flag rates and review outcomes. Include an unbiased sample of unflagged meters in pilots to reduce inspection-feedback bias. Segment thresholds require documented field evidence and approval.')
    st.write('India’s DPDP Act and applicable Rules require legal review before deployment. This synthetic demo is not a legal compliance certification.')
    st.subheader('Known limits')
    st.write('Synthetic consumption; non-Indian reference networks; simplified neutral physics; uncertain technical loss; noisy labels; imperfect mapping; voltage resolution; unmetered upstream load; validation reuse for calibration.')
    st.caption('Score calls write a local audit entry with the model version, input hash, flags and reasons. Production access control and grievance integration are pilot work.')
