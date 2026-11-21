"""Append-only, isolated IntelliSignal diagnosis and selection experiment.

Use --prepare before --validation. No final-test results can be opened without
an immutable freeze.json. No code in the production environment is changed.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, pickle, random, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from agents.dqn_agent import DQNAgent
from agents.dqn_config import load_dqn_config
from rl.environment import IntelliSignalEnv
from rl.state import APPROACH_ORDER
from simulation.agent_simulation import QUIET_SUMO_ARGS, run_fixed_time_episode
from utils.metrics import MetricsCollector
import utils.scenario_builder as builder

SCENARIOS = ['heavy', 'moderate']
SHORTLIST = ['incumbent_best','incumbent_final','prior_best','prior_final','prior_ep100','prior_ep200']
METRICS = ['average_waiting_time','average_queue_length','max_queue_length','throughput','teleports','vehicles_departed','vehicles_still_in_network']

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, value, exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)

def checkpoint(scenario, policy):
    if policy.startswith('incumbent_'):
        return ROOT/'models'/('dqn_'+scenario)/(policy.split('_',1)[1]+'.pt')
    name={'prior_best':'best.pt','prior_final':'final.pt','prior_ep100':'checkpoint_ep100.pt','prior_ep200':'checkpoint_ep200.pt'}[policy]
    return ROOT/'experiments/dqn_300_20261117'/scenario/'models'/name

def isolate_inputs(scenario, partition):
    base=HERE/'generated'/partition/scenario
    for attribute,folder in [('ROUTES_DIR','routes'),('ADDITIONAL_DIR','additional'),('SUMOCFG_DIR','config')]:
        setattr(builder,attribute,str(base/folder))

def protected():
    roots=['models','config','results','sumo','verification','experiments']
    return {p.relative_to(ROOT).as_posix():digest(p) for folder in roots for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and HERE not in p.parents and '__pycache__' not in p.parts}

def seed_audit():
    # Recognized fields are parsed, not digit substrings in hashes or metrics.
    observations=[]; checked=[]
    names={'seed','seeds','sumo_seed','episode_seed','training_seed','training_seeds','training_traffic_seeds','traffic_seed','traffic_seeds','reference_seeds','holdout_seeds','validation_seeds','final_test_seeds'}
    def walk(value,path,key=''):
        if isinstance(value,dict):
            for k,v in value.items(): walk(v,path,k)
        elif isinstance(value,list):
            for v in value: walk(v,path,key)
        elif key in names and isinstance(value,(int,float)) and int(value)==value:
            observations.append({'path':path,'field':key,'seed':int(value)})
    for folder in ['experiments','verification','results','models','config']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or HERE in p.parents: continue
            rel=p.relative_to(ROOT).as_posix()
            if p.suffix=='.json':
                try: walk(json.loads(p.read_text(encoding='utf-8')),rel); checked.append(rel)
                except (ValueError,UnicodeError): pass
            elif p.suffix=='.csv':
                try:
                    with p.open(newline='',encoding='utf-8-sig') as f:
                        for row in csv.DictReader(f):
                            for k,v in row.items():
                                if k in names and v and v.lstrip('-').isdigit(): observations.append({'path':rel,'field':k,'seed':int(v)})
                    checked.append(rel)
                except (ValueError,UnicodeError): pass
    hits=[o for o in observations if 2001<=o['seed']<=2010 or 3001<=o['seed']<=3020]
    return {'scope':'Parsed recognized seed fields including sumo_seed,episode_seed,training_traffic_seeds in JSON and CSV under experiments,verification,results,models,config; binary checkpoint configs additionally enumerated in manifest. Cannot prove use absent from external/deleted records.','files_checked':checked,'reserved_seed_hits':hits,'observed_seeds':sorted(set(o['seed'] for o in observations)),'training_traffic_union_expected':[42,343]}

def prepare():
    HERE.mkdir(parents=True,exist_ok=True)
    if (HERE/'manifest.json').exists(): raise RuntimeError('Predeclaration already exists; preserve it.')
    audit=seed_audit(); write_json(HERE/'seed_audit.json',audit,True)
    if audit['reserved_seed_hits']: raise RuntimeError('Reserved seeds were found; audit before evaluation.')
    entries=[]
    for scenario in SCENARIOS:
        for policy in SHORTLIST:
            p=checkpoint(scenario,policy); c=torch.load(p,map_location='cpu',weights_only=False)
            entries.append({'scenario':scenario,'policy':policy,'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'episode':c['episode'],'seed':c['seed'],'config':c['config']})
    manifest={
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'state':'predeclared before new validation',
        'duration_seconds':1800,'training_seeds':[42,43,44],'training_traffic_union':[42,343],
        'historical_examined_traffic_seeds':list(range(1001,1011)), 'validation_seeds':list(range(2001,2011)), 'reserved_final_seeds':list(range(3001,3021)),
        'scenarios':SCENARIOS,'torch_threads':1,'shortlist_order':SHORTLIST,'checkpoints':entries,
        'validation_budget_episodes':160,
        'selection':'Candidate universe incumbent_final,prior_best,prior_final,prior_ep100,prior_ep200. First restrict to validation-gate-eligible checkpoints if any; rank by lowest mean unrounded waiting, then mean unrounded queue, then highest throughput, then shortlist order. If none eligible rank all and retain only as diagnostic challenger. No exploratory reward selection.',
        'new_training_trigger':'Heavy: run bounded matched control and lower-LR if no candidate-universe checkpoint satisfies validation gate. Moderate: do not retrain if incumbent_best/prior_best effective agreement on their common-observation probes is >=99% and no validation-superior checkpoint; otherwise same bounded recipes.',
        'validation_gate':'Lower mean waiting, waiting wins on >5/10 traffic seeds, no worse mean queue/throughput/teleports vs incumbent_best.',
        'bounded_recipes':{'control':{'learning_rate':0.001,'gamma':0.9},'lower_lr':{'learning_rate':0.0003,'gamma':0.9}},
        'training_budget':{'episodes_per_run':300,'duration_seconds':1800,'heavy_runs_max':6,'moderate_runs_max':6,'total_episodes_max':3600,'epsilon_decay_steps':27000,'epsilon_start':1.0,'epsilon_end':0.05,'training_checkpoint_shortlist':['best.pt','checkpoint_ep100.pt','checkpoint_ep200.pt','final.pt'],'no_other_hyperparameter_changes':True},
        'training_selection':'Same validation waiting/queue/throughput/order procedure for each training seed, each recipe. Choose recipe using mean waiting over training seeds x traffic seeds subject to validation guardrails. Keep each training seed as a separate replicate. No cherry-picking training seed.',
        'final_freeze':'Write frozen per-scenario recipe and every replicate checkpoint path+SHA256 before opening ANY seed3001-3020. No change after final results.',
        'promotion':'All: lower mean waiting, waiting wins on >10/20 paired traffic seeds, no worse mean queue,throughput,teleports vs incumbent_best. For new recipe also all guardrails vs matched validation-selected control, waiting improves in >=2/3 training seeds, and upper bound of 95% t CI for training-seed-mean waiting delta vs control below0. Do not change thresholds after results. No automatic replacement.',
        'uncertainty':'Report paired deltas, traffic-seed SD and training-seed means/SD separately. Historical existing recipes have one training seed, so training variability unavailable; do not claim independent retraining reproducibility.',
        'rng_isolation':'Evaluation separate process from training; greedy argmax only, no learning/replay writes. Capture/restore Python,NumPy,Torch global RNG around agent creation and evaluations; assert unchanged evaluation RNG fingerprints.',
        'recovery':'Reuse preserved resumable_training.py full-state episode-boundary snapshots + bounded PermissionError retry without production modifications.',
        'source_sha256':digest(__file__)
    }
    write_json(HERE/'manifest.json',manifest,True)
    write_json(HERE/'protected_before.json',protected(),True)
    print('PREDECLARED',len(entries),'checkpoints',flush=True)

def rng_state(): return (random.getstate(),np.random.get_state(),torch.get_rng_state())
def rng_restore(state): random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
def rng_hash(): return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().tolist()))).hexdigest()

def load_agents(scenario,entries=None):
    state=rng_state()
    try:
        agents={policy:DQNAgent.load(str(checkpoint(scenario,policy)),scenario=scenario) for policy in SHORTLIST}
        for policy,agent in agents.items():
            agent._diagnostic_gamma=float(DQNAgent._read_checkpoint(str(checkpoint(scenario,policy)))['config']['gamma'])
        # Precisely match the historical seed42 untrained control.
        agents['untrained']=DQNAgent(observation_size=10,scenario=scenario,seed=42,trainable=False)
        return agents
    finally: rng_restore(state)

def raw_vector(env,info):
    return [info['queue_per_approach'][a]/env.state_extractor.max_queue for a in APPROACH_ORDER]+[info['waiting_per_approach'][a]/env.state_extractor.max_waiting for a in APPROACH_ORDER]+[float(info['phase']),info['green_elapsed']/env.state_extractor.max_green]

def policy_episode(scenario,seed,policy,agents,partition):
    isolate_inputs(scenario,partition)
    before=rng_hash(); env=IntelliSignalEnv(scenario=scenario,episode_duration=1800,seed=seed,extra_sumo_args=QUIET_SUMO_ARGS)
    rows=[]; collector=None; agent=agents[policy]; gamma=getattr(agent,'_diagnostic_gamma',float(agent.config.gamma))
    try:
        obs,info=env.reset(seed=seed); collector=MetricsCollector(step_length=env.step_length,conn=env.connection)
        env.step_callback=collector.collect_step
        while True:
            values=agent.q_values(obs); requested=int(np.argmax(values)); pre=info
            probes={p:int(np.argmax(a.q_values(obs))) for p,a in agents.items()}
            effective={p:int(env.controller.resolve(a)[0]) for p,a in probes.items()}
            nxt,reward,terminated,truncated,info=env.step(requested)
            with torch.no_grad(): next_q=float(agent.target_net(torch.from_numpy(nxt).unsqueeze(0)).max().item())
            row={'decision':len(rows),'time':pre['simulation_time'],'observation':obs.tolist(),'raw_normalized':raw_vector(env,pre),'phase':pre['phase'],'green_elapsed':pre['green_elapsed'],'requested_action':requested,'effective_action':info['effective_action'],'switch_blocked':info['switch_blocked'],'forced_switch':info['forced_switch'],'reward':reward,'q_keep':float(values[0]),'q_switch':float(values[1]),'q_gap':float(abs(values[0]-values[1])),'target_next_max':next_q,'td_error_offline':float(reward+gamma*next_q*(not terminated)-values[requested]),'terminated':terminated,'truncated':truncated,'probes_requested':probes,'probes_effective':effective,**info['reward_components']}
            rows.append(row); obs=nxt
            if terminated or truncated: break
        result=collector.summary(scenario)
        result.update({'waiting_unrounded':sum(collector._waiting_time.values())/len(collector._waiting_time),'queue_unrounded':collector._queue_sum/collector.steps,'episode_return':sum(r['reward'] for r in rows),'waiting_component':sum(r['waiting_penalty'] for r in rows),'queue_component':sum(r['queue_penalty'] for r in rows),'fairness_component':sum(r['fairness_penalty'] for r in rows),'seed':seed,'policy':policy,'decisions':len(rows),'elapsed_seconds':1800,'rng_unchanged':before==rng_hash(),'td_definition':'Offline chronological greedy rollout TD; stored target, checkpoint gamma; bootstrap at truncation, mask terminated only. Not training minibatch errors.'})
        assert result['rng_unchanged']
        assert len(rows)==180 and collector.steps==1800 and result['simulation_time']==1800
        assert abs(result['episode_return']-sum(result[c] for c in ['waiting_component','queue_component','fairness_component']))<1e-7
        return {'summary':result,'decisions':rows,'timeseries':collector.timeseries}
    finally: env.close()

def evaluate(partition,scenarios=SCENARIOS):
    manifest=json.loads((HERE/'manifest.json').read_text())
    if partition=='final': raise RuntimeError('Final evaluation requires separate frozen selection implementation.')
    for entry in manifest['checkpoints']:
        assert digest(ROOT/entry['path'])==entry['sha256'],'Checkpoint changed since declaration'
    for scenario in scenarios:
        agents=load_agents(scenario)
        agent_before={p:(a.learn_steps,len(a.memory),hashlib.sha256(pickle.dumps({k:v.tolist() for k,v in a.policy_net.state_dict().items()})).hexdigest()) for p,a in agents.items()}
        for policy in SHORTLIST+['untrained','fixed_time']:
            for seed in manifest['validation_seeds']:
                out=HERE/partition/scenario/policy/(str(seed)+'.json')
                if out.exists(): continue
                started=time.time()
                if policy=='fixed_time':
                    isolate_inputs(scenario,partition); s,ts=run_fixed_time_episode(scenario,seed,1800)
                    data={'summary':dict(s,policy=policy,seed=seed),'timeseries':ts}
                else: data=policy_episode(scenario,seed,policy,agents,partition)
                data['summary']['wall_seconds']=round(time.time()-started,3)
                write_json(out,data,True)
                print(partition,scenario,policy,seed,data['summary']['average_waiting_time'],data['summary']['wall_seconds'],flush=True)
        agent_after={p:(a.learn_steps,len(a.memory),hashlib.sha256(pickle.dumps({k:v.tolist() for k,v in a.policy_net.state_dict().items()})).hexdigest()) for p,a in agents.items()}
        assert agent_before==agent_after,'Evaluation mutated training state'
        write_json(HERE/partition/scenario/'inference_state_unchanged.json',{'before':agent_before,'after':agent_after},True)

if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true');parser.add_argument('--validation',action='store_true');parser.add_argument('--scenario',choices=SCENARIOS)
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.validation: evaluate('validation',[args.scenario] if args.scenario else SCENARIOS)
