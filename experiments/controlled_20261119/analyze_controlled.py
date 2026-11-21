"""Analyze saved records only. Never opens the reserved final traffic seeds."""
import csv, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_controlled as run

def save_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def metrics(summary):
    return {'waiting':summary['waiting_unrounded'] if 'waiting_unrounded' in summary else summary['average_waiting_time'], 'queue':summary['queue_unrounded'] if 'queue_unrounded' in summary else summary['average_queue_length'], 'throughput':summary['throughput'],'teleports':summary['teleports']}

def gate(candidate,incumbent):
    cs={s['seed']:metrics(s) for s in candidate}; bs={s['seed']:metrics(s) for s in incumbent}
    seeds=sorted(set(cs)&set(bs)); assert len(seeds)==10
    delta={m:float(np.mean([cs[s][m]-bs[s][m] for s in seeds])) for m in ['waiting','queue','throughput','teleports']}
    wins=sum(cs[s]['waiting']<bs[s]['waiting']-1e-10 for s in seeds)
    checks={'waiting_lower':delta['waiting']<-1e-10,'waiting_majority':wins>5,'queue_no_worse':delta['queue']<=1e-10,'throughput_no_worse':delta['throughput']>=-1e-10,'teleports_no_worse':delta['teleports']<=1e-10}
    return {'eligible':all(checks.values()),'checks':checks,'paired_mean_delta':delta,'waiting_wins':wins,'traffic_seed_count':len(seeds)}

def analyze():
    records=[];diagnostics=[];cross=[];groups={}
    for scenario in run.SCENARIOS:
        groups[scenario]={}
        for policy in run.SHORTLIST+['untrained','fixed_time']:
            paths=list((HERE/'validation'/scenario/policy).glob('*.json'))
            if len(paths)!=10: raise RuntimeError(f'Incomplete {scenario}/{policy}: {len(paths)}/10')
            data=[json.loads(p.read_text()) for p in sorted(paths)]
            summaries=[x['summary'] for x in data]; groups[scenario][policy]=summaries
            for d in data:
                s=d['summary'];records.append(dict({k:v for k,v in s.items() if isinstance(v,(str,int,float,bool))},stage='validation',scenario=scenario,policy=policy,recipe='historical' if policy in run.SHORTLIST else policy,train_seed=42 if policy in run.SHORTLIST else None,initialization_seed=42 if policy=='untrained' else None,traffic_seed=s['seed']))
            if policy=='fixed_time': continue
            rows=[r for d in data for r in d['decisions']]
            raw=np.array([r['raw_normalized'] for r in rows]);q=np.array([[r['q_keep'],r['q_switch']] for r in rows]);td=np.array([r['td_error_offline'] for r in rows])
            greens=[r['green_elapsed'] for r in rows if r['effective_action']==1]
            incomplete=[d['decisions'][-1]['green_elapsed']+(6 if d['decisions'][-1]['effective_action'] else 10) for d in data]
            # First green starts at t=0. Every effective switch ends the prior
            # green and starts a new six-second green after four-second yellow.
            incomplete=[6 if d['decisions'][-1]['effective_action'] else d['decisions'][-1]['green_elapsed']+10 for d in data]
            diagnostic={'scenario':scenario,'policy':policy,'decisions':len(rows),'requested_switches':sum(r['requested_action'] for r in rows),'effective_switches':sum(r['effective_action'] for r in rows),'blocked_switches':sum(r['switch_blocked'] for r in rows),'forced_switches':sum(r['forced_switch'] for r in rows),'completed_green_duration_mean':float(np.mean(greens)),'completed_green_duration_min':min(greens),'completed_green_duration_max':max(greens),'completed_green_duration_counts':{str(x):greens.count(x) for x in sorted(set(greens))},'censored_final_green_durations':incomplete,'clip_fraction_by_dim':np.mean(raw>1,axis=0).tolist(),'saturation_fraction_by_dim':np.mean(raw>=1,axis=0).tolist(),'raw_normalized_max_by_dim':raw.max(axis=0).tolist(),'q_min':float(q.min()),'q_max':float(q.max()),'q_mean':float(q.mean()),'q_gap_mean':float(np.mean(np.abs(q[:,0]-q[:,1]))),'td_mean':float(td.mean()),'td_abs_mean':float(np.abs(td).mean()),'td_abs_p95':float(np.percentile(np.abs(td),95)),'td_abs_max':float(np.abs(td).max()),'td_label':'Offline greedy evaluation TD using checkpoint target network, not original training minibatch TD','mean_waiting_reward_component':float(np.mean([s['waiting_component'] for s in summaries])),'mean_queue_reward_component':float(np.mean([s['queue_component'] for s in summaries])),'mean_fairness_reward_component':float(np.mean([s['fairness_component'] for s in summaries]))}
            diagnostic['waiting_fraction_of_reward_cost']=diagnostic['mean_waiting_reward_component']/(diagnostic['mean_waiting_reward_component']+diagnostic['mean_queue_reward_component'])
            diagnostics.append(diagnostic)
            for a,b in itertools.combinations(run.SHORTLIST,2):
                cross.append({'scenario':scenario,'rollout_policy':policy,'a':a,'b':b,'decisions':len(rows),'requested_agreement':float(np.mean([r['probes_requested'][a]==r['probes_requested'][b] for r in rows])),'effective_agreement':float(np.mean([r['probes_effective'][a]==r['probes_effective'][b] for r in rows]))})
    selection={}
    correlations={}
    reversals=[]
    for scenario,g in groups.items():
        incumbent=g['incumbent_best']; candidates=[p for p in run.SHORTLIST if p!='incumbent_best']
        gates={p:gate(g[p],incumbent) for p in candidates};eligible=[p for p in candidates if gates[p]['eligible']]
        rank=lambda p:(np.mean([metrics(s)['waiting'] for s in g[p]]),np.mean([metrics(s)['queue'] for s in g[p]]),-np.mean([metrics(s)['throughput'] for s in g[p]]),run.SHORTLIST.index(p))
        selected=min(eligible or candidates,key=rank)
        agreement=[c['effective_agreement'] for c in cross if c['scenario']==scenario and c['rollout_policy'] in ['incumbent_best','prior_best'] and c['a']=='incumbent_best' and c['b']=='prior_best']
        agreement=float(np.mean(agreement))
        training_needed=not bool(eligible) and (scenario=='heavy' or agreement<0.99)
        selection[scenario]={'candidate':selected,'validation_gate':gates[selected],'all_candidate_gates':gates,'incumbent_prior_best_effective_common_observation_agreement':agreement,'new_training_needed':training_needed,'reason':'No validation-gate-eligible checkpoint; trigger controlled training' if training_needed else ('Validation-gate-eligible checkpoint available; avoid retraining' if eligible else 'Moderate effective agreement>=99%; no validation superiority, preserve incumbent and avoid retraining')}
        flattened=[s for p in run.SHORTLIST for s in g[p]]
        reward=np.array([s['episode_return'] for s in flattened]); waits=np.array([metrics(s)['waiting'] for s in flattened]); queues=np.array([metrics(s)['queue'] for s in flattened]); through=np.array([s['throughput'] for s in flattened])
        correlations[scenario]={'unit':'checkpoint-traffic-seed rollout; correlated repeated seeds/checkpoints, descriptive only','n':len(flattened),'reward_vs_waiting_pearson':float(np.corrcoef(reward,waits)[0,1]),'reward_vs_queue_pearson':float(np.corrcoef(reward,queues)[0,1]),'reward_vs_throughput_pearson':float(np.corrcoef(reward,through)[0,1])}
        for a,b in itertools.combinations(run.SHORTLIST,2):
            ad={s['seed']:s for s in g[a]};bd={s['seed']:s for s in g[b]}
            for seed in ad:
                ra,rb=ad[seed],bd[seed]
                if (ra['episode_return']-rb['episode_return'])*(metrics(ra)['waiting']-metrics(rb)['waiting'])>1e-8:
                    reversals.append({'scenario':scenario,'traffic_seed':seed,'policy_a':a,'policy_b':b,'reward_a':ra['episode_return'],'reward_b':rb['episode_return'],'waiting_a':metrics(ra)['waiting'],'waiting_b':metrics(rb)['waiting'],'throughput_a':ra['throughput'],'throughput_b':rb['throughput']})
    save_csv(HERE/'validation_metrics.csv',records)
    save_csv(HERE/'policy_agreement.csv',cross)
    run.write_json(HERE/'diagnostics.json',{'policies':diagnostics,'reward_correlations':correlations,'rank_reversal_examples':reversals[:20],'rank_reversal_count':len(reversals),'measurement_definitions':{'reported_waiting':'Per-vehicle latest accumulated waiting, mean over all inserted vehicles','reward_waiting':'Decision-interval average of current incoming-lane waiting sums','queue':'Mean halting vehicles on incoming lanes; exact same per-second queue samples used in reward and collector','throughput':'Arrivals; not a direct reward term','clipping':'Raw normalized value >1; exact-boundary saturation>=1 reported separately. Phase dimension is categorical, not clipping concern.','greens':'Completed actual green seconds at effective switches; end-of-episode ongoing greens reported separately as censored','safety_observation':'max_green is checked at decision boundaries; 66-second completed green can occur from6+6*10. Safety rules preserved.'}})
    run.write_json(HERE/'selection.json',selection)
    weight_comparison={}
    for scenario in run.SCENARIOS:
        policies={}
        for policy in run.SHORTLIST:
            state=torch.load(run.checkpoint(scenario,policy),map_location='cpu',weights_only=False)['policy_state_dict']
            h=hashlib.sha256()
            for name,tensor in sorted(state.items()):
                array=tensor.detach().cpu().numpy()
                h.update(name.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
            policies[policy]={'policy_tensor_sha256':h.hexdigest(),'state':state}
        trace=[]
        for seed in range(2001,2011):
            a=json.loads((HERE/'validation'/scenario/'incumbent_best'/(str(seed)+'.json')).read_text())['decisions'];b=json.loads((HERE/'validation'/scenario/'prior_best'/(str(seed)+'.json')).read_text())['decisions']
            trace.append({'traffic_seed':seed,'decisions':len(a),'requested_sequence_equal':[r['requested_action'] for r in a]==[r['requested_action'] for r in b],'effective_sequence_equal':[r['effective_action'] for r in a]==[r['effective_action'] for r in b],'phase_sequence_equal':[r['phase'] for r in a]==[r['phase'] for r in b],'observation_sequence_equal':[r['observation'] for r in a]==[r['observation'] for r in b]})
        sa=policies['incumbent_best']['state'];sb=policies['prior_best']['state']
        weight_comparison[scenario]={'tensor_digests':{p:details['policy_tensor_sha256'] for p,details in policies.items()},'incumbent_prior_best_all_policy_tensors_equal':all(torch.equal(sa[k],sb[k]) for k in sa),'incumbent_prior_best_max_abs_parameter_difference':max(float((sa[k]-sb[k]).abs().max()) for k in sa),'actual_trace_comparison':trace,'scope':'Measured validation rollout states only; equality here does not imply equality on all possible observations.'}
    run.write_json(HERE/'weight_comparison.json',weight_comparison)
    print(json.dumps(selection,indent=2))

if __name__=='__main__': analyze()
