"""Freeze selection, then evaluate the reserved final set exactly once."""
from __future__ import annotations
import argparse, csv, hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import run_controlled as base
from analyze_controlled import metrics, save_csv
from artifact_io import write_json, valid_or_preserve, valid_episode
from agents.dqn_agent import DQNAgent
from simulation.agent_simulation import run_fixed_time_episode

HERE=base.HERE
FINAL_SEEDS=list(range(3001,3021))
FIELDS=['average_waiting_time','average_queue_length','max_queue_length','throughput','teleports','vehicles_departed','vehicles_still_in_network']

def freeze():
    path=HERE/'freeze.json'
    if path.exists():
        frozen=json.loads(path.read_text());verify_freeze(frozen);print('EXISTING_FREEZE_VERIFIED');return
    if any((HERE/'final').rglob('*.json')): raise RuntimeError('Final results exist before freeze')
    selection=json.loads((HERE/'selection.json').read_text()); recipes={}
    if (HERE/'recipe_selection.json').exists(): recipes=json.loads((HERE/'recipe_selection.json').read_text())
    entries={};eligibility={}
    for scenario in base.SCENARIOS:
        entries[scenario]=[]
        inc=base.checkpoint(scenario,'incumbent_best')
        entries[scenario].append({'label':'incumbent','recipe':'incumbent','train_seed':42,'path':inc.relative_to(base.ROOT).as_posix(),'sha256':base.digest(inc)})
        if selection[scenario]['new_training_needed']:
            chosen=recipes[scenario]['selected_recipe']
            eligible=chosen=='lower_lr' and recipes[scenario]['lower_lr_vs_control_gate']['eligible'] and recipes[scenario]['lower_lr_vs_incumbent_gate']['eligible'] and recipes[scenario]['training_seeds_improved']>=2
            eligibility[scenario]={'validation_eligible':bool(eligible),'diagnostic_only':not bool(eligible),'reason':'Frozen matched-recipe validation criteria; control is a diagnostic challenger if lower_lr does not qualify.'}
            for role,recipe in [('selected',chosen),('control','control')]:
                for selected in recipes[scenario]['replicates'][recipe]:
                    entries[scenario].append({'label':role+'_seed'+str(selected['train_seed']),'recipe':recipe,'role':role,'train_seed':selected['train_seed'],'path':selected['path'],'sha256':selected['sha256']})
        else:
            chosen=selection[scenario]['candidate'];candidate=base.checkpoint(scenario,chosen)
            eligible=selection[scenario]['validation_gate']['eligible']
            eligibility[scenario]={'validation_eligible':bool(eligible),'diagnostic_only':not bool(eligible),'reason':'Original checkpoint validation gate; diagnostic-only checkpoints cannot be rescued by final results.'}
            entries[scenario].append({'label':'selected','recipe':chosen,'role':'selected','train_seed':42,'path':candidate.relative_to(base.ROOT).as_posix(),'sha256':base.digest(candidate)})
    sources={p.name:base.digest(p) for p in [Path(__file__).resolve(),HERE/'run_controlled.py',HERE/'artifact_io.py',HERE/'analyze_controlled.py',HERE/'run_final_parallel.py']}
    amendment={'reason':'Historical compare_controllers.py untrained uses initialization seed0; frozen validation runner used initialization seed42. Preserve those measured validation controls and identify them correctly. Add historical seed0 control for final evaluation while retaining seed42 as a separate additional control.','historical_source':{'path':'evaluation/compare_controllers.py','sha256':base.digest(base.ROOT/'evaluation/compare_controllers.py')},'final_untrained_controls':[{'label':'untrained_historical0','initialization_seed':0,'role':'primary historical control'},{'label':'untrained','initialization_seed':42,'role':'additional new initialization control used in validation'}],'added_final_episodes':40,'promotion_thresholds_changed':False,'checkpoint_selection_changed':False,'created_before_any_final_result':True}
    write_json(HERE/'prefinal_control_amendment.json',amendment,True)
    value={'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'final_traffic_seeds':FINAL_SEEDS,'selection_sha256':base.digest(HERE/'selection.json'),'recipe_selection_sha256':base.digest(HERE/'recipe_selection.json') if recipes else None,'manifest_sha256':base.digest(HERE/'manifest.json'),'prefinal_control_amendment_sha256':base.digest(HERE/'prefinal_control_amendment.json'),'source_sha256':sources,'entries':entries,'selection_eligibility':eligibility,'untrained_controls':amendment['final_untrained_controls'],'promotion':'Original manifest thresholds unchanged. Validation-ineligible diagnostic challengers cannot qualify for promotion from final results. No automatic incumbent replacement.','numerical_precision':'Waiting and queue gates and mean/SD outputs use unrounded per-run values when available. All learned policies and untrained have unrounded values; fixed-time collector output is rounded2decimals and comparisons to fixed-time are descriptive. Raw reported collector values remain in final_metrics.csv. Numerical equality tolerance1e-10 avoids floating-point artifacts.','evaluation_budget_episodes':sum((len(x)+3)*20 for x in entries.values())}
    value['execution']={'max_concurrent_scenario_workers':2,'torch_threads_per_worker':1,'isolation':'One process per scenario with private generated inputs and per-scenario logs. Never concurrent runs within the same scenario.','launcher':'run_final_parallel.py'}
    verify_freeze(value);write_json(path,value,True);print('FROZEN',value['evaluation_budget_episodes'],'final episodes',flush=True)

def verify_freeze(value):
    assert value['final_traffic_seeds']==FINAL_SEEDS
    assert value['selection_sha256']==base.digest(HERE/'selection.json')
    assert value['manifest_sha256']==base.digest(HERE/'manifest.json')
    assert value['prefinal_control_amendment_sha256']==base.digest(HERE/'prefinal_control_amendment.json')
    if value['recipe_selection_sha256']: assert value['recipe_selection_sha256']==base.digest(HERE/'recipe_selection.json')
    for name,expected in value['source_sha256'].items(): assert base.digest(HERE/name)==expected, 'Frozen source differs: '+name
    for entries in value['entries'].values():
        for entry in entries: assert base.digest(base.ROOT/entry['path'])==entry['sha256']

def evaluate(scenario=None):
    frozen=json.loads((HERE/'freeze.json').read_text());verify_freeze(frozen)
    for s in ([scenario] if scenario else base.SCENARIOS):
        agents=base.load_agents(s)
        state=base.rng_state()
        try:
            agents['untrained_historical0']=DQNAgent(observation_size=10,scenario=s,seed=0,trainable=False)
            for entry in frozen['entries'][s]:
                agents[entry['label']]=DQNAgent.load(str(base.ROOT/entry['path']),scenario=s)
                config=torch.load(base.ROOT/entry['path'],map_location='cpu',weights_only=False)['config']
                agents[entry['label']]._diagnostic_gamma=float(config['gamma'])
        finally:base.rng_restore(state)
        state_before={name:(a.learn_steps,len(a.memory),hashlib.sha256(pickle.dumps({k:v.tolist() for k,v in a.policy_net.state_dict().items()})).hexdigest()) for name,a in agents.items()}
        for label in [e['label'] for e in frozen['entries'][s]]+['untrained_historical0','untrained','fixed_time']:
            for seed in FINAL_SEEDS:
                out=HERE/'final'/s/label/(str(seed)+'.json')
                if valid_episode(out,seed,label):continue
                started=time.time()
                if label=='fixed_time':
                    base.isolate_inputs(s,'final/'+s);summary,ts=run_fixed_time_episode(s,seed,1800)
                    data={'summary':dict(summary,policy=label,seed=seed),'timeseries':ts}
                else:data=base.policy_episode(s,seed,label,agents,'final/'+s)
                data['summary'].update(wall_seconds=round(time.time()-started,3),stage='final')
                write_json(out,data,True)
                print('FINAL',s,label,seed,data['summary']['average_waiting_time'],data['summary']['wall_seconds'],flush=True)
        state_after={name:(a.learn_steps,len(a.memory),hashlib.sha256(pickle.dumps({k:v.tolist() for k,v in a.policy_net.state_dict().items()})).hexdigest()) for name,a in agents.items()}
        assert state_before==state_after
        marker=HERE/'final'/s/'inference_state_unchanged.json'
        if not marker.exists():write_json(marker,{'before':state_before,'after':state_after},True)

def final_analysis():
    frozen=json.loads((HERE/'freeze.json').read_text());verify_freeze(frozen)
    records=[];reports={};paired=[];training_variability=[]
    for scenario,entries in frozen['entries'].items():
        groups={}
        entry_by_label={e['label']:e for e in entries}
        for label in list(entry_by_label)+['untrained_historical0','untrained','fixed_time']:
            paths=[HERE/'final'/scenario/label/(str(t)+'.json') for t in FINAL_SEEDS]
            summaries=[json.loads(p.read_text())['summary'] for p in paths]
            assert all(s['simulation_time']==1800 for s in summaries)
            groups[label]=summaries
            for summary in summaries:
                e=entry_by_label.get(label,{})
                records.append(dict({k:v for k,v in summary.items() if isinstance(v,(str,int,float,bool))},stage='final',scenario=scenario,label=label,recipe=e.get('recipe',label),train_seed=e.get('train_seed'),initialization_seed=0 if label=='untrained_historical0' else (42 if label=='untrained' else None),traffic_seed=summary['seed']))
        selected=[e['label'] for e in entries if e.get('role')=='selected']
        controls=[e['label'] for e in entries if e.get('role')=='control']
        def value(summary,metric):
            if metric=='average_waiting_time':return metrics(summary)['waiting']
            if metric=='average_queue_length':return metrics(summary)['queue']
            return summary[metric]
        def aggregate(labels):
            return [{m:float(np.mean([value(groups[label][i],m) for label in labels])) for m in FIELDS} for i in range(20)]
        selected_agg=aggregate(selected);inc=aggregate(['incumbent']);control=aggregate(controls) if controls else None
        def compare(reference,reference_name):
            deltas={m:np.array([c[m]-b[m] for c,b in zip(selected_agg,reference)]) for m in FIELDS}
            for m,v in deltas.items():paired.append({'scenario':scenario,'comparison':'selected_minus_'+reference_name,'metric':m,'paired_mean_delta':float(v.mean()),'paired_traffic_seed_sd':float(v.std(ddof=1)),'traffic_seeds':20})
            wins=int(np.sum(deltas['average_waiting_time']<-1e-10));ties=int(np.sum(np.abs(deltas['average_waiting_time'])<=1e-10))
            checks={'lower_waiting':float(deltas['average_waiting_time'].mean())<-1e-10,'waiting_wins_more_than_half':wins>10,'queue_no_worse':float(deltas['average_queue_length'].mean())<=1e-10,'throughput_no_worse':float(deltas['throughput'].mean())>=-1e-10,'teleports_no_worse':float(deltas['teleports'].mean())<=1e-10}
            return {'checks':checks,'passes':all(checks.values()),'waiting_wins':wins,'waiting_ties':ties,'waiting_losses':20-wins-ties,'mean_deltas':{m:float(v.mean()) for m,v in deltas.items()}}
        vs_inc=compare(inc,'incumbent');vs_fixed=compare(aggregate(['fixed_time']),'fixed_time');vs_untrained=compare(aggregate(['untrained_historical0']),'untrained_historical0');vs_control=compare(control,'control') if control else None
        for label,summaries in groups.items():
            e=entry_by_label.get(label,{})
            for m in FIELDS:
                vals=np.array([value(s,m) for s in summaries]);training_variability.append({'scenario':scenario,'label':label,'role':e.get('role',label),'recipe':e.get('recipe',label),'train_seed':e.get('train_seed'),'initialization_seed':0 if label=='untrained_historical0' else (42 if label=='untrained' else None),'metric':m,'traffic_seed_mean':float(vals.mean()),'traffic_seed_sd':float(vals.std(ddof=1))})
        training_check=None
        if controls:
            train_deltas=[]
            for sl in selected:
                seed=entry_by_label[sl]['train_seed'];cl='control_seed'+str(seed)
                train_deltas.append(float(np.mean([metrics(x)['waiting']-metrics(y)['waiting'] for x,y in zip(groups[sl],groups[cl])])))
            mean=float(np.mean(train_deltas));sd=float(np.std(train_deltas,ddof=1));margin=4.302652729911275*sd/np.sqrt(3)
            training_check={'training_seed_waiting_deltas':train_deltas,'mean':mean,'sample_sd':sd,'ci95_t_df2':[mean-margin,mean+margin],'at_least_two_training_seeds_improve':sum(x<-1e-10 for x in train_deltas)>=2,'ci_upper_below_zero':mean+margin<-1e-10}
        guardrails_passed=vs_inc['passes'] and (not controls or (vs_control['passes'] and training_check['at_least_two_training_seeds_improve'] and training_check['ci_upper_below_zero']))
        passes=frozen['selection_eligibility'][scenario]['validation_eligible'] and guardrails_passed
        reports[scenario]={'selected_recipe':entry_by_label[selected[0]]['recipe'],'selected_training_seeds':[entry_by_label[x]['train_seed'] for x in selected],'incumbent_comparison':vs_inc,'fixed_time_comparison':vs_fixed,'untrained_comparison':vs_untrained,'untrained_control_definition':'Primary untrained initializationseed0 historical recipe; extra initializationseed42 reported separately','control_comparison':vs_control,'training_uncertainty':training_check,'validation_eligibility':frozen['selection_eligibility'][scenario],'final_guardrails_passed':bool(guardrails_passed),'promotion_criteria_passed':bool(passes),'decision':'Eligible for human consideration; incumbent preserved' if passes else 'RETAIN INCUMBENT: improvement not demonstrated under declared criteria','means':{role:{m:float(np.mean([x[m] for x in aggregate(labels)])) for m in FIELDS} for role,labels in [('selected',selected),('incumbent',['incumbent']),('fixed_time',['fixed_time']),('untrained',['untrained']),('untrained_historical0',['untrained_historical0'])]+([('control',controls)] if controls else [])}}
    save_csv(HERE/'final_metrics.csv',records);save_csv(HERE/'paired_final_metrics.csv',paired);save_csv(HERE/'training_traffic_variability.csv',training_variability)
    write_json(HERE/'final_decision.json',reports,True)
    print(json.dumps(reports,indent=2))

if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--evaluate',action='store_true');parser.add_argument('--analyze',action='store_true');parser.add_argument('--scenario',choices=base.SCENARIOS)
    args=parser.parse_args()
    if args.freeze:freeze()
    if args.evaluate:evaluate(args.scenario)
    if args.analyze:final_analysis()
