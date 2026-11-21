"""Bounded matched training and validation in separate processes.

Read training_predeclaration.json, then run --run. Existing artifacts are
immutable. Every recipe is freshly trained for all three declared seeds.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import torch
import run_controlled as base
from analyze_controlled import gate, metrics
from agents.dqn_config import load_dqn_config
from agents.dqn_agent import DQNAgent
import numpy as np
from artifact_io import write_json, valid_or_preserve, valid_episode

HERE=base.HERE
RECIPES={'control':0.001,'lower_lr':0.0003}
TRAIN_SEEDS=[42,43,44]
TRAIN_CHECKPOINTS=['best','ep100','ep200','final']

def run_directory(scenario,recipe,seed): return HERE/'training_runs'/scenario/recipe/('seed'+str(seed))
def ckpt(directory,name): return directory/'models'/({'ep100':'checkpoint_ep100','ep200':'checkpoint_ep200'}.get(name,name)+'.pt')
def config(seed,recipe):
    return load_dqn_config(episodes=300,episode_duration=1800,seed=seed,learning_rate=RECIPES[recipe],gamma=0.9,epsilon_decay_steps=27000,checkpoint_every=100)

def predeclare():
    selection=json.loads((HERE/'selection.json').read_text())
    scenarios=[s for s in base.SCENARIOS if selection[s]['new_training_needed']]
    payload={
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scenarios':scenarios,
        'changes':'One factor only: learning_rate0.001 vs0.0003; gamma0.9 and standard DQN unchanged.',
        'training_seeds':TRAIN_SEEDS,'recipes':{r:[config(s,r).as_dict() for s in TRAIN_SEEDS] for r in RECIPES},
        'torch_threads':1,'max_concurrent_worker_processes':3,'runs_max':len(scenarios)*6,'episodes_max':len(scenarios)*1800,
        'validation_budget_episodes_max':len(scenarios)*240,
        'final_budget_episodes_max':len(scenarios)*200+(len(base.SCENARIOS)-len(scenarios))*100,
        'checkpoint_selection':'For each training seed and recipe, greedy evaluate best,ep100,ep200,final on2001-2010 in separate child process. Rank lowest mean unrounded waiting, then queue, then highest throughput, then order best,ep100,ep200,final. Identical selection procedure for control and lower_lr. Training best is only a candidate.',
        'recipe_selection':'Average the three validation-selected replicate metrics for each traffic seed. Select lower_lr only if it passes the original validation gate against BOTH incumbent_best and matched validation-selected control and waiting improves vs control in>=2/3 training seeds. Otherwise choose control as diagnostic challenger and retain incumbent pending final. No additional recipes or hyperparameters.',
        'promotion_criteria':'Exactly original manifest; selected control is not a novel improvement vs itself and cannot satisfy the new recipe vs control superiority criterion. No automatic replacement.',
        'runtime_estimate':'Serial estimate6-9hours heavy6runs plus240validation; up to3 isolated seed worker processes per recipe, each one Torch thread, may reduce wall time. No runtime guarantee.',
        'recovery_source':{'path':'experiments/dqn_300_20261117/resumable_training.py','sha256':base.digest(base.ROOT/'experiments/dqn_300_20261117/resumable_training.py')},
        'source_sha256':base.digest(__file__)
    }
    write_json(HERE/'training_predeclaration.json',payload,True)
    print(json.dumps(payload,indent=2))

def trainer():
    source=base.ROOT/'experiments/dqn_300_20261117/resumable_training.py'
    spec=importlib.util.spec_from_file_location('preserved_resumable_training',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def validate_run(scenario,recipe,seed):
    declaration=json.loads((HERE/'training_predeclaration.json').read_text())
    assert scenario in declaration['scenarios'] and seed in TRAIN_SEEDS and recipe in RECIPES
    directory=run_directory(scenario,recipe,seed)
    agents=base.load_agents(scenario)
    state=base.rng_state()
    try:
        for name in TRAIN_CHECKPOINTS:
            key=recipe+'_seed'+str(seed)+'_'+name
            agents[key]=DQNAgent.load(str(ckpt(directory,name)),scenario=scenario)
            agents[key]._diagnostic_gamma=config(seed,recipe).gamma
    finally: base.rng_restore(state)
    for name in TRAIN_CHECKPOINTS:
        key=recipe+'_seed'+str(seed)+'_'+name
        for traffic_seed in range(2001,2011):
            out=directory/'validation'/name/(str(traffic_seed)+'.json')
            if valid_episode(out,traffic_seed,key): continue
            started=time.time();data=base.policy_episode(scenario,traffic_seed,key,agents,'training_validation/'+scenario+'/'+recipe+'/seed'+str(seed))
            data['summary'].update(train_seed=seed,recipe=recipe,checkpoint=name,stage='validation',wall_seconds=round(time.time()-started,3))
            write_json(out,data,True)
            print('TRAIN_VALIDATION',scenario,recipe,seed,name,traffic_seed,data['summary']['average_waiting_time'],flush=True)
    groups={name:[json.loads((directory/'validation'/name/(str(t)+'.json')).read_text())['summary'] for t in range(2001,2011)] for name in TRAIN_CHECKPOINTS}
    rank=lambda n:(np.mean([metrics(s)['waiting'] for s in groups[n]]),np.mean([metrics(s)['queue'] for s in groups[n]]),-np.mean([metrics(s)['throughput'] for s in groups[n]]),TRAIN_CHECKPOINTS.index(n))
    chosen=min(TRAIN_CHECKPOINTS,key=rank)
    target=directory/'selected.json'
    if not target.exists(): write_json(target,{'scenario':scenario,'recipe':recipe,'train_seed':seed,'checkpoint':chosen,'path':ckpt(directory,chosen).relative_to(base.ROOT).as_posix(),'sha256':base.digest(ckpt(directory,chosen)),'validation_means':{n:{m:float(np.mean([metrics(s)[m] for s in groups[n]])) for m in ['waiting','queue','throughput','teleports']} for n in TRAIN_CHECKPOINTS}},True)

def train_run(scenario,recipe,seed):
    directory=run_directory(scenario,recipe,seed);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'selected.json').exists(): return
    base.isolate_inputs(scenario,'training/'+scenario+'/'+recipe+'/seed'+str(seed))
    started=time.time()
    print('TRAIN_START',scenario,recipe,seed,flush=True)
    summary=trainer().train(scenario,config(seed,recipe),model_dir=str(directory/'models'),results_dir=str(directory/'history'),quiet=False,progress_every=10,save_plots=False)
    if not (directory/'completed.json').exists(): write_json(directory/'completed.json',summary,True)
    print('TRAIN_COMPLETE',scenario,recipe,seed,round(time.time()-started,1),flush=True)
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--validate-run','--scenario',scenario,'--recipe',recipe,'--seed',str(seed)],check=True)

def launch():
    d=json.loads((HERE/'training_predeclaration.json').read_text())
    assert d['source_sha256']==base.digest(__file__),'Training source changed after declaration'
    assert d['recovery_source']['sha256']==base.digest(base.ROOT/d['recovery_source']['path'])
    for scenario in d['scenarios']:
        for recipe in RECIPES:
            children=[]
            for seed in TRAIN_SEEDS:
                directory=run_directory(scenario,recipe,seed);directory.mkdir(parents=True,exist_ok=True)
                if (directory/'selected.json').exists(): continue
                # Process boundary keeps validation model creation, RNG and SUMO
                # environment objects wholly separate from training recovery.
                log=(directory/'console.log').open('a',encoding='utf-8')
                child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--train-run','--scenario',scenario,'--recipe',recipe,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
                children.append((seed,child,log));print('WORKER_LAUNCHED',scenario,recipe,seed,child.pid,flush=True)
            failures=[]
            for seed,child,log in children:
                code=child.wait();log.close()
                if code: failures.append((seed,code))
            if failures: raise RuntimeError(f'Worker failures preserve snapshots for recovery: {failures}')
    select_recipes()

def averaged_summaries(scenario,recipe):
    replicate_groups=[]
    for seed in TRAIN_SEEDS:
        directory=run_directory(scenario,recipe,seed);selected=json.loads((directory/'selected.json').read_text())
        replicate_groups.append([json.loads((directory/'validation'/selected['checkpoint']/(str(t)+'.json')).read_text())['summary'] for t in range(2001,2011)])
    keys=['waiting_unrounded','queue_unrounded','throughput','teleports']
    return [dict(seed=t,**{k:float(np.mean([g[i][k] for g in replicate_groups])) for k in keys}) for i,t in enumerate(range(2001,2011))],replicate_groups

def select_recipes():
    declaration=json.loads((HERE/'training_predeclaration.json').read_text()); result={}
    for scenario in declaration['scenarios']:
        incumbent=[json.loads((HERE/'validation'/scenario/'incumbent_best'/(str(t)+'.json')).read_text())['summary'] for t in range(2001,2011)]
        control,c_reps=averaged_summaries(scenario,'control');lower,l_reps=averaged_summaries(scenario,'lower_lr')
        vs_control=gate(lower,control);vs_incumbent=gate(lower,incumbent)
        deltas=[float(np.mean([metrics(l)['waiting']-metrics(c)['waiting'] for l,c in zip(lg,cg)])) for lg,cg in zip(l_reps,c_reps)]
        selected='lower_lr' if vs_control['eligible'] and vs_incumbent['eligible'] and sum(x<-1e-10 for x in deltas)>=2 else 'control'
        result[scenario]={'selected_recipe':selected,'lower_lr_vs_control_gate':vs_control,'lower_lr_vs_incumbent_gate':vs_incumbent,'training_seed_waiting_deltas':deltas,'training_seeds_improved':sum(x<-1e-10 for x in deltas),'replicates':{r:[json.loads((run_directory(scenario,r,s)/'selected.json').read_text()) for s in TRAIN_SEEDS] for r in RECIPES}}
    write_json(HERE/'recipe_selection.json',result,True)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--predeclare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--train-run',action='store_true');p.add_argument('--validate-run',action='store_true');p.add_argument('--scenario',choices=base.SCENARIOS);p.add_argument('--recipe',choices=list(RECIPES));p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.predeclare: predeclare()
    if args.validate_run: validate_run(args.scenario,args.recipe,args.seed)
    if args.train_run: train_run(args.scenario,args.recipe,args.seed)
    if args.run:
        import msvcrt
        with (HERE/'training.lock').open('a+b') as lock:
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            try: launch()
            finally: lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
