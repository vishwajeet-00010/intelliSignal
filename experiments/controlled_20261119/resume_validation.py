"""Idempotent continuation of the predeclared validation partition only.

Preserves invalid/incomplete JSON evidence and uses the exact frozen episode
implementation. Never opens final-test results or retrains any model.
"""
import hashlib,json,pickle,time
import torch
import run_controlled as base
from artifact_io import valid_episode,write_json

def fingerprint(agents):
    return {p:{'learn_steps':a.learn_steps,'replay_size':len(a.memory),'policy_sha256':hashlib.sha256(pickle.dumps({k:v.tolist() for k,v in a.policy_net.state_dict().items()})).hexdigest()} for p,a in agents.items()}

def resume():
    manifest=json.loads((base.HERE/'manifest.json').read_text())
    frozen=json.loads((base.HERE/'validation_source_freeze.json').read_text())
    assert frozen['sha256']==base.digest(base.HERE/'run_controlled.py')
    for entry in manifest['checkpoints']:assert base.digest(base.ROOT/entry['path'])==entry['sha256']
    for scenario in base.SCENARIOS:
        agents=base.load_agents(scenario);before=fingerprint(agents)
        for policy in base.SHORTLIST+['untrained','fixed_time']:
            for seed in manifest['validation_seeds']:
                out=base.HERE/'validation'/scenario/policy/(str(seed)+'.json')
                if valid_episode(out,seed,policy):continue
                started=time.time()
                if policy=='fixed_time':
                    base.isolate_inputs(scenario,'validation');summary,ts=base.run_fixed_time_episode(scenario,seed,1800)
                    data={'summary':dict(summary,policy=policy,seed=seed),'timeseries':ts}
                else:data=base.policy_episode(scenario,seed,policy,agents,'validation')
                data['summary']['wall_seconds']=round(time.time()-started,3)
                write_json(out,data,True)
                print('RESUMED_VALIDATION',scenario,policy,seed,data['summary']['average_waiting_time'],flush=True)
        after=fingerprint(agents);assert before==after
        marker=base.HERE/'validation'/scenario/'resumed_inference_state_unchanged.json'
        if not marker.exists():write_json(marker,{'before':before,'after':after},True)

if __name__=='__main__':torch.set_num_threads(1);resume()
