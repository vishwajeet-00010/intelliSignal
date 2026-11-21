"""Durable append-only JSON writes with Windows lock retries."""
import json, os, time
from pathlib import Path

def replace_retry(source,destination,attempts=20):
    for attempt in range(attempts):
        try: os.replace(source,destination);return
        except PermissionError:
            if attempt==attempts-1: raise
            time.sleep(min(.25*(attempt+1),2))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def valid_or_preserve(path,required_keys=()):
    """Preserve an invalid partial file before allowing an isolated retry."""
    path=Path(path)
    if not path.exists(): return False
    try:
        value=read_json(path)
        if not isinstance(value,dict): raise ValueError('Expected JSON mapping')
        if not all(k in value for k in required_keys): raise ValueError('Missing keys')
        return True
    except (ValueError,UnicodeError):
        recovery=path.parent/'recovery';recovery.mkdir(parents=True,exist_ok=True)
        target=recovery/(path.name+'.invalid.'+str(time.time_ns()))
        replace_retry(path,target)
        return False

def write_json(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and exclusive:
        existing=read_json(path)
        if existing!=value: raise FileExistsError('Existing valid artifact differs: '+str(path))
        return
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid())+'.'+str(time.time_ns()))
    with temporary.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.flush();os.fsync(handle.fileno())
    if exclusive and path.exists(): raise FileExistsError('Concurrent writer detected; temporary preserved: '+str(temporary))
    replace_retry(temporary,path)

def valid_episode(path,seed,policy):
    """Skip only a structurally complete episode; retain partial evidence."""
    path=Path(path)
    if not valid_or_preserve(path,('summary','timeseries')):return False
    value=read_json(path);summary=value['summary']
    valid=isinstance(summary,dict) and summary.get('seed')==seed and summary.get('policy')==policy and summary.get('simulation_time')==1800 and isinstance(value['timeseries'],list) and len(value['timeseries'])==1800
    if policy!='fixed_time':valid=valid and isinstance(value.get('decisions'),list) and len(value['decisions'])==180
    if valid:return True
    recovery=path.parent/'recovery';recovery.mkdir(parents=True,exist_ok=True)
    replace_retry(path,recovery/(path.name+'.incomplete.'+str(time.time_ns())))
    return False
