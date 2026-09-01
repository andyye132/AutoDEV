"""Client side of the remote fitness queue: copy a tester program to the cluster inbox, wait for scripts/eval_server.py to
write the result, return the evaluate_hooks-style dict. Used when AUTODEV_REMOTE_EVAL=1 (see autodev.search_eval.evaluate_hooks)."""
import os, subprocess, time, uuid, json

HOST = os.environ.get("AUTODEV_REMOTE_HOST", "hyak")
QUEUE = os.environ.get("AUTODEV_REMOTE_QUEUE", "/gscratch/raivn/andy132/AutoDev/runs/hyak/evalq")
SSH_OPTS = ["-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15"]


def _ssh(cmd, timeout=90):
    return subprocess.run(["ssh", *SSH_OPTS, HOST, cmd], capture_output=True, text=True, timeout=timeout)


def _retry(fn, tries=4, wait=15, what="ssh"):
    last = None
    for _ in range(tries):
        try:
            r = fn()
            if r.returncode == 0: return r
            last = r.stderr
        except subprocess.TimeoutExpired as e:
            last = str(e)
        time.sleep(wait)
    raise RuntimeError(f"{what} failed after {tries} tries: {str(last)[-500:]}")


def remote_evaluate(program_path, timeout_s=None, poll_s=None, log=None):
    timeout_s = timeout_s or float(os.environ.get("AUTODEV_REMOTE_TIMEOUT", "5400"))
    poll_s = poll_s or float(os.environ.get("AUTODEV_REMOTE_POLL", "20"))
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    tmp, final, out = f"{QUEUE}/inbox/{name}.py.tmp", f"{QUEUE}/inbox/{name}.py", f"{QUEUE}/outbox/{name}.json"
    _retry(lambda: subprocess.run(["scp", *SSH_OPTS, program_path, f"{HOST}:{tmp}"], capture_output=True, text=True, timeout=120), what="scp")
    _retry(lambda: _ssh(f"mv {tmp} {final}"), what="ssh mv")
    if log: log(f"remote eval {name}: submitted")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        try:
            r = _ssh(f"cat {out} 2>/dev/null")
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0 and r.stdout.strip():
            res = json.loads(r.stdout)
            if log: log(f"remote eval {name}: score {res.get('score')} after {time.time() - t0:.0f}s")
            if res.get("error"):
                raise RuntimeError("remote evaluation failed:\n" + res["error"][-2000:])
            res["remote_name"] = name
            return res
    raise TimeoutError(f"remote evaluation {name} not finished after {timeout_s:.0f}s")
