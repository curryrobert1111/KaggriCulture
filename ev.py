"""Fast evaluation.
  python ev.py A.py [--h2h B.py -n N] [--ghost clean|decem|all] [-j 2]
Ghost: our agent vs recorded top agents (tape). H2H: seeds 1..N both sides.
"""
import argparse, importlib.util, os, sys, time
from multiprocessing import Pool
import sim

CLEAN = [("115321748", 1), ("115324670", 0), ("115343521", 1), ("115350314", 0), ("115350323", 0)]
DECEM = [("115321748", 0), ("115324670", 1), ("115343521", 0), ("115350314", 1), ("115350323", 1)]
_MODS = {}


def load_agent(path):
    if path not in _MODS:
        spec = importlib.util.spec_from_file_location("ag_%d" % len(_MODS), os.path.abspath(path))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _MODS[path] = m
    m = _MODS[path]
    m._STATE.clear()
    return m.agent


def job(a):
    kind = a[0]
    if kind == "ghost":
        _, path, rid, seat, ms = a
        r, _ = sim.run_ghost(sim.load(f"replays/{rid}.json"), load_agent(path), seat, my_seed=ms)
        return (kind, rid, seat, r[1 - seat], r[seat])
    _, pa, pb, seed, swap = a
        # separate module objects for A and B even if same file
    A = load_agent(pa); B = load_agent(pb) if pb != pa else None
    if B is None:
        spec = importlib.util.spec_from_file_location("agB", os.path.abspath(pb)); mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb); B = mb.agent
    r, _ = sim.run([B, A] if swap else [A, B], seed)
    if swap: r = r[::-1]
    return (kind, seed, swap, r[0], r[1])


def evaluate(path, ghost="all", h2h=None, n=0, j=2, seed0=1, quiet=True, wseeds=1):
    jobs = []
    for ms in range(wseeds):
        if ghost in ("clean", "all"): jobs += [("ghost", path, rid, s, ms) for rid, s in CLEAN]
        if ghost in ("decem", "all"): jobs += [("ghost", path, rid, s, ms) for rid, s in DECEM]
    if h2h:
        jobs += [("h2h", path, h2h, seed0 + i, sw) for i in range(n) for sw in (False, True)]
    out = {"clean": [], "decem": [], "h2h": []}
    with Pool(j) as pool:
        for res in pool.imap_unordered(job, jobs):
            if res[0] == "ghost":
                key = "clean" if (res[1], res[2]) in CLEAN else "decem"
                out[key].append((res[3], res[4]))
            else:
                out["h2h"].append((res[3], res[4]))
            if not quiet: print(res, flush=True)
    summ = {}
    for k, v in out.items():
        if v:
            g = [a - b for a, b in v]; m = sum(g) / len(g)
            se = (sum((x - m) ** 2 for x in g) / max(1, len(g) - 1)) ** 0.5 / len(g) ** 0.5
            summ[k] = dict(n=len(v), win=sum(a > b for a, b in v), you=sum(a for a, _ in v) / len(v),
                           opp=sum(b for _, b in v) / len(v), gap=m, se=se)
    return summ


def fmt(s):
    return "  ".join(f"{k}: {v['win']}/{v['n']} you {v['you']:.0f} opp {v['opp']:.0f} gap {v['gap']:+.0f}±{v['se']:.0f}" for k, v in s.items())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("--h2h"); ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--ghost", default="all"); ap.add_argument("-j", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=1); ap.add_argument("-v", action="store_true")
    ap.add_argument("-w", type=int, default=1)
    a = ap.parse_args()
    t = time.time()
    s = evaluate(a.a, a.ghost, a.h2h, a.n, a.j, a.seed0, not a.v, a.w)
    print(a.a, fmt(s), f"({time.time()-t:.0f}s)")
