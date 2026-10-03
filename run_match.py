"""Local tournament harness.

usage:
  python run_match.py main.py starter -n 10
  python run_match.py main.py main_v0.py -n 20 -j 2
Agents are .py paths (must define `agent`) or built-in names: pass, random, starter.
Each seed is played twice with sides swapped to remove seat bias.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool


def load_agent(spec, tag):
    if spec.endswith(".py"):
        path = os.path.abspath(spec)
        name = f"agent_{tag}_{abs(hash(path))}"
        mod_spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return mod.agent
    return spec


def play(args):
    a, b, seed, swap, save = args
    from kaggle_environments import make
    A = load_agent(a, "A")
    B = load_agent(b, "B")
    agents = [B, A] if swap else [A, B]
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    t = time.time()
    env.run(agents)
    final = env.steps[-1]
    r = [s.reward for s in final]
    st = [s.status for s in final]
    if swap:
        r, st = r[::-1], st[::-1]
    if save:
        with open(f"replays/{seed}_{int(swap)}.json", "w") as f:
            json.dump(env.toJSON(), f)
    return seed, swap, r, st, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("-j", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--noswap", action="store_true")
    args = ap.parse_args()
    os.makedirs("replays", exist_ok=True)
    jobs = []
    for i in range(args.n):
        jobs.append((args.a, args.b, args.seed0 + i, False, args.save))
        if not args.noswap:
            jobs.append((args.a, args.b, args.seed0 + i, True, args.save))
    wins = losses = ties = 0
    ra, rb = [], []
    with Pool(args.j) as pool:
        for seed, swap, r, st, dt in pool.imap_unordered(play, jobs):
            ra.append(r[0] or 0)
            rb.append(r[1] or 0)
            if (r[0] or 0) > (r[1] or 0): wins += 1
            elif (r[0] or 0) < (r[1] or 0): losses += 1
            else: ties += 1
            print(f"seed {seed} swap={int(swap)}  A={r[0]:>9}  B={r[1]:>9}  {st}  {dt:.1f}s", flush=True)
    n = len(ra)
    print(f"\nA={args.a} vs B={args.b}:  W/L/T = {wins}/{losses}/{ties}   "
          f"mean A={sum(ra)/n:.0f}  mean B={sum(rb)/n:.0f}")


if __name__ == "__main__":
    main()
