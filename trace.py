"""Play one game and print a per-day summary for player 0.

usage: python trace.py main.py main.py --seed 5
"""
import argparse
from collections import Counter
from run_match import load_agent
from kaggle_environments import make

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("-p", type=int, default=0)
args = ap.parse_args()

env = make("kaggriculture", configuration={"seed": args.seed}, debug=True)
env.run([load_agent(args.a, "A"), load_agent(args.b, "B")])
P = args.p
sold = Counter(); rev = Counter()
prev_money = None
for step, states in enumerate(env.steps):
    obs = states[0].observation
    act = states[P].action or {}
    if step > 0:
        for o in (act.get("market") or []):
            pass
    farm = obs["farms"][P]
    if obs["hour"] == 0 or step == len(env.steps) - 1:
        comp = Counter()
        for row in farm["tiles"]:
            for t in row:
                if t is None: comp["empty"] += 1
                elif t == "LOCKED": pass
                elif t.get("kind") == "PLANT": comp[t["crop"][:4]] += 1
                elif t.get("animal"): comp[t["animal"]] += 1
                else: comp[t["kind"]] += 1
        pr = obs["market"]["prices"]
        print(f"d{obs['day']:>2} ${farm['money']:>8.0f} opp ${obs['farms'][1-P]['money']:>8.0f} "
              f"q={len(farm['unlocked_quadrants'])} "
              f"{dict(comp)}  shops={len(obs['town']['unlocked_shops'])} "
              f"P: M{pr['MELON']} E{pr['EGG']} Mi{pr['MILK']} W{pr['WOOL']} S{pr['STRAWBERRY']} F{pr['FERTILIZER']} Wh{pr['WHEAT']} C{pr['CARROT']} T{pr['TOMATO']}")
    # count hands
    if obs["hour"] == 5:
        print(f"     hands={len(farm['hands'])} shed={ {k:v for k,v in states[P].observation['private']['shed'].items() if v} }")
print("final", [s.reward for s in env.steps[-1]])
print("shops", env.steps[-1][0].observation["town"]["unlocked_shops"])
