"""Ice & Fire analysis (after the "A Song of Ice and Fire" notebook) for Kaggriculture.

Ice  = a command that EVERY game of a team submits at a step (scripted part).
Fire = a command only some games submit (the agent reacted to its game).

usage
  # 1) analyse a top team's replays (seat = the team, found by name, else the winner)
  python icefire.py "replays/*.json" --team Majkel1337

  # 2) same numbers for your own agent (plays N local games first) and compare
  python icefire.py "replays/*.json" --team Majkel1337 --mine main.py --games 12

  # 3) export the top team's scripted opening and embed it into a copy of your agent
  python icefire.py "replays/*.json" --team Majkel1337 --export-opening opening.json --min-share 0.6
  python icefire.py --embed opening.json main.py main_opening.py
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

TURNS_PER_DAY = 24
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


# ----------------------------------------------------------------------------
# loading games: list of dicts {"actions": [action per step], "states": [farm per step], ...}
# ----------------------------------------------------------------------------
def names_of(rep):
    info = rep.get("info") or {}
    return info.get("TeamNames") or info.get("team_names") or ["P0", "P1"]


def pick_seat(rep, team):
    names = names_of(rep)
    if team:
        for i, n in enumerate(names):
            if n and team.lower() in str(n).lower():
                return i
        return None
    fin = rep["steps"][-1]
    r0, r1 = fin[0].get("reward") or 0, fin[1].get("reward") or 0
    return 0 if r0 >= r1 else 1


def game_from_replay(rep, seat):
    steps = rep["steps"]
    acts = []
    farms = []
    for t in range(len(steps) - 1):
        a = steps[t + 1][seat].get("action")
        acts.append(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})
        farms.append(steps[t][0]["observation"]["farms"][seat])
    fin = steps[-1]
    won = (fin[seat].get("reward") or 0) > (fin[1 - seat].get("reward") or 0)
    return {"actions": acts, "farms": farms, "won": won,
            "shops": steps[-1][0]["observation"]["town"]["unlocked_shops"]}


def load_games(pattern, team, winners_only):
    games = []
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        seat = pick_seat(rep, team)
        if seat is None:
            continue
        g = game_from_replay(rep, seat)
        if winners_only and not g["won"]:
            continue
        g["path"] = p
        games.append(g)
    return games


def play_my_games(agent_path, n, opponent="starter", seed0=500):
    from kaggle_environments import make
    import importlib.util
    spec = importlib.util.spec_from_file_location("mine_mod", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    games = []
    for i in range(n):
        env = make("kaggriculture", configuration={"seed": seed0 + i}, debug=True)
        env.run([mod.agent, opponent])
        rep = env.toJSON()
        g = game_from_replay(rep, 0)
        g["path"] = f"local seed {seed0 + i}"
        games.append(g)
    return games


# ----------------------------------------------------------------------------
# command units (article section 1) and bundles (section 2)
# ----------------------------------------------------------------------------
def op_of(cmd):
    if not cmd:
        return None
    c = cmd[0]
    if c in MOVES:
        return "MOVE"
    if len(cmd) > 1 and c not in ("DROP",):
        return f"{c}:{cmd[1]}"
    return c


def command_set(a):
    s = set()
    for u in [a.get("farmer")] + list(a.get("hands") or []):
        o = op_of(u)
        if o:
            s.add(o)
    for m in a.get("market") or []:
        o = op_of(m)
        if o:
            s.add(o)
    return s


def bundle(a):
    hands = sorted(op_of(h) or "PASS" for h in (a.get("hands") or []))
    mk = sorted({op_of(m) for m in (a.get("market") or []) if op_of(m)})
    return (op_of(a.get("farmer")) or "PASS", len(hands), tuple(hands), tuple(mk))


def ice_fire(games):
    N = len(games)
    T = min(len(g["actions"]) for g in games)
    per_day_agree = defaultdict(list)
    ice_total, units_total = 0, 0
    ice_by_cat = Counter()
    divergence = []
    for t in range(T):
        cnt = Counter()
        for g in games:
            cnt.update(command_set(g["actions"][t]))
        for cmd, n in cnt.items():
            a = (n - 1) / (N - 1) if N > 1 else 1.0
            per_day_agree[t // TURNS_PER_DAY].append(a)
            units_total += 1
            if n == N:
                ice_total += 1
                ice_by_cat[cmd.split(":")[0]] += 1
        b = Counter(bundle(g["actions"][t]) for g in games)
        divergence.append(1 - max(b.values()) / N)
    return {
        "N": N, "units": units_total, "ice": ice_total, "ice_by_cat": ice_by_cat,
        "day_agree": {d: sum(v) / len(v) for d, v in per_day_agree.items()},
        "divergence": divergence,
    }


def land_steps(games):
    out = []
    for g in games:
        k = 0
        for t, a in enumerate(g["actions"]):
            for m in a.get("market") or []:
                if m and m[0] == "BUY_LAND":
                    # confirm it succeeded: quadrant count rose on the next state
                    if t + 1 < len(g["farms"]) and len(g["farms"][t + 1]["unlocked_quadrants"]) > len(g["farms"][t]["unlocked_quadrants"]):
                        out.append((k, t))
                        k += 1
    return out


def money_curve(games):
    days = defaultdict(list)
    for g in games:
        for d in range(30):
            i = d * TURNS_PER_DAY
            if i < len(g["farms"]):
                days[d].append(g["farms"][i]["money"])
    return {d: sorted(v)[len(v) // 2] for d, v in days.items()}


def mix_at(games, day):
    c = Counter()
    i = day * TURNS_PER_DAY
    for g in games:
        if i >= len(g["farms"]):
            continue
        for row in g["farms"][i]["tiles"]:
            for t in row:
                if isinstance(t, dict):
                    c[t.get("crop") or t.get("animal") or t.get("kind")] += 1
    n = max(1, len(games))
    return {k: round(v / n, 1) for k, v in c.most_common(7)}


def summarize(label, games):
    r = ice_fire(games)
    lands = land_steps(games)
    by_k = defaultdict(list)
    for k, t in lands:
        by_k[k].append(t)
    print(f"\n##### {label}: {r['N']} games")
    print(f"ice {r['ice']} of {r['units']} command-units ({100*r['ice']/max(1,r['units']):.1f}%); "
          f"ice by class: {dict(r['ice_by_cat'].most_common(6))}")
    da = r["day_agree"]
    print("mean agreement by day: " + " ".join(f"d{d}:{100*da[d]:.0f}%" for d in sorted(da) if d < 30))
    dv = r["divergence"]
    blocks = [(0, 72), (72, 144), (144, 216), (216, 360), (360, 720)]
    print("bundle divergence: " + "  ".join(f"steps {a}-{b}: {sum(dv[a:b])/max(1,len(dv[a:b])):.3f}" for a, b in blocks if a < len(dv)))
    first_split = next((t for t, x in enumerate(dv) if x > 0.2), None)
    print(f"first step where >20% of games leave the majority bundle: {first_split}"
          + (f" (day {first_split // 24}, hour {first_split % 24})" if first_split is not None else ""))
    for k in sorted(by_k):
        c = Counter(by_k[k])
        top = ", ".join(f"step {t} (d{t//24}h{t%24}) x{n}" for t, n in c.most_common(3))
        print(f"land #{k+1}: {len(by_k[k])}/{len(games)} games; most common: {top}")
    mc = money_curve(games)
    print("median money by day: " + " ".join(f"d{d}:{mc[d]:.0f}" for d in (1, 3, 5, 7, 10, 13, 16, 20, 25, 29) if d in mc))
    for d in (1, 5, 10, 20):
        print(f"avg tiles on day {d}: {mix_at(games, d)}")
    return r


# ----------------------------------------------------------------------------
# opening export / embed
# ----------------------------------------------------------------------------
def farm_sig(farm):
    rows = []
    for row in farm["tiles"]:
        rows.append("".join(
            "." if t is None else "#" if t == "LOCKED" else
            (t.get("crop") or t.get("animal") or t.get("kind") or "?")[0] for t in row))
    return "|".join(rows) + f"/h{len(farm['hands'])}/q{len(farm['unlocked_quadrants'])}"


def export_opening(games, min_share, max_steps):
    """Majority action per step, following only games still on the script."""
    live = list(range(len(games)))
    N = len(games)
    script = []
    for t in range(max_steps):
        if not live:
            break
        cnt = Counter(json.dumps(games[i]["actions"][t], sort_keys=True) for i in live)
        best, n = cnt.most_common(1)[0]
        new_live = [i for i in live if json.dumps(games[i]["actions"][t], sort_keys=True) == best]
        if len(new_live) < min_share * N:
            break
        sig = farm_sig(games[new_live[0]]["farms"][t])
        script.append({"action": json.loads(best), "sig": sig})
        live = new_live
    print(f"opening: {len(script)} steps (to day {len(script)//24}, hour {len(script)%24}); "
          f"{len(live)}/{N} games follow it all the way")
    return script


EMBED_MARK = "# ==== EMBEDDED OPENING (generated by icefire.py) ===="


def embed(opening_path, agent_in, agent_out):
    with open(opening_path) as f:
        script = json.load(f)
    src = open(agent_in, encoding="utf-8").read()
    if EMBED_MARK in src:
        src = src[:src.index(EMBED_MARK)].rstrip() + "\n"
    if "def agent(obs, config=None):" not in src:
        sys.exit("agent file must define `def agent(obs, config=None):`")
    src = src.replace("def agent(obs, config=None):", "def _base_agent(obs, config=None):")
    src += f'''

{EMBED_MARK}
OPENING = {json.dumps(script)}


def _farm_sig(farm):
    rows = []
    for row in farm["tiles"]:
        rows.append("".join(
            "." if t is None else "#" if t == "LOCKED" else
            (t.get("crop") or t.get("animal") or t.get("kind") or "?")[0] for t in row))
    return "|".join(rows) + "/h%d/q%d" % (len(farm["hands"]), len(farm["unlocked_quadrants"]))


_OPEN_STATE = {{}}


# Kaggle calls the LAST callable in main.py: keep `agent` last.
def agent(obs, config=None):
    p = obs["player"]
    step = obs["day"] * 24 + obs["hour"]
    st = _OPEN_STATE.setdefault(p, {{"on": True, "last": -1}})
    if step < st["last"]:
        st.update(on=True)            # new game in the same process
    st["last"] = step
    if st["on"] and step < len(OPENING):
        if _farm_sig(obs["farms"][p]) == OPENING[step]["sig"]:
            _base_agent(obs, config)  # keep the base agent's memory warm; ignore its action
            return OPENING[step]["action"]
        st["on"] = False              # the board left the script: hand over for good
    return _base_agent(obs, config)
'''
    open(agent_out, "w", encoding="utf-8").write(src)
    print(f"wrote {agent_out} with a {len(script)}-step opening")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="?")
    ap.add_argument("--team", default=None, help="team name substring (else: the winner of each replay)")
    ap.add_argument("--all", action="store_true", help="include lost games (default: winners only, like the article)")
    ap.add_argument("--mine", default=None, help="your agent .py to compare")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--export-opening", default=None)
    ap.add_argument("--min-share", type=float, default=0.6)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--embed", nargs=3, metavar=("OPENING_JSON", "AGENT_IN", "AGENT_OUT"))
    args = ap.parse_args()

    if args.embed:
        embed(*args.embed)
        return
    top = []
    if args.replays:
        top = load_games(args.replays, args.team, winners_only=not args.all)
        if not top:
            sys.exit("no games matched (check the glob and --team)")
        summarize(f"TOP ({args.team or 'replay winners'})", top)
    if args.mine:
        mine = play_my_games(args.mine, args.games, args.opponent)
        summarize(f"MINE ({args.mine} vs {args.opponent})", mine)
    if args.export_opening:
        if not top:
            sys.exit("--export-opening needs replays")
        script = export_opening(top, args.min_share, args.max_steps)
        with open(args.export_opening, "w") as f:
            json.dump(script, f)
        print(f"saved {args.export_opening}")


if __name__ == "__main__":
    main()
