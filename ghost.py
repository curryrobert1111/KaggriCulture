"""Ghost race: play your agent against a top agent's recorded game.

A downloaded replay holds the episode seed and every action of both players,
but the seed alone does NOT reproduce the world: the engine's daily RNG draws
weed spawns for every empty tile of BOTH farms before drawing the next shop,
so a different opponent changes the shop sequence. This script therefore
patches the local engine to force:
  * the recorded shop sequence,
  * the recorded weed spawns on the tape player's farm,
and replays the top agent's actions open-loop ("tape") from its original seat,
while your agent plays the other seat.

usage:
  python ghost.py replay.json main.py                 # auto-picks the winner as the tape
  python ghost.py replay.json main.py --top 0         # choose the tape seat
  python ghost.py replays/*.json main.py --summary     # many replays, one line each + totals
  python ghost.py replay.json --selfcheck             # tape vs tape must reproduce the replay exactly
"""
import argparse
import glob
import json
import random
import sys
from collections import Counter, defaultdict

from kaggle_environments import make
import kaggle_environments.envs.kaggriculture.kaggriculture as K

# Turns per day. This is a GAME RULE, not the seed -- never change it.
# The episode seed is read automatically from the replay file (info.seed).
TURNS_PER_DAY = 24
TPD = TURNS_PER_DAY


# ----------------------------------------------------------------------------
# replay helpers
# ----------------------------------------------------------------------------
def load_replay(path):
    if TPD != 24:
        sys.exit("TPD/TURNS_PER_DAY must be 24 (it is turns per day, not the seed). "
                 "The seed is read from the replay file automatically.")
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    if "steps" not in rep:
        sys.exit(f"{path}: not an episode replay (download one with: kaggle competitions replay <EPISODE_ID>)")
    n = len(rep["steps"])
    if n < 2 * TPD:
        sys.exit(f"{path}: only {n} steps -- the episode ended early (probably an agent error). Pick another episode.")
    if (rep.get("info") or {}).get("seed") is None:
        print(f"warning: {path} has no info.seed; the shops/weeds are still forced from the recording.")
    return rep


def n_days(rep):
    return min(30, len(rep["steps"]) // TPD)


def obs0(rep, i):
    return rep["steps"][i][0]["observation"]


def team_names(rep):
    info = rep.get("info") or {}
    names = info.get("TeamNames") or info.get("team_names")
    return names if names else ["P0", "P1"]


def tape_agent(rep, seat):
    steps = rep["steps"]

    def act(obs, config=None):
        t = obs["step"]
        if t + 1 < len(steps):
            a = steps[t + 1][seat].get("action")
            if isinstance(a, dict):
                return a
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return act


# ----------------------------------------------------------------------------
# faithful world: patch the engine's end-of-day
# ----------------------------------------------------------------------------
_ORIG_END_OF_DAY = K._end_of_day


def install_world(rep, tape_seat, my_seed=0, forced_seats=None):
    """Force recorded shops + recorded weeds on the tape farm(s)."""
    forced_seats = {tape_seat} if forced_seats is None else set(forced_seats)
    n = len(rep["steps"])
    shops_by_day = {}
    for d in range(0, 31):
        i = d * TPD
        if i < n:
            shops_by_day[d] = list(obs0(rep, i)["town"]["unlocked_shops"])
    weeds_by_day = {}
    for seat in forced_seats:
        for d in range(0, 30):
            i = (d + 1) * TPD
            if i < n:
                tiles = obs0(rep, i)["farms"][seat]["tiles"]
                weeds_by_day[(seat, d)] = {(x, y) for y, row in enumerate(tiles) for x, t in enumerate(row)
                                           if isinstance(t, dict) and t.get("kind") == "WEED"}

    def patched_end_of_day(state, env, day):
        o = state[0].observation
        cfg = env.configuration
        board = int(K.get(cfg, "boardSize", 10))
        tpd = max(1, int(K.get(cfg, "turnsPerDay", 24)))
        chance = float(K.get(cfg, "weedSpawnChance", 0.005))
        cap = int(K.get(cfg, "shedCapacity", 100))
        rng = random.Random(my_seed * 1_000_003 + day)
        for pid, farm in enumerate(o.farms):
            private = state[pid].observation.private
            K._daily_refresh_plants(farm, day, tpd)
            K._daily_refresh_animals(farm, day)
            if pid in forced_seats:
                for (x, y) in weeds_by_day.get((pid, day), ()):
                    if farm["tiles"][y][x] is None:
                        farm["tiles"][y][x] = {"kind": "WEED"}
            else:
                K._spawn_weeds(farm, board, chance, rng)
            K._drop_inventories_to_shed(private, cap)
            farm["farmer"] = list(K._default_spawn(board))
            farm["hands"] = []
            farm["hires_today"] = 0
            private["inventories"] = [{}]
        if day + 1 in shops_by_day:
            o.town["unlocked_shops"] = list(shops_by_day[day + 1])

    K._end_of_day = patched_end_of_day


def uninstall_world():
    K._end_of_day = _ORIG_END_OF_DAY


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def tile_mix(farm):
    c = Counter()
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                c["empty"] += 1
            elif t == "LOCKED":
                continue
            elif t.get("kind") == "PLANT":
                c[t["crop"]] += 1
            elif t.get("animal"):
                c[t["animal"]] += 1
            else:
                c[t["kind"]] += 1
    return c


def commands_per_day(steps, seat):
    """Counter per day of class:good commands (moves merged), like the Ice/Fire article."""
    out = defaultdict(Counter)
    for i in range(1, len(steps)):
        a = steps[i][seat].get("action") or {}
        day = (i - 1) // TPD
        units = [a.get("farmer")] + list(a.get("hands") or [])
        for u in units:
            if not u:
                continue
            op = u[0]
            if op in ("NORTH", "SOUTH", "EAST", "WEST"):
                op = "MOVE"
            key = op + (":" + str(u[1]) if len(u) > 1 and op not in ("MOVE",) else "")
            out[day][key] += 1
        for m in a.get("market") or []:
            key = m[0] + (":" + str(m[1]) if len(m) > 1 else "")
            qty = m[2] if len(m) > 2 and isinstance(m[2], (int, float)) else 1
            out[day][key] += qty
    return out


def fidelity(rep, env_steps, tape_seat):
    """Per day, tiles where the live tape farm differs from the recorded one."""
    bad = {}
    for d in range(30):
        i = d * TPD
        if i >= len(rep["steps"]) or i >= len(env_steps):
            break
        a = obs0(rep, i)["farms"][tape_seat]["tiles"]
        b = env_steps[i][0].observation["farms"][tape_seat]["tiles"]
        diff = 0
        for y in range(len(a)):
            for x in range(len(a[y])):
                ta, tb = a[y][x], b[y][x]
                ka = ta if not isinstance(ta, dict) else (ta.get("kind"), ta.get("crop"), ta.get("animal"))
                kb = tb if not isinstance(tb, dict) else (tb.get("kind"), tb.get("crop"), tb.get("animal"))
                if ka != kb:
                    diff += 1
        bad[d] = diff
    return bad


def run_ghost(rep, my_agent, tape_seat, my_seed=0):
    install_world(rep, tape_seat, my_seed)
    try:
        seed = (rep.get("info") or {}).get("seed")
        cfg = {"episodeSteps": len(rep["steps"])}
        if seed is not None:
            cfg["seed"] = seed
        env = make("kaggriculture", configuration=cfg, debug=True)
        agents = [None, None]
        agents[tape_seat] = tape_agent(rep, tape_seat)
        agents[1 - tape_seat] = my_agent
        env.run(agents)
    finally:
        uninstall_world()
    return env


def report(rep, env, tape_seat, name, verbose=True):
    me = 1 - tape_seat
    names = team_names(rep)
    D = n_days(rep)
    orig_tape = [obs0(rep, d * TPD)["farms"][tape_seat]["money"] for d in range(D)]
    fin_rep = rep["steps"][-1]
    orig_final = (fin_rep[tape_seat].get("reward"), fin_rep[me].get("reward"))
    fin = env.steps[-1]
    live_tape, live_me = fin[tape_seat].reward, fin[me].reward
    fid = fidelity(rep, env.steps, tape_seat)
    if verbose:
        print(f"\n=== {name}")
        print(f"tape = seat {tape_seat} ({names[tape_seat]}), you = seat {me}; seed {(rep.get('info') or {}).get('seed')}")
        print(f"shops: {obs0(rep, len(rep['steps']) - 1)['town']['unlocked_shops']}")
        print(f"original final: top {orig_final[0]}  vs  {names[me]} {orig_final[1]}")
        print(f"ghost final:    top {live_tape}  vs  you {live_me}   -> {'WIN' if live_me > live_tape else 'LOSS' if live_me < live_tape else 'TIE'}")
        print("\nday |   you $ |  top $ live | top $ orig | tiles diverged | you: land hands mix                     | top: land hands mix")
        for d in range(D):
            i = d * TPD
            if i >= len(env.steps):
                break
            o = env.steps[i][0].observation
            fm, ft = o["farms"][me], o["farms"][tape_seat]
            h = min(i + 5, len(env.steps) - 1)
            hm = len(env.steps[h][0].observation["farms"][me]["hands"])
            ht = len(env.steps[h][0].observation["farms"][tape_seat]["hands"])
            mm = ",".join(f"{k[:4]}{v}" for k, v in tile_mix(fm).most_common(5))
            mt = ",".join(f"{k[:4]}{v}" for k, v in tile_mix(ft).most_common(5))
            print(f"{d:>3} | {fm['money']:>7.0f} | {ft['money']:>11.0f} | {orig_tape[d]:>10.0f} | {fid.get(d, 0):>14} | "
                  f"q{len(fm['unlocked_quadrants'])} h{hm:<2} {mm:<32} | q{len(ft['unlocked_quadrants'])} h{ht:<2} {mt}")
        # what the top agent does differently, by day block
        cmd_top = commands_per_day(rep["steps"], tape_seat)
        cmd_me = commands_per_day([[{"action": st[k].action} for k in range(2)] for st in env.steps], me)
        print("\nbiggest command gaps (top minus you), per 5-day block:")
        for b0 in range(0, 30, 5):
            tot_t, tot_m = Counter(), Counter()
            for d in range(b0, b0 + 5):
                tot_t.update(cmd_top.get(d, {}))
                tot_m.update(cmd_me.get(d, {}))
            keys = set(tot_t) | set(tot_m)
            gaps = sorted(((tot_t[k] - tot_m[k], k) for k in keys if k != "MOVE"), key=lambda x: -abs(x[0]))[:6]
            print(f"  days {b0:>2}-{b0+4:<2}: " + "  ".join(f"{k} {g:+d}" for g, k in gaps))
    drift = max(fid.values()) if fid else 0
    return {"name": name, "you": live_me, "top": live_tape, "top_orig": orig_final[0], "drift": drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--top", type=int, default=None, help="tape seat (default: replay winner)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    paths, agent_path = [], None
    for p in args.replays:
        if p.endswith(".py"):
            agent_path = p
        else:
            paths.extend(sorted(glob.glob(p)))
    if not args.selfcheck and not agent_path:
        sys.exit("give your agent .py after the replay files")

    rows = []
    for path in paths:
        rep = load_replay(path)
        fin = rep["steps"][-1]
        seat = args.top if args.top is not None else (0 if (fin[0].get("reward") or 0) >= (fin[1].get("reward") or 0) else 1)
        if args.selfcheck:
            install_world(rep, seat, forced_seats={0, 1})
            try:
                env = make("kaggriculture", configuration={"episodeSteps": len(rep["steps"]),
                                                           "seed": (rep.get("info") or {}).get("seed")}, debug=True)
                env.run([tape_agent(rep, 0), tape_agent(rep, 1)])
            finally:
                uninstall_world()
            got = [s.reward for s in env.steps[-1]]
            want = [fin[0].get("reward"), fin[1].get("reward")]
            print(f"{path}: replay {want}  re-simulated {got}  -> {'EXACT' if got == want else 'MISMATCH'}")
            continue
        import importlib.util
        spec = importlib.util.spec_from_file_location("my_agent_mod", agent_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env = run_ghost(rep, mod.agent, seat)
        rows.append(report(rep, env, seat, path, verbose=not args.summary))
        if args.summary:
            r = rows[-1]
            print(f"{path}: you {r['you']:.0f}  top {r['top']:.0f} (orig {r['top_orig']})  "
                  f"{'WIN' if r['you'] > r['top'] else 'LOSS'}  tape-drift {r['drift']} tiles")
    if rows and len(rows) > 1:
        w = sum(r["you"] > r["top"] for r in rows)
        gap = sum(r["you"] - r["top"] for r in rows) / len(rows)
        print(f"\n{len(rows)} replays: you win {w}/{len(rows)}, mean gap {gap:+.0f}")


if __name__ == "__main__":
    main()
