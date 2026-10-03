"""Lean Kaggriculture simulator: drives the real engine's interpreter without
kaggle_environments' per-step deepcopies (~5x faster). Agents get the live
observation and MUST NOT mutate it.

  run_game(agents, seed)                       -> (rewards, sim)
  run_ghost(rep, my_agent, tape_seat)          -> (rewards, sim)   (recorded shops + tape weeds forced)
"""
import json, random, sys
import kaggle_environments.envs.kaggriculture.kaggriculture as K
from kaggle_environments.utils import Struct

CFG = {'actTimeout': 1, 'boardSize': 10, 'episodeSteps': 720, 'farmHandCostMult': 1, 'marketParams': {},
       'maxMarketOrdersPerTurn': 10, 'runTimeout': 1200, 'shedCapacity': 100, 'startingMoney': 3000,
       'townCenterSellInterval': 24, 'townShopSellInterval': 4, 'townShopUnlockInterval': 3,
       'turnsPerDay': 24, 'weedSpawnChance': 0.005}


class Env:
    def __init__(self, seed):
        self.configuration = Struct(**CFG)
        self.configuration.seed = seed
        self.info = {"seed": seed}
        self.done = False


class St:
    __slots__ = ("observation", "action", "status", "reward")


def new_game(seed):
    env = Env(seed)
    state = []
    for i in range(2):
        s = St()
        s.observation = Struct(step=0, remainingOverageTime=60)
        s.action = None
        s.status = "ACTIVE"
        s.reward = 0
        state.append(s)
    # _initialize uses resolve_episode_seed(env): give it what it needs
    K.resolve_episode_seed = lambda e: e.info.get("seed", 0)
    K._initialize(state, env)
    return env, state


def step(env, state, actions, t):
    for i in range(2):
        state[i].action = actions[i]
        state[i].observation.step = t
    K.interpreter(state, env)


def run(agents, seed, record=None):
    env, state = new_game(seed)
    for t in range(CFG["episodeSteps"] - 1):
        acts = []
        for i in range(2):
            o = state[i].observation
            o.step = t
            try:
                a = agents[i](o, env.configuration)
            except Exception as e:  # mirror kaggle: agent error = forfeit-ish; here just pass
                a = {"farmer": ["PASS"], "hands": [], "market": []}
            acts.append(a)
        if record is not None:
            record(t, state, acts)
        step(env, state, acts, t)
        if state[0].status == "DONE":
            break
    return [state[0].reward, state[1].reward], state


# ---------------------------------------------------------------- ghost ----
_ORIG_EOD = K._end_of_day


def install_world(rep, forced_seats, my_seed=0):
    TPD = 24
    n = len(rep["steps"])
    shops_by_day = {d: list(rep["steps"][d * TPD][0]["observation"]["town"]["unlocked_shops"])
                    for d in range(31) if d * TPD < n}
    weeds = {}
    for seat in forced_seats:
        for d in range(30):
            i = (d + 1) * TPD
            if i < n:
                tiles = rep["steps"][i][0]["observation"]["farms"][seat]["tiles"]
                weeds[(seat, d)] = [(x, y) for y, row in enumerate(tiles) for x, t in enumerate(row)
                                    if isinstance(t, dict) and t.get("kind") == "WEED"]

    def eod(state, env, day):
        o = state[0].observation
        rng = random.Random(my_seed * 1_000_003 + day)
        for pid, farm in enumerate(o.farms):
            private = state[pid].observation.private
            K._daily_refresh_plants(farm, day, 24)
            K._daily_refresh_animals(farm, day)
            if pid in forced_seats:
                for (x, y) in weeds.get((pid, day), ()):
                    if farm["tiles"][y][x] is None:
                        farm["tiles"][y][x] = {"kind": "WEED"}
            else:
                K._spawn_weeds(farm, 10, 0.005, rng)
            K._drop_inventories_to_shed(private, 100)
            farm["farmer"] = list(K._default_spawn(10))
            farm["hands"] = []
            farm["hires_today"] = 0
            private["inventories"] = [{}]
        if day + 1 in shops_by_day:
            o.town["unlocked_shops"] = list(shops_by_day[day + 1])
    K._end_of_day = eod


def uninstall():
    K._end_of_day = _ORIG_EOD


def tape(rep, seat):
    steps = rep["steps"]

    def act(obs, cfg=None):
        t = obs["step"]
        if t + 1 < len(steps):
            a = steps[t + 1][seat].get("action")
            if isinstance(a, dict):
                return a
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return act


def run_ghost(rep, my_agent, tape_seat, record=None, my_seed=0):
    install_world(rep, {tape_seat}, my_seed)
    try:
        agents = [None, None]
        agents[tape_seat] = tape(rep, tape_seat)
        agents[1 - tape_seat] = my_agent
        return run(agents, rep["info"]["seed"], record)
    finally:
        uninstall()


_REP = {}
def load(path):
    if path not in _REP:
        _REP[path] = json.load(open(path))
    return _REP[path]


if __name__ == "__main__":
    import importlib.util, time
    spec = importlib.util.spec_from_file_location("m", sys.argv[2]); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    t = time.time()
    r, _ = run_ghost(load(sys.argv[1]), M.agent, int(sys.argv[3]))
    print(r, time.time() - t)
