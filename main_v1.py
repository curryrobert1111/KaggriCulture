"""
Kaggriculture agent — market-aware planner + greedy task scheduler.

Structure
---------
1. Game constants (mirrors kaggriculture.py).
2. Market model: exact price function, expected town consumption,
   pending-supply forecast for both farms.
3. Tile planner: for each free tile pick the use (crop or animal) with the best
   value per tile-day, given projected prices after everyone's pending supply.
4. Job builder: per tile, the ordered list of actions it needs right now.
5. Scheduler: greedy (unit, tile) matching on value - travel cost.
6. Market orders: sell / hire / land / animals / seeds / wheat for feed.

All tunable knobs live in PARAMS so a local tournament can tune them.
"""
import math

# ----------------------------------------------------------------------------
# 1. Constants
# ----------------------------------------------------------------------------
CROPS = {
    "WHEAT":      {"seed": 10, "fyd": 2, "myd": 4, "interval": 0, "max": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "fyd": 2, "myd": 3, "interval": 0, "max": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "fyd": 8, "myd": 8, "interval": 1, "max": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "fyd": 10, "myd": 10, "interval": 2, "max": 4, "ongoing": True},
    "MELON":      {"seed": 80, "fyd": 10, "myd": 12, "interval": 0, "max": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "fyd": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "fyd": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "fyd": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
MARKET_PARAMS = {
    "WHEAT":      {"base": 25, "I0": 10000, "T": 400, "bf": "sqrt", "bt": 0.80, "af": "log", "at": 0.20},
    "CARROT":     {"base": 35, "I0": 10000, "T": 450, "bf": "hinge", "bt": 1.00, "af": "sqrt", "at": 0.70},
    "TOMATO":     {"base": 60, "I0": 10000, "T": 200, "bf": "hinge", "bt": 0.40, "af": "sqrt", "at": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "bf": "sqrt", "bt": 0.70, "af": "linear", "at": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "bf": "log", "bt": 0.20, "af": "sq", "at": 3.60},
    "EGG":        {"base": 50, "I0": 10000, "T": 332, "bf": "hinge", "bt": 0.40, "af": "log", "at": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "bf": "sqrt", "bt": 0.60, "af": "linear", "at": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "bf": "log", "bt": 0.20, "af": "sq", "at": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "bf": "linear", "bt": 0.40, "af": "linear", "at": 0.40},
}
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
TPD = 24
DAYS = 30
BOARD = 10
HALF = BOARD // 2
SHED_TILES = [(HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF)]
LAST_DAY = DAYS - 1
LAST_ACT_HOUR = 22          # step 718 is the last processed step
MAX_SHOPS = 8
SHOP_UNLOCK = 3
SHED_CAP = 100

PARAMS = {
    "travel_cost": 3.0,       # $ per step of walking in the scheduler
    "labor_per_action": 4.0,  # $ cost of one action when valuing tile uses
    "max_hands": 16,
    "hand_marginal_cap": 400, # never pay more than this for the next hand
    "land_reserve": 600,      # cash to keep after buying land
    "land_last_day": 20,
    "sell_hold_ratio": 1.15,  # hold if forecast price > ratio * now
    "hold_shed_limit": 60,
    "opp_weight": 1.0,        # weight on opponent's pending supply
    "drop_value": 350,        # carry value that triggers a shed drop trip
    "cash_reserve": 150,
    "min_rate": 3.0,          # min $/tile/day to bother planting
    "cap_lambda": 0.12,       # $ per $ per day charged on tied-up capital
    "cap_ref": 3000,          # lambda scales with cap_ref / money
    "op_reserve": 250,        # cash kept for hands + feed
    "travel_per_job": 2.5,
    "dist_offset": 0.5,
    "wheat_buffer": 4,
    "use_fert": True,
    "land_plant_cash": 1000,
    "lam_schedule": [1.0],
    "fill_reserve": 12,        # cash held back per still-unplanned tile
    "land_max_empty": 3,
    "land_steps": None,        # e.g. [150, 218, 222]: buy quadrant k at/after this step
    "land_sched_reserve": 300,
    "hire_last_hour": 14,
    "unit_efficiency": 0.8,
}

# ----------------------------------------------------------------------------
# 2. Market model
# ----------------------------------------------------------------------------
def _shape(f, x, T):
    x = max(0.0, x)
    if f == "linear": return x
    if f == "sq": return x * x
    if f == "sqrt": return math.sqrt(x)
    if f == "log": return math.log(1.0 + x)
    if f == "log10": return math.log10(1.0 + x)
    if f == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x

_AMP = {}
for _k, _p in MARKET_PARAMS.items():
    _AMP[_k] = (_p["bt"] * _p["base"] / _shape(_p["bf"], _p["T"], _p["T"]),
                _p["at"] * _p["base"] / _shape(_p["af"], _p["T"], _p["T"]))


def price(item, inv):
    p = MARKET_PARAMS[item]
    I0 = p["I0"]
    if inv < I0:
        v = p["base"] + _AMP[item][0] * _shape(p["bf"], I0 - inv, p["T"])
    else:
        v = p["base"] - _AMP[item][1] * _shape(p["af"], inv - I0, p["T"])
    return max(1, int(round(v)))


def sell_units_value(item, inv, n):
    """Revenue from selling n units starting at market inventory inv."""
    tot = 0.0
    for j in range(int(n)):
        pr = price(item, inv + j)
        tot += pr
        if pr > 1:
            continue
        # at floor: remaining units all sell at 1 and do not move inventory
        tot += (n - j - 1)
        break
    return tot


def shop_daily_rate(shop):
    prods = SHOPS[shop]
    mult = 2 if len(prods) == 1 else 1
    return {it: mult * (TPD // 4) for it in prods}


EXP_SHOP_RATE = {it: 0.0 for it in PRODUCTS}
for _s in SHOPS:
    for _it, _r in shop_daily_rate(_s).items():
        EXP_SHOP_RATE[_it] += _r / len(SHOPS)


def consumption(item, d0, d1, shops):
    """Expected units the town removes between start of day d0 and start of d1."""
    if d1 <= d0:
        return 0.0
    days = d1 - d0
    c = 0.0 if item == "FERTILIZER" else float(days)
    known = 0.0
    for s in shops:
        known += shop_daily_rate(s).get(item, 0)
    c += known * days
    n = len(shops)
    for k in range(n + 1, MAX_SHOPS + 1):
        u = SHOP_UNLOCK * k
        if u >= d1:
            break
        c += EXP_SHOP_RATE[item] * (d1 - max(u, d0))
    return c


# ----------------------------------------------------------------------------
# helpers on tiles
# ----------------------------------------------------------------------------
def is_plant(t):
    return isinstance(t, dict) and t.get("kind") == "PLANT"


def is_animal(t):
    return isinstance(t, dict) and t.get("animal") and t.get("kind") in ("COOP", "PASTURE")


def is_empty_struct(t):
    return isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and not t.get("animal")


def is_weed(t):
    return isinstance(t, dict) and t.get("kind") == "WEED"


def productions_done(crop, planted, day):
    """# ongoing productions already available by the start of `day`."""
    cd = CROPS[crop]
    n = 0
    for k in range(cd["max"]):
        if planted + cd["fyd"] + k * cd["interval"] <= day:
            n += 1
    return n


def production_eves(crop, planted):
    """Days D whose end-of-day refresh fires a scheduled production."""
    cd = CROPS[crop]
    return [planted + cd["fyd"] - 1 + k * cd["interval"] for k in range(cd["max"])]


def fert_gain_units(t, day):
    """Extra units from fertilizing this ongoing plant today."""
    if t.get("fertilized_until_day", -1) >= day:
        return 0
    eves = production_eves(t["crop"], t["planted_day"])
    if day not in eves:
        return 0
    return sum(1 for D in eves if day <= D <= day + 2 and D <= LAST_DAY - 1)


def one_time_remaining_increments(t, day):
    """How many more yield units watering can still add to a one-time crop."""
    cd = CROPS[t["crop"]]
    age = day - t["planted_day"]
    ws = (cd["myd"] + 1) // 2
    start = age + (1 if t["watered_today"] else 0)
    days = max(0, cd["myd"] - max(start, ws) + 1)
    fert_days = 0
    fu = t.get("fertilized_until_day", -1)
    if fu >= 0:
        for a in range(max(start, ws), cd["myd"] + 1):
            if t["planted_day"] + a <= fu:
                fert_days += 1
    return min(cd["max"] - t["yield_units"], days + fert_days)


def expected_one_time_yield(t, day):
    return t["yield_units"] + one_time_remaining_increments(t, day)


def farm_pending(farm, day, private=None):
    """Units of each product this farm is expected to still bring to market."""
    pend = {it: 0.0 for it in PRODUCTS}
    rem = max(0, LAST_DAY - day)
    for row in farm["tiles"]:
        for t in row:
            if is_plant(t):
                cd = CROPS[t["crop"]]
                if cd["ongoing"]:
                    left = 0
                    for k in range(cd["max"]):
                        d = t["planted_day"] + cd["fyd"] + k * cd["interval"]
                        if day < d <= LAST_DAY:
                            left += 1
                    pend[t["crop"]] += t["yield_units"] + left
                else:
                    pend[t["crop"]] += expected_one_time_yield(t, day)
            elif is_animal(t):
                a = ANIMALS[t["animal"]]
                pend[a["product"]] += t["yield_units"] + rem * (1 + a["interval"]) / a["interval"] * 0.9
                pend["FERTILIZER"] += rem
    if private:
        for it in PRODUCTS:
            pend[it] += private["shed"].get(it, 0)
            for inv in private["inventories"]:
                pend[it] += inv.get(it, 0)
    return pend


# ----------------------------------------------------------------------------
# 3. Tile-use valuation
# ----------------------------------------------------------------------------
def crop_plan(crop, day):
    """(harvest schedule [(day, units)], occupancy days) for planting today."""
    cd = CROPS[crop]
    if not cd["ongoing"]:
        if crop == "MELON":
            h, u = day + 10, 6
        elif crop == "WHEAT":
            h, u = day + 4, 4
        else:
            h, u = day + 3, 3
        if h > LAST_DAY:
            # partial crop: harvest on the last day if at least first_yield_day
            if day + cd["fyd"] > LAST_DAY:
                return None
            h = LAST_DAY
            age = h - day
            ws = (cd["myd"] + 1) // 2
            u = 1 + max(0, age - ws + 1)
        return [(h, u)], h - day
    sched = []
    for k in range(cd["max"]):
        d = day + cd["fyd"] + k * cd["interval"]
        if d <= LAST_DAY:
            sched.append((d, 1))
    if not sched:
        return None
    return sched, sched[-1][0] - day


def value_crop(crop, day, market_inv, extra, shops, labor):
    plan = crop_plan(crop, day)
    if plan is None:
        return None
    sched, occ = plan
    units = sum(u for _, u in sched)
    sale_day = sched[len(sched) // 2][0]
    inv = market_inv[crop] + extra[crop] - consumption(crop, day, sale_day, shops)
    rev = sell_units_value(crop, inv, units)
    cd = CROPS[crop]
    actions = occ * (0.6 if not cd["ongoing"] else 0.5) + 2 + len(sched)
    val = rev - cd["seed"] - actions * labor
    occ = max(1, occ)
    return {"kind": "crop", "item": crop, "units": units, "value": val,
            "rate": val / occ, "cost": cd["seed"], "occ": occ}


def value_animal(animal, day, market_inv, extra, shops, labor, place_day=None):
    a = ANIMALS[animal]
    p = day if place_day is None else place_day
    R = LAST_DAY - p  # production days available (last production end of day 28)
    if R <= a["fyd"]:
        return None
    # first production gets 1 + banked care (capped at max_held)
    units = min(a["max_held"], 1 + a["fyd"] - 1)
    rest = R - a["fyd"]
    units += (rest // a["interval"]) * (1 + a["interval"])
    prod = a["product"]
    sale_day = p + (a["fyd"] + R) // 2
    inv = market_inv[prod] + extra[prod] - consumption(prod, day, sale_day, shops)
    rev = sell_units_value(prod, inv, units)
    fert_units = R
    finv = market_inv["FERTILIZER"] + extra["FERTILIZER"] - consumption("FERTILIZER", day, sale_day, shops)
    rev_f = sell_units_value("FERTILIZER", finv, fert_units)
    wheat_cost = R * max(25, price("WHEAT", market_inv["WHEAT"] - R))
    actions = R * 4.5 + 4
    val = rev + rev_f - wheat_cost - a["cost"] - actions * labor
    # cash-flow payback: fertilizer (net of feed) from day 1, product from fyd
    f_day = rev_f / max(1, fert_units) - wheat_cost / max(1, R)
    p_day = rev / max(1, units) * (1 + a["interval"]) / a["interval"]
    cum, payback = 0.0, 99
    for d in range(1, R + 1):
        cum += f_day - 4 * labor
        if d == a["fyd"]:
            cum += rev / max(1, units) * min(a["max_held"], a["fyd"])
        elif d > a["fyd"]:
            cum += p_day / (1 + a["interval"])
        if cum >= a["cost"]:
            payback = d
            break
    return {"kind": "animal", "item": animal, "product": prod, "units": units,
            "fert": fert_units, "value": val, "rate": val / max(1, R + 1),
            "cost": a["cost"], "occ": R + 1, "payback": payback}


def adjusted_rate(o, lam):
    """Value per tile-day after charging for the capital tied up."""
    if o["kind"] == "crop":
        payback = o["occ"]
    else:
        payback = o.get("payback", 99)
    return (o["value"] - lam * o["cost"] * payback) / max(1, o["occ"])


# ----------------------------------------------------------------------------
# persistent state (keyed by player so local self-play in one process is safe)
# ----------------------------------------------------------------------------
_STATE = {}


def _state(player):
    s = _STATE.get(player)
    if s is None:
        s = {"plans": {}, "targets": {}, "last_step": -1}
        _STATE[player] = s
    return s


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_shed(pos):
    return min(SHED_TILES, key=lambda s: manhattan(pos, s))


def move_toward(pos, target):
    x, y = pos
    tx, ty = target
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return ["PASS"]


def unlocked(tile):
    return tile != "LOCKED"


# ----------------------------------------------------------------------------
# 4. Plan free tiles
# ----------------------------------------------------------------------------
def plan_tiles(me, opp, private, market, shops, day, hour, st, budget):
    """Shadow-price search: raise the capital charge until cash covers the free tiles."""
    base_plans = st["plans"]
    best = None
    for mult in PARAMS["lam_schedule"]:
        trial = {k: dict(v) for k, v in base_plans.items()}
        res, starved = _plan_tiles(me, opp, private, market, shops, day, hour, trial, budget, mult)
        best = res
        if starved == 0:
            break
    st["plans"] = best
    return best


def _plan_tiles(me, opp, private, market, shops, day, hour, plans, budget, lam_mult):
    tiles = me["tiles"]
    # drop stale plans
    for key in list(plans.keys()):
        x, y = key
        t = tiles[y][x]
        pl = plans[key]
        if pl["kind"] == "crop" and (t is not None and not is_weed(t) and not is_empty_struct(t)):
            del plans[key]
        elif pl["kind"] == "animal" and is_animal(t):
            del plans[key]
        elif pl["kind"] == "crop" and pl.get("day") != day:
            del plans[key]   # re-plan crops daily with fresh prices
        elif pl["kind"] == "animal" and pl.get("day") != day and t is None:
            del plans[key]   # not built yet: re-plan with fresh prices

    extra = farm_pending(me, day, private)
    oppp = farm_pending(opp, day)
    for it in PRODUCTS:
        extra[it] += PARAMS["opp_weight"] * oppp[it]
    for pl in plans.values():
        if pl["kind"] == "crop":
            extra[pl["item"]] += pl["units"]
        else:
            extra[pl["product"]] += pl["units"]
            extra["FERTILIZER"] += pl["fert"]
            budget -= 0  # structure already paid? animals bought later

    # committed spend for existing unexecuted plans
    for pl in plans.values():
        budget -= pl["cost"]

    free = []
    for y in range(BOARD):
        for x in range(BOARD):
            t = tiles[y][x]
            if t == "LOCKED" or (x, y) in plans:
                continue
            if t is None or is_weed(t):
                free.append((x, y))
            elif is_empty_struct(t):
                # a structure whose animal escaped / never placed: plan animal
                free.append((x, y))
    # animals near the shed (visited often), crops further out
    free.sort(key=lambda p: min(manhattan(p, s) for s in SHED_TILES))

    late = hour > LAST_ACT_HOUR - 2
    labor = PARAMS["labor_per_action"]
    lam = lam_mult * PARAMS["cap_lambda"] * min(3.0, max(0.1, PARAMS["cap_ref"] / max(1.0, float(me["money"]))))
    starved = 0
    for idx_free, pos in enumerate(free):
        x, y = pos
        t = tiles[y][x]
        opts = []
        hold = PARAMS["fill_reserve"] * (len(free) - idx_free - 1)
        plant_day = day if not late else day + 1
        if is_empty_struct(t):
            kinds = [a for a in ANIMALS if ANIMALS[a]["structure"] == t["kind"]]
            for a in kinds:
                v = value_animal(a, day, market["inventory"], extra, shops, labor)
                if v: opts.append(v)
            for c in CROPS:
                v = value_crop(c, plant_day, market["inventory"], extra, shops, labor)
                if v:
                    v["value"] -= labor * 2
                    v["rate"] = v["value"] / v["occ"]
                    opts.append(v)
        else:
            for c in CROPS:
                v = value_crop(c, plant_day, market["inventory"], extra, shops, labor)
                if v: opts.append(v)
            for a in ANIMALS:
                v = value_animal(a, day, market["inventory"], extra, shops, labor)
                if v: opts.append(v)
        best = None
        for o in opts:
            if o["cost"] > budget - (hold if o["cost"] > PARAMS["fill_reserve"] else 0):
                if adjusted_rate(o, lam) >= PARAMS["min_rate"]:
                    starved += 1
                continue
            adj = adjusted_rate(o, lam)
            if adj < PARAMS["min_rate"]:
                continue
            if best is None or adj > best["adj"]:
                best = dict(o)
                best["adj"] = adj
        if best is None:
            continue
        starved = starved  # noqa
        best = dict(best)
        best["day"] = day
        plans[pos] = best
        budget -= best["cost"]
        if best["kind"] == "crop":
            extra[best["item"]] += best["units"]
        else:
            extra[best["product"]] += best["units"]
            extra["FERTILIZER"] += best["fert"]
    return plans, starved


# ----------------------------------------------------------------------------
# 5. Jobs
# ----------------------------------------------------------------------------
def build_jobs(me, private, market, day, hour, plans, seeds_avail):
    """pos -> {"acts": [(action, value, need_item_or_None), ...]} in execution order."""
    jobs = {}
    # animals already owned (shed + carried) but not placed: sunk cost, place them
    spare = {}
    for a in ANIMALS:
        n = private["shed"].get(a, 0) + sum(i.get(a, 0) for i in private["inventories"])
        if n > 0:
            spare[a] = n
    tiles = me["tiles"]
    prices = market["prices"]
    last_day = day == LAST_DAY
    for y in range(BOARD):
        for x in range(BOARD):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            acts = []
            if is_plant(t):
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                pr = prices[t["crop"]]
                if not cd["ongoing"]:
                    inc = one_time_remaining_increments(t, day)
                    ws = (cd["myd"] + 1) // 2
                    in_window = ws <= age <= cd["myd"]
                    if not t["watered_today"]:
                        wv = 0
                        if in_window and inc > 0:
                            wv += pr * (2 if t.get("fertilized_until_day", -1) >= day else 1)
                        if t["consecutive_unwatered"] >= 1 and not last_day:
                            # dies tonight if not watered: worth the whole plant
                            wv += 20 + pr * expected_one_time_yield(t, day)
                        if wv > 0:
                            acts.append((["WATER"], wv, None))
                    if age >= cd["fyd"] and (inc == 0 or last_day or age >= cd["myd"]):
                        acts.append((["HARVEST"], 40 + pr * t["yield_units"] * 0.5, None))
                    elif age >= cd["fyd"] and t["watered_today"] and inc == 0:
                        acts.append((["HARVEST"], 40, None))
                else:
                    done = productions_done(t["crop"], t["planted_day"], day)
                    finished = done >= cd["max"]
                    if not t["watered_today"] and not finished and not last_day:
                        fert_active = t.get("fertilized_until_day", -1) >= day
                        if t["consecutive_unwatered"] >= 1 or fert_active:
                            left = t["yield_units"] + (cd["max"] - done)
                            acts.append((["WATER"], 20 + pr * left + (pr if fert_active else 0), None))
                    g = fert_gain_units(t, day) if PARAMS["use_fert"] else 0
                    if g > 0:
                        fv = g * pr - prices["FERTILIZER"]
                        if fv > 0:
                            # harvest first so the doubled yield does not hit the held cap
                            acts.append((["FERTILIZE"], fv, "FERTILIZER"))
                            if t["watered_today"] is False and not any(a[0][0] == "WATER" for a in acts):
                                acts.append((["WATER"], 30, None))
                    if t["yield_units"] > 0:
                        v = 20 + pr * t["yield_units"] * 0.4 + (60 if t["yield_units"] >= cd["max"] - 1 else 0)
                        acts.insert(0, (["HARVEST"], v, None))
                    elif finished:
                        acts.append((["DIG"], 25, None))
            elif is_animal(t):
                a = ANIMALS[t["animal"]]
                if not t["fed_today"] and day < LAST_DAY:
                    acts.append((["FEED"], 90 + (400 if t["consecutive_unfed"] >= 1 else 0), "WHEAT"))
                if not t["cared_today"] and day < LAST_DAY - 1:
                    acts.append((["CARE"], 25, None))
                if t.get("fertilizer_available"):
                    acts.append((["COLLECT_FERTILIZER"], 40, None))
                if t["yield_units"] > 0:
                    v = 15 + 0.3 * prices[a["product"]] * t["yield_units"]
                    if t["yield_units"] >= a["max_held"] - 1:
                        v += 60
                    acts.append((["HARVEST"], v, None))
            elif (is_empty_struct(t) and day < LAST_DAY and
                  any(spare.get(a, 0) > 0 and ANIMALS[a]["structure"] == t["kind"] for a in ANIMALS)):
                pl = plans.get((x, y))
                cand = [a for a in ANIMALS if spare.get(a, 0) > 0 and ANIMALS[a]["structure"] == t["kind"]]
                a = pl["item"] if pl and pl["kind"] == "animal" and pl["item"] in cand else cand[0]
                spare[a] -= 1
                acts.append((["PLACE", a], 120, a))
            elif (t is None or is_weed(t) or is_empty_struct(t)) and (x, y) in plans:
                pl = plans[(x, y)]
                if is_weed(t) or (pl["kind"] == "crop" and is_empty_struct(t)):
                    acts.append((["DIG"], 30, None))
                elif pl["kind"] == "crop":
                    if hour <= LAST_ACT_HOUR - 1 and seeds_avail.get(pl["item"], 0) > 0:
                        acts.append((["PLANT", pl["item"]], 50 + max(0, pl.get("adj", 0)), None))
                elif pl["kind"] == "animal":
                    struct = ANIMALS[pl["item"]]["structure"]
                    if t is None:
                        acts.append((["BUILD_" + struct], 45, None))
                    elif is_empty_struct(t) and t["kind"] == struct:
                        acts.append((["PLACE", pl["item"]], 90, pl["item"]))
            if acts:
                jobs[(x, y)] = {"acts": acts}
    return jobs


def job_value_for(job, inv, shed):
    """(value, needs_pickup_item) of the actions this unit can do."""
    v = 0.0
    pick = None
    for a, val, need in job["acts"]:
        if need is None or inv.get(need, 0) > 0:
            v += val
        elif shed.get(need, 0) > 0:
            v += val
            pick = need
    return v, pick


SELLABLE = set(PRODUCTS)


def carry_value(inv, prices):
    v = 0
    for it, n in inv.items():
        if it in prices and it != "WHEAT":
            v += prices[it] * n
    return v


# ----------------------------------------------------------------------------
# main agent
# ----------------------------------------------------------------------------
def _agent(obs):
    player = obs["player"]
    farms = obs["farms"]
    me = farms[player]
    opp = farms[1 - player]
    private = obs["private"]
    market = obs["market"]
    shops = list(obs["town"]["unlocked_shops"])
    day = obs["day"]
    hour = obs["hour"]
    step = day * TPD + hour
    st = _state(player)
    if step < st["last_step"]:
        # new game in the same process
        _STATE[player] = None
        _STATE.pop(player, None)
        st = _state(player)
    st["last_step"] = step

    tiles = me["tiles"]
    shed = dict(private["shed"])
    seeds = dict(private["seeds"])
    invs = [dict(i) for i in private["inventories"]]
    units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
    while len(invs) < len(units):
        invs.append({})
    prices = market["prices"]
    money = float(me["money"])
    orders = []

    n_animals = sum(1 for row in tiles for t in row if is_animal(t))

    # ---------------- land ----------------
    n_extra = len(me["unlocked_quadrants"]) - 1
    if n_extra < len(LAND_ORDER) and day <= PARAMS["land_last_day"] and hour <= 18:
        cost = LAND_PRICES[n_extra]
        n_empty = sum(1 for row in tiles for t in row if t is None or is_weed(t))
        sched = PARAMS.get("land_steps")
        if sched and n_extra < len(sched):
            ok = step >= sched[n_extra] and money - cost >= PARAMS["land_sched_reserve"]
        else:
            ok = (money - cost >= PARAMS["land_reserve"] * (1 + n_extra) + PARAMS["land_plant_cash"]
                  and n_empty <= PARAMS["land_max_empty"])
        if ok:
            orders.append(["BUY_LAND"])
            money -= cost
            q = LAND_ORDER[n_extra]
            # optimistic local unlock so we plan these tiles now
            for y in range(BOARD):
                for x in range(BOARD):
                    if tiles[y][x] == "LOCKED" and (("N" if y < HALF else "S") + ("W" if x < HALF else "E")) == q:
                        tiles[y][x] = None

    # ---------------- planning ----------------
    budget = money - PARAMS["op_reserve"] - (n_animals * 60 if day < LAST_DAY else 0)
    plans = plan_tiles(me, opp, private, market, shops, day, hour, st, budget)

    # seeds needed for crop plans on tiles that are plantable now
    seed_need = {}
    for (x, y), pl in plans.items():
        t = tiles[y][x]
        if pl["kind"] == "crop" and (t is None or is_weed(t)):
            seed_need[pl["item"]] = seed_need.get(pl["item"], 0) + 1
    seed_orders = []
    for c, n in seed_need.items():
        buy = n - seeds.get(c, 0)
        if buy > 0 and hour <= LAST_ACT_HOUR - 2:
            seed_orders.append(["BUY_SEED", c, buy])

    # animals needed for empty structures
    animal_orders = []
    want_animals = {}
    for (x, y), pl in plans.items():
        t = tiles[y][x]
        if (pl["kind"] == "animal" and is_empty_struct(t)
                and t["kind"] == ANIMALS[pl["item"]]["structure"]):
            want_animals[pl["item"]] = want_animals.get(pl["item"], 0) + 1
    for a, n in want_animals.items():
        have = shed.get(a, 0) + sum(i.get(a, 0) for i in invs)
        buy = n - have
        if buy > 0 and hour <= LAST_ACT_HOUR - 4:
            animal_orders.append(["BUY_ANIMAL", a, buy])

    # wheat for feed: today's unfed animals + animals to be placed
    need_feed = sum(1 for row in tiles for t in row if is_animal(t) and not t["fed_today"])
    if day < LAST_DAY:
        need_feed += sum(want_animals.values())
    have_wheat = shed.get("WHEAT", 0) + sum(i.get("WHEAT", 0) for i in invs)
    wheat_orders = []
    want_wheat = need_feed + (PARAMS["wheat_buffer"] if need_feed > 0 else 0)
    if want_wheat > have_wheat and day < LAST_DAY:
        wheat_orders.append(["BUY_PRODUCT", "WHEAT", want_wheat - have_wheat])

    # ---------------- jobs & scheduling ----------------
    seeds_after = dict(seeds)  # seeds bought this turn arrive after actions
    jobs = build_jobs(me, private, market, day, hour, plans, seeds_after)

    fert_need = sum(1 for j in jobs.values() for a in j["acts"] if a[0][0] == "FERTILIZE")
    st["fert_need"] = max(1, int(math.ceil(fert_need / 3.0)))
    last_day = day == LAST_DAY
    unit_actions = []
    if st.get("tday") != day:
        st["targets"] = {}
        st["tday"] = day
    targets = st["targets"]
    plant_count = {}

    def shed_dist(p):
        return min(manhattan(p, s_) for s_ in SHED_TILES)

    def carrying(inv):
        return any(n > 0 for it, n in inv.items() if it in SELLABLE)

    def drop_needed(ui, pos):
        inv = invs[ui]
        cv = carry_value(inv, prices)
        must = last_day and carrying(inv) and hour + shed_dist(pos) >= LAST_ACT_HOUR - 1
        return must or cv >= PARAMS["drop_value"]

    def feasible_last_day(pos, tpos, n_acts):
        # reach tile, act, walk back and DROP by the last processed hour
        return hour + manhattan(pos, tpos) + n_acts + shed_dist(tpos) + 1 <= LAST_ACT_HOUR

    # 1) keep valid targets
    taken = set()
    for ui in list(targets.keys()):
        tpos = targets[ui]
        if ui >= len(units):
            del targets[ui]
            continue
        if tpos == "SHED":
            if not any(n > 0 for it, n in invs[ui].items() if it in SELLABLE):
                del targets[ui]
            continue
        if tpos not in jobs or job_value_for(jobs[tpos], invs[ui], shed)[0] <= 0:
            del targets[ui]
            continue
        if last_day and (drop_needed(ui, units[ui]) or
                         not feasible_last_day(units[ui], tpos, len(jobs[tpos]["acts"]))):
            targets[ui] = "SHED" if carrying(invs[ui]) else None
            if targets[ui] is None:
                del targets[ui]
            continue
        taken.add(tpos)

    # 2) assign free units greedily (best score first)
    free_units = [ui for ui in range(len(units)) if ui not in targets]
    pairs = []
    for ui in free_units:
        pos = units[ui]
        inv = invs[ui]
        if drop_needed(ui, pos):
            targets[ui] = "SHED"
            continue
        for tpos, job in jobs.items():
            if tpos in taken:
                continue
            v, pick = job_value_for(job, inv, shed)
            if v <= 0:
                continue
            if pick:
                s_ = nearest_shed(pos)
                d = manhattan(pos, s_) + 1 + manhattan(s_, tpos)
            else:
                d = manhattan(pos, tpos)
            if last_day and not feasible_last_day(pos, tpos, len(job["acts"]) + (1 if pick else 0)):
                continue
            n_acts = len(job["acts"])
            pairs.append((v / (d + n_acts + PARAMS["dist_offset"]), ui, tpos))
    pairs.sort(key=lambda p: -p[0])
    for sc, ui, tpos in pairs:
        if ui in targets or tpos in taken or sc <= 0:
            continue
        targets[ui] = tpos
        taken.add(tpos)

    # 3) act
    picked = st.setdefault("picked", {})
    if picked.get("day") != day:
        picked.clear()
        picked["day"] = day
    carried_wheat = sum(i.get("WHEAT", 0) for i in invs)
    for ui, pos in enumerate(units):
        inv = invs[ui]
        tpos = targets.get(ui)
        act = None
        # start-of-day wheat share for feeding
        if (day < LAST_DAY and pos in SHED_TILES and inv.get("WHEAT", 0) == 0
                and shed.get("WHEAT", 0) > 0 and need_feed > carried_wheat and ui not in picked):
            crew = max(len(units), st.get("crew", 1))
            share = min(shed["WHEAT"] - PARAMS["wheat_buffer"], need_feed - carried_wheat,
                        int(math.ceil(need_feed / crew)) + 1)
            if share > 0:
                act = ["PICKUP", "WHEAT", share]
                shed["WHEAT"] -= share
                carried_wheat += share
                picked[ui] = True
        if act is not None:
            unit_actions.append(act)
            continue
        act = ["PASS"]
        if tpos == "SHED":
            s_ = nearest_shed(pos)
            act = ["DROP"] if pos == s_ else move_toward(pos, s_)
        elif tpos is not None:
            job = jobs[tpos]
            v, pick = job_value_for(job, inv, shed)
            act = None
            if pos == tpos:
                for a_, val, need in job["acts"]:
                    if need and inv.get(need, 0) <= 0:
                        continue
                    if a_[0] == "PLANT":
                        c = a_[1]
                        if plant_count.get(c, 0) + 1 > seeds.get(c, 0):
                            continue
                        plant_count[c] = plant_count.get(c, 0) + 1
                    act = a_
                    break
                if act is None and not pick:
                    del targets[ui]
                    act = ["PASS"]
            if act is None:
                if pick:
                    s_ = nearest_shed(pos)
                    if pos == s_:
                        act = _pickup(pick, shed, need_feed, units, st)
                    else:
                        act = move_toward(pos, s_)
                elif v <= 0:
                    del targets[ui]
                    act = ["PASS"]
                else:
                    act = move_toward(pos, tpos)
        else:
            if pos in SHED_TILES and carrying(inv):
                act = ["DROP"]
            elif last_day and carrying(inv):
                act = move_toward(pos, nearest_shed(pos))
        unit_actions.append(act)

    # which items will reach the shed this turn
    incoming = {}
    for ui, act in enumerate(unit_actions):
        if act and act[0] == "DROP":
            for it, n in invs[ui].items():
                incoming[it] = incoming.get(it, 0) + n

    # ---------------- selling ----------------
    sell_orders = []
    shed_total = sum(shed.values())
    wheat_keep = n_animals + sum(want_animals.values()) + PARAMS["wheat_buffer"] + 2 if day < LAST_DAY else 0
    ext = farm_pending(opp, day)
    fert_keep = 0
    if PARAMS["use_fert"] and not last_day:
        for row in tiles:
            for t in row:
                if is_plant(t) and CROPS[t["crop"]]["ongoing"]:
                    eves = production_eves(t["crop"], t["planted_day"])
                    if any(day <= D <= day + 1 for D in eves):
                        fert_keep += 1
        fert_keep = max(0, fert_keep - sum(i.get("FERTILIZER", 0) for i in invs))
    for it in PRODUCTS:
        avail = shed.get(it, 0) + incoming.get(it, 0)
        if it == "WHEAT":
            avail -= wheat_keep
        if it == "FERTILIZER" and PARAMS["use_fert"]:
            avail -= fert_keep
        if avail <= 0:
            continue
        inv = market["inventory"][it]
        now_p = price(it, inv)
        n_sell = avail
        if not last_day and it not in ("WHEAT", "FERTILIZER") and shed_total < PARAMS["hold_shed_limit"]:
            horizon = min(LAST_DAY, day + 3)
            fut_inv = inv - consumption(it, day, horizon, shops) + ext.get(it, 0) * 0.5
            fut_p = price(it, fut_inv)
            if fut_p > now_p * PARAMS["sell_hold_ratio"]:
                # sell only units that stay above the forecast
                n_sell = 0
                while n_sell < avail and price(it, inv + n_sell) >= fut_p:
                    n_sell += 1
        if n_sell > 0:
            sell_orders.append(["SELL", it, int(n_sell)])

    # ---------------- hiring ----------------
    hire_orders = []
    if hour <= PARAMS["hire_last_hour"]:
        workload = 0.0
        for tpos, job in jobs.items():
            workload += len(job["acts"]) + PARAMS["travel_per_job"]
        for (x, y), pl in plans.items():
            if (x, y) not in jobs:
                workload += 2 + PARAMS["travel_per_job"]
        workload += n_animals * 0.3  # wheat pickups
        hours_left = max(1, LAST_ACT_HOUR - hour)
        want_units = int(math.ceil(workload / (hours_left * PARAMS["unit_efficiency"])))
        want_hands = max(0, min(PARAMS["max_hands"], want_units - 1))
        if hour == 0:
            st["crew"] = want_hands + 1
        hired = len(me["hands"])
        n_h = me["hires_today"]
        spend = money
        while hired < want_hands and len(hire_orders) < 8:
            cost = _fib(n_h)
            if cost > PARAMS["hand_marginal_cap"] or cost > spend - 30:
                break
            hire_orders.append(["HIRE"])
            spend -= cost
            hired += 1
            n_h += 1

    # order priority: sells first (cash), land, hires, wheat, animals, seeds
    land_orders = [o for o in orders if o[0] == "BUY_LAND"]
    all_orders = sell_orders + land_orders + wheat_orders + hire_orders + animal_orders + seed_orders
    all_orders = all_orders[:10]

    return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": all_orders}


def _pickup(need, shed, need_feed, units, st):
    if need == "WHEAT":
        n = max(1, min(shed.get("WHEAT", 0), need_feed // max(1, len(units)) + 3))
        shed["WHEAT"] = shed.get("WHEAT", 0) - n
        return ["PICKUP", "WHEAT", n]
    if need == "FERTILIZER":
        n = max(1, min(shed.get(need, 0), st.get("fert_need", 1)))
        shed[need] = shed.get(need, 0) - n
        return ["PICKUP", need, n]
    shed[need] = shed.get(need, 0) - 1
    return ["PICKUP", need, 1]


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# Kaggle calls the LAST callable defined in main.py, so `agent` must stay last.
def agent(obs, config=None):
    try:
        return _agent(obs)
    except Exception as e:  # never crash on the ladder
        import traceback
        traceback.print_exc()
        return {"farmer": ["PASS"], "hands": [], "market": []}
