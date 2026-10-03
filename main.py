"""
Kaggriculture agent — market-aware planner + nearest-first task scheduler.

Structure
---------
1. Game constants (mirrors kaggriculture.py).
2. Market model: exact price function, expected town consumption.
3. Trajectory model (v5): projected inventory of every product for every
   remaining day from BOTH farms' production schedules; a unit sold on day d
   is priced at day d, and each new tile use is charged for the price damage
   it does to our own later sales ("own externality").
4. Tile planner: global greedy over (crop/animal) options by capital-adjusted
   value per tile-day; saves cash for a much better unaffordable option.
5. Job builder: per tile, the ordered list of actions it needs right now.
6. Scheduler: nearest job first (score = value / distance**10), urgent
   critical jobs (plant dies / animal escapes tonight) preempt the nearest
   unit, units water/feed critical tiles they walk over, animals waiting in
   the shed are dispatched to their coop/pasture.
7. Market orders: sell / hire / land / animals / seeds / wheat for feed.

Changes learned from DECEM replays
----------------------------------------------
* Day 0 is DECEM's exact opening tape (2 cows + 3 sheep, melons, wheat),
  guarded step by step; we hand over at the first mismatch.
* Land on a schedule: steps 147 / 198 / 250 (all top teams do this).
* Hands: floor schedule 4,3,5,5,6,6,9,8,9,10,12..., marginal hire cap $150.
* Items stay in the units' pockets (auto-dropped at night) unless cash-poor.

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
    "hand_marginal_cap": 150, # never pay more than this for the next hand
    "land_reserve": 600,      # cash to keep after buying land
    "land_last_day": 20,
    "sell_hold_ratio": 99,  # hold if forecast price > ratio * now
    "hold_shed_limit": 60,
    "opp_weight": 1.0,        # weight on opponent's pending supply
    "drop_value": 1000,        # carry value that triggers a shed drop trip
    "cash_reserve": 150,
    "min_rate": 3.0,          # min $/tile/day to bother planting
    "cap_lambda": 0.12,       # $ per $ per day charged on tied-up capital
    "cap_ref": 3000,          # lambda scales with cap_ref / money
    "op_reserve": 50,        # cash kept for hands + feed
    "travel_per_job": 2.5,
    "dist_offset": 0.5,
    "wheat_buffer": 4,
    "use_fert": True,
    "land_plant_cash": 1000,
    "lam_schedule": [1.0],
    "fill_reserve": 12,        # cash held back per still-unplanned tile
    "land_max_empty": 3,
    "land_steps": [147, 198, 250],        # e.g. [150, 218, 222]: buy quadrant k at/after this step
    "land_sched_reserve": 0,
    "hire_last_hour": 14,
    "unit_efficiency": 0.8,
    "use_tape": True,
    "place_value": 400,
    "save_ratio": 2.0,
    "animal_actions": 4.5,
    "ext_weight": 1.0,
    "dist_pow": 10.0,
    "crit_hour": 99,
    "crit_slack": -100,
    "crit_near": 1,
    "resticky": False,
    "preempt": True,
    "preempt_slack": 2,
    "enroute": "crit",
    "enroute_min": 0,
    "collect_base": 40,
    "collect_mult": 0.0,
    "job_min": 0,
    "early_straw_day": -1,
    "early_straw_mult": 1.0,
    "animal_eff": 0.8,
    "dispatch_place": True,
    "stick_dist": 0,
    "zone_mult": 1e30,
    "use_zones": False,
    "care_mult": 0.16,
    "val_pow": 1.0,
    "animal_reserve": 20,
    "hire_floor": [4, 3, 5, 5, 6, 6, 9, 8, 9, 10, 12],
    "poor_money": 1000,
    "poor_drop_value": 80,
    "reserve_tiles": False,
    "reserve_days": 2,
    "fert_mult_tomato": 1.0,
    "fert_mult_strawberry": 1.0,
    "lam_max": 3.0,
    "fert_buy_price": 0,
    "calib_w": 0.0,
    "calib_opp_w": 0.0,
    "early_harvest_day": 5,
    "sell_premium_first": True,
    "router": False,
    "evening_hour": 17,
    "evening_value": 300,
    "sell_sort": "slope",
    "npv_r0": 0.0,
    "sell_phase": -1,
    "care_crit": False,
    "feed_days": 1,
    "care_crit_price": 0,
    "npv_ref": 3000.0,
    "npv_max": 3.0,
    "npv_min": 0.1,
    "router_last": 14,
    "router_scale": 1.0,
    "router_sheep": 8.0,
    "router_cow": 4.0,
    "router_goose": 3.0,
    "windows": False,
    "imitate": False,
    "use_hour23": True,
    "harvest_crit": False,
    "cluster_crops": False,
    "mirror": False,
    "fert_onetime": False,
    "fert_internal_mult": 1.0,
    "fert_onetime_min": 5.0,
    "fert_keep_extra": 0,
    "drift_w": 0.0,
    "drift_days": 3,
    "drift_cap": 30.0,
    "carry_wheat_keep": 5,
    "tmpl_cats": ["STRAWBERRY"],
    "mirror_scale": 1.0,
    "clu_same_day": 3.0,
    "clu_same": 1.0,
    "clu_free": 0.3,
    "imit_animal_last": 12,
    "imit_goose_free": False,
    "imit_floor": -50.0,
    "win_wheat": [0, 29],
    "win_carrot": [10, 29],
    "win_tomato": [8, 19],
    "win_strawberry": [0, 7],
    "win_melon": [0, 6],
    "win_goose": [0, 10],
    "win_cow": [0, 10],
    "win_sheep": [0, 12],
    "calib_wheat": 1.0,
    "calib_carrot": 1.0,
    "calib_tomato": 1.0,
    "calib_strawberry": 1.0,
    "calib_melon": 1.0,
    "calib_egg": 1.0,
    "calib_milk": 1.0,
    "calib_wool": 1.0,
    "calib_fertilizer": 1.0,

    "tmpl": True,
    "tmpl_force": True,
    "tmpl_force_days": 7,
    "tmpl_floor_animal": -500.0,
    "tmpl_floor_crop": 0.0,
    "tmpl_mult": 1.0,
    "tmpl_add": 0.0,
    "tmpl_scale_animal": 1.0,
    "tmpl_scale_strawberry": 1.0,
    "tmpl_scale_melon": 1.0,
    "tmpl_scale_tomato": 1.0,
    "hire_floor_scale": 1.0,
    "fert_keep_days": 1,
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
    fr = n - int(n)
    if fr > 1e-9:
        tot += fr * price(item, inv + int(n))
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



# ----------------------------------------------------------------------------
# 3b. Market trajectory model: projected inventory of every product for every
#     remaining day, from both farms' production schedules and town demand.
#     A unit sold on day d is priced at the projected inventory of day d.
# ----------------------------------------------------------------------------
def price_f(item, inv):
    p = MARKET_PARAMS[item]
    I0 = p["I0"]
    if inv < I0:
        v = p["base"] + _AMP[item][0] * _shape(p["bf"], I0 - inv, p["T"])
    else:
        v = p["base"] - _AMP[item][1] * _shape(p["af"], inv - I0, p["T"])
    return max(1.0, v)


class Own:
    """Our own future sales per product/day, and the suffix sums of the price
    slope they are exposed to (the damage one more unit sold on day d does to
    every unit of ours sold on day >= d)."""
    def __init__(self, day):
        self.day = day
        self.O = {P: [0.0] * DAYS for P in PRODUCTS}
        self.S = {P: [0.0] * (DAYS + 1) for P in PRODUCTS}

    def add(self, events):
        for P, d, n in events:
            if n > 0 and d <= LAST_DAY:
                self.O[P][max(d, self.day)] += n

    def refresh(self, T, prods=None):
        for P in (prods or PRODUCTS):
            S = self.S[P]
            O = self.O[P]
            row = T[P]
            S[DAYS] = 0.0
            for d in range(DAYS - 1, self.day - 1, -1):
                o = O[d]
                if o:
                    inv = row[d]
                    S[d] = S[d + 1] + o * (price_f(P, inv) - price_f(P, inv + 1))
                else:
                    S[d] = S[d + 1]

    def ext(self, events):
        e = 0.0
        for P, d, n in events:
            if n > 0 and d <= LAST_DAY:
                e += n * self.S[P][max(d, self.day)]
        return e


def traj_new(market, shops, day):
    T = {}
    for P in PRODUCTS:
        inv0 = market["inventory"][P]
        row = [0.0] * DAYS
        for d in range(day, DAYS):
            row[d] = inv0 - consumption(P, day, d + 0.5, shops)
        T[P] = row
    return T



# ----------------------------------------------------------------------------
# Empirical calibration of the market projection: mean(real inventory -
# projected inventory) by product, decision-day bucket (d<9, d<17, later) and
# horizon h, measured over 22 simulated games vs top-team tapes and self-play.
# It captures the supply both farms will add later (future plantings,
# fertilizer doubling) and fertilizer that is used instead of sold.
# ----------------------------------------------------------------------------
RESID = {"CARROT":[[1,1,1,1,2,4,7,10,14,18,22,28,34,40,46,52,59,67,76,86,98,110,120,130,140,152,165,180,191,196],[-3,-6,-7,-4,2,11,19,28,37,48,59,72,86,99,108,115,123,132,144,158,170,176,176,176,176,176,176,176,176,176],[-10,-14,-14,-6,9,28,45,63,80,98,117,141,157,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167]],"EGG":[[1,1,0,0,0,0,0,1,3,5,8,12,17,22,28,33,40,46,52,59,66,73,79,84,89,93,100,106,111,112],[-1,-2,-2,-2,-1,0,2,4,7,10,13,16,19,23,28,32,36,40,47,61,73,81,81,81,81,81,81,81,81,81],[-2,-2,-1,0,1,2,4,6,8,11,13,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]],"FERTILIZER":[[-5,-6,-5,-3,0,5,12,18,24,30,35,39,41,42,41,39,37,34,31,28,26,33,53,79,104,121,134,138,139,138],[-25,-30,-39,-48,-60,-73,-87,-102,-118,-134,-151,-167,-183,-195,-203,-207,-210,-209,-204,-193,-182,-175,-175,-175,-175,-175,-175,-175,-175,-175],[-50,-63,-83,-105,-126,-147,-167,-186,-203,-219,-232,-244,-251,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255,-255]],"MELON":[[0,1,5,9,12,12,14,16,17,16,11,9,12,21,29,35,40,44,47,50,52,56,61,67,75,82,88,91,92,93],[8,13,14,11,8,8,6,4,2,0,-2,-2,-1,1,2,4,7,10,16,19,21,20,20,20,20,20,20,20,20,20],[0,0,0,0,-1,-2,-3,-4,-4,-5,-6,-6,-7,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6]],"MILK":[[2,2,2,3,3,4,5,7,11,18,27,36,46,56,65,73,81,89,97,105,113,125,144,166,189,206,217,223,226,228],[-8,-8,-8,-9,-9,-10,-11,-13,-14,-16,-17,-19,-20,-20,-19,-17,-14,-9,-5,-1,1,3,3,3,3,3,3,3,3,3],[-13,-13,-15,-17,-21,-24,-27,-31,-35,-37,-38,-37,-36,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34,-34]],"STRAWBERRY":[[2,2,2,2,3,3,4,5,6,9,15,23,34,50,70,95,124,158,194,231,268,303,334,360,383,402,418,431,439,443],[0,1,3,5,8,12,16,21,28,36,46,60,77,98,121,148,181,217,260,296,319,326,326,326,326,326,326,326,326,326],[-29,-23,-16,-9,-1,8,17,27,37,47,58,72,82,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87]],"TOMATO":[[1,1,1,2,2,3,4,5,6,7,8,9,11,13,16,19,22,25,29,34,38,43,48,53,58,63,66,70,72,74],[2,2,1,1,1,0,0,0,1,2,5,8,11,15,17,20,23,27,31,36,39,40,40,40,40,40,40,40,40,40],[-1,-1,1,2,3,3,3,3,3,5,9,15,20,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22]],"WHEAT":[[4,-2,-13,-21,-23,-15,-2,17,43,74,108,145,182,217,250,280,309,338,367,397,431,453,462,458,454,457,466,490,510,525],[-2,-6,-10,-2,24,66,112,155,197,238,280,323,369,415,461,504,547,590,633,673,698,707,707,707,707,707,707,707,707,707],[5,1,-2,6,34,78,129,180,231,282,333,387,424,444,444,444,444,444,444,444,444,444,444,444,444,444,444,444,444,444]],"WOOL":[[1,1,2,2,1,2,2,2,4,5,6,7,8,9,10,12,13,15,18,21,25,31,39,48,55,59,64,69,73,75],[-6,-8,-10,-13,-17,-21,-25,-29,-32,-34,-36,-38,-38,-37,-35,-32,-27,-21,-12,1,11,16,16,16,16,16,16,16,16,16],[-13,-15,-16,-19,-22,-26,-29,-31,-33,-35,-36,-39,-41,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43,-43]]}


# Opponent-supply calibration: actual cumulative sales of top-team tapes minus
# what our model projects from their visible farm, by product, decision-day
# bucket and horizon (their future plantings / purchases / fertilizer use).
RESID_OPP = {"CARROT":[[0,0,0,0,0,0,1,2,3,6,8,12,16,21,27,33,41,50,61,74,89,103,115,124,134,145,158,172,183,189],[0,-1,0,2,6,11,17,23,31,40,51,64,79,94,107,118,129,142,156,172,183,189,189,189,189,189,189,189,189,189],[1,2,7,17,30,45,59,75,90,107,123,140,151,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157]],"EGG":[[0,0,0,0,0,1,3,5,9,14,20,26,33,41,48,55,63,70,77,84,92,100,107,113,119,124,130,137,141,143],[-1,-1,-2,-2,-1,1,2,3,5,6,7,8,10,12,13,15,17,21,30,50,68,81,81,81,81,81,81,81,81,81],[0,1,2,4,5,6,7,8,9,11,12,14,15,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]],"FERTILIZER":[[4,5,6,9,13,18,24,29,34,38,43,46,49,50,51,52,52,52,51,49,47,51,61,75,88,99,109,114,115,113],[0,-2,-6,-11,-17,-24,-31,-39,-47,-56,-65,-75,-86,-95,-103,-108,-114,-117,-117,-108,-99,-92,-92,-92,-92,-92,-92,-92,-92,-92],[-4,-10,-19,-28,-38,-47,-57,-67,-77,-87,-97,-106,-112,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115,-115]],"MELON":[[0,1,4,7,9,10,10,10,11,11,9,6,6,8,10,11,12,12,12,12,12,13,14,15,17,18,20,22,24,24],[10,12,13,11,9,8,8,7,6,5,5,5,5,5,5,5,6,7,7,5,3,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],"MILK":[[1,1,2,2,2,2,2,3,4,6,9,11,12,13,13,13,12,12,11,11,10,12,17,26,36,46,55,60,62,61],[1,1,0,-1,-2,-4,-7,-9,-12,-15,-19,-22,-24,-27,-29,-31,-33,-34,-36,-37,-38,-39,-39,-39,-39,-39,-39,-39,-39,-39],[-5,-6,-8,-9,-12,-14,-16,-19,-22,-25,-27,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28,-28]],"STRAWBERRY":[[0,0,0,0,0,0,0,0,1,5,10,17,25,35,46,57,68,79,88,96,104,113,124,133,142,150,157,164,167,169],[-1,1,5,9,14,18,23,27,31,35,38,42,46,52,57,63,69,75,81,87,91,93,93,93,93,93,93,93,93,93],[-1,1,5,8,11,14,17,20,24,30,36,44,49,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53]],"TOMATO":[[0,0,0,0,0,0,0,0,0,0,0,1,2,5,8,11,15,19,24,29,35,40,44,47,50,55,60,66,71,73],[0,0,0,-1,-1,0,1,2,4,7,11,16,22,28,33,37,42,48,56,64,70,72,72,72,72,72,72,72,72,72],[-1,0,2,5,7,9,11,13,16,21,28,38,44,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47]],"WHEAT":[[12,16,21,31,46,67,90,116,144,174,204,233,260,285,307,330,354,379,406,435,469,500,529,552,574,595,619,649,672,686],[22,26,31,43,61,85,108,130,154,179,206,235,269,304,341,379,419,462,506,553,586,602,602,602,602,602,602,602,602,602],[14,16,19,33,56,88,123,157,192,227,264,303,332,346,346,346,346,346,346,346,346,346,346,346,346,346,346,346,346,346]],"WOOL":[[2,2,3,3,3,3,4,6,7,9,11,13,15,18,20,23,27,31,35,40,45,51,59,66,71,75,78,82,85,86],[1,0,-1,-2,-3,-3,-3,-3,-2,-1,1,3,5,7,9,11,13,14,18,24,30,34,34,34,34,34,34,34,34,34],[-2,-1,1,2,3,4,5,7,8,9,10,11,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14]]}


def traj_calibrate_opp(T, day):
    w = PARAMS["calib_opp_w"]
    if not w:
        return
    b = 0 if day < 9 else (1 if day < 17 else 2)
    for P, rows in RESID_OPP.items():
        row = rows[b]
        tp = T[P]
        for d in range(day, DAYS):
            tp[d] += w * row[d - day]


def traj_calibrate(T, day):
    w = PARAMS["calib_w"]
    if not w:
        return
    b = 0 if day < 9 else (1 if day < 17 else 2)
    for P, rows in RESID.items():
        wp = w * PARAMS["calib_" + P.lower()]
        if not wp:
            continue
        row = rows[b]
        tp = T[P]
        for d in range(day, DAYS):
            tp[d] += wp * row[d - day]



_PROJ = {}
_CUR = [0]


def _drift_correct(T, market, day, hour):
    """Online bias correction: compare today's market inventory with what our
    projections of the last few days predicted for today; the per-day error
    rate (unmodelled drain/supply of this world) is extrapolated forward."""
    ps = _PROJ.setdefault(_CUR[0], {})
    rec = ps.setdefault("rec", {})
    if ps.get("last_day", -1) > day:
        rec.clear()
    ps["last_day"] = day
    if day not in rec:
        rec[day] = {P: list(T[P]) for P in PRODUCTS}
        for d in list(rec):
            if d < day - PARAMS["drift_days"] - 1:
                del rec[d]
    for P in PRODUCTS:
        num = den = 0.0
        for h in range(1, PARAMS["drift_days"] + 1):
            r = rec.get(day - h)
            if r is None:
                continue
            num += market["inventory"][P] - r[P][day]
            den += h
        if den <= 0:
            continue
        rho = PARAMS["drift_w"] * num / den
        lim = PARAMS["drift_cap"]
        rho = max(-lim, min(lim, rho))
        row = T[P]
        for d in range(day, DAYS):
            row[d] += rho * (d - day)


def traj_add(T, P, d, n, day):
    if n == 0 or d > LAST_DAY:
        return
    row = T[P]
    for k in range(max(d, day), DAYS):
        row[k] += n


def traj_add_events(T, events, day, w=1.0):
    for P, d, n in events:
        traj_add(T, P, d, n * w, day)


def plant_events(t, day):
    cd = CROPS[t["crop"]]
    P = t["crop"]
    ev = []
    if not cd["ongoing"]:
        h = t["planted_day"] + cd["myd"]
        u = expected_one_time_yield(t, day)
        ev.append((P, min(LAST_DAY, max(day, h)), u))
    else:
        if t["yield_units"] > 0:
            ev.append((P, day, t["yield_units"]))
        m = fert_mult(t["crop"])
        for k in range(cd["max"]):
            d = t["planted_day"] + cd["fyd"] + k * cd["interval"]
            if day < d <= LAST_DAY:
                ev.append((P, d, m))
    return ev


def animal_events(animal, placed, day, held=0, fert_now=False, include_feed=True):
    a = ANIMALS[animal]
    P = a["product"]
    ev = []
    if held:
        ev.append((P, day, held))
    first = placed + a["fyd"]
    d = first
    while d <= LAST_DAY:
        if d > day:
            u = min(a["max_held"], a["fyd"]) if d == first else 1 + a["interval"]
            ev.append((P, d, u * PARAMS["animal_eff"]))
        d += a["interval"]
    if fert_now:
        ev.append(("FERTILIZER", day, 1))
    for d in range(max(day, placed) + 1, DAYS):
        ev.append(("FERTILIZER", d, 1))
    if include_feed:
        for d in range(max(day, placed), LAST_DAY):
            ev.append(("WHEAT", d, -1))
    return ev


def farm_events(farm, day, private=None):
    ev = []
    for row in farm["tiles"]:
        for t in row:
            if is_plant(t):
                ev.extend(plant_events(t, day))
            elif is_animal(t):
                ev.extend(animal_events(t["animal"], t["placed_day"], day, t["yield_units"],
                                        bool(t.get("fertilizer_available"))))
    if private:
        for it in PRODUCTS:
            n = private["shed"].get(it, 0)
            for inv in private["inventories"]:
                n += inv.get(it, 0)
            if n and it != "WHEAT":
                ev.append((it, day, n))
    return ev


_DISC = [0.0]


def events_value(T, events, day):
    """Revenue of adding these events on top of trajectory T (marginal)."""
    rev = 0.0
    added = {}
    for P, d, n in sorted(events, key=lambda e: e[1]):
        if d > LAST_DAY:
            continue
        d = max(d, day)
        base = T[P][d] + added.get(P, 0.0)
        disc = (1.0 + _DISC[0]) ** (-(d - day)) if _DISC[0] else 1.0
        if n > 0:
            rev += disc * sell_units_value(P, base, n)
        else:
            # buying (feed): pay the price at the lowered inventory
            for j in range(int(-n)):
                rev -= disc * price(P, base - j - 1)
        added[P] = added.get(P, 0.0) + n
    return rev


def crop_events(crop, day):
    plan = crop_plan(crop, day)
    if plan is None:
        return None, 0
    sched, occ = plan
    m = fert_mult(crop)
    return [(crop, d, u * m) for d, u in sched], occ


def fert_mult(crop):
    if not CROPS[crop]["ongoing"]:
        return 1.0
    return PARAMS["fert_mult_" + crop.lower()]


def value_crop_T(crop, day, T, labor, now_day, own=None):
    ev, occ = crop_events(crop, day)
    if ev is None:
        return None
    cd = CROPS[crop]
    rev = events_value(T, ev, now_day)
    if own is not None:
        rev -= PARAMS["ext_weight"] * own.ext(ev)
    actions = occ * (0.6 if not cd["ongoing"] else 0.5) + 2 + len(ev)
    if cd["ongoing"] and fert_mult(crop) > 1:
        nf = 2 * (fert_mult(crop) - 1)
        rev -= nf * price_f("FERTILIZER", T["FERTILIZER"][now_day])
        actions += 2 * nf
    val = rev - cd["seed"] - actions * labor
    occ = max(1, occ)
    return {"kind": "crop", "item": crop, "units": sum(e[2] for e in ev), "value": val,
            "rate": val / occ, "cost": cd["seed"], "occ": occ, "events": ev}


def value_animal_T(animal, day, T, labor, own=None):
    a = ANIMALS[animal]
    R = LAST_DAY - day
    if R <= a["fyd"]:
        return None
    ev = animal_events(animal, day, day)
    prod_ev = [e for e in ev if e[0] == a["product"]]
    rev_all = events_value(T, ev, day)
    if own is not None:
        rev_all -= PARAMS["ext_weight"] * own.ext(ev)
    units = sum(e[2] for e in prod_ev)
    actions = R * PARAMS["animal_actions"] + 4
    val = rev_all - a["cost"] - actions * labor
    # payback day: cumulative cash from the events
    cum, payback = 0.0, 99
    by_day = {}
    for P, d, n in ev:
        if n > 0:
            by_day[d] = by_day.get(d, 0.0) + n * price(P, T[P][d])
        else:
            by_day[d] = by_day.get(d, 0.0) + n * price(P, T[P][d])
    for k, d in enumerate(range(day, DAYS)):
        cum += by_day.get(d, 0.0) - PARAMS["animal_actions"] * labor
        if cum >= a["cost"]:
            payback = k + 1
            break
    return {"kind": "animal", "item": animal, "product": a["product"], "units": units,
            "fert": R, "value": val, "rate": val / max(1, R + 1),
            "cost": a["cost"], "occ": R + 1, "payback": payback, "events": ev}



# ----------------------------------------------------------------------------
# Top-team template (10 ladder games of DECEM / DSM / Yizhou / Boey agree):
# how many animals / strawberries / melons / tomatoes a farm holds by day.
# The planner gets a pull toward these counts; the market model still picks
# which animal and refuses uses with negative value.
# ----------------------------------------------------------------------------
TEMPLATE = {
    # averages of 10 top-team farms (DECEM, DSM, Yizhou, Boey), by day
    "ANIMAL":     [5, 5, 5, 6, 7, 8, 10, 13, 14, 17, 19, 21, 21, 21, 21, 22],
    "GOOSE":      [0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 5, 6, 6, 6, 6, 6],
    "STRAWBERRY": [0, 0, 0, 2, 4, 7, 11, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 30, 0],
    "MELON":      [4, 6, 7, 8, 9, 9, 9, 11, 11, 11, 0],
    "TOMATO":     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 5, 6, 6, 7, 7, 8, 10, 12, 13, 0],
}


_MIRROR = None


def _opp_counts(opp):
    c = {}
    for row in opp["tiles"]:
        for t in row:
            if is_animal(t):
                c["ANIMAL"] = c.get("ANIMAL", 0) + 1
                c[t["animal"]] = c.get(t["animal"], 0) + 1
            elif is_plant(t):
                c[t["crop"]] = c.get(t["crop"], 0) + 1
    return c


_ROUTE = {}


def _router_targets(shops, day):
    """Shop router (what top teams do after the first two shops open): the
    animal line follows the shops that buy its product."""
    first = shops[:2]
    wool = sum(1 for s in first if "WOOL" in SHOPS[s])
    milk = sum(1 for s in first if "MILK" in SHOPS[s])
    egg = sum(1 for s in first if "EGG" in SHOPS[s])
    r = PARAMS["router_scale"]
    ramp = min(1.0, max(0.0, (day - 5) / 4.0))     # build up over days 6..9
    return {"SHEEP": 3 + r * ramp * PARAMS["router_sheep"] * wool,
            "COW": 3 + r * ramp * PARAMS["router_cow"] * milk,
            "GOOSE": r * ramp * PARAMS["router_goose"] * (1 + egg)}


def _template_target(cat, day):
    if _MIRROR is not None:
        return _MIRROR.get(cat, 0) * PARAMS["mirror_scale"]
    if cat in _ROUTE:
        return _ROUTE[cat]
    row = TEMPLATE.get(cat)
    if not row:
        return 0
    v = row[min(day, len(row) - 1)]
    return v * PARAMS.get("tmpl_scale_" + cat.lower(), 1.0)


def _template_counts(me, private, plans):
    c = {}
    for row in me["tiles"]:
        for t in row:
            if is_animal(t):
                c["ANIMAL"] = c.get("ANIMAL", 0) + 1
                c[t["animal"]] = c.get(t["animal"], 0) + 1
            elif is_plant(t):
                c[t["crop"]] = c.get(t["crop"], 0) + 1
    for a in ANIMALS:
        n = private["shed"].get(a, 0) + sum(i.get(a, 0) for i in private["inventories"])
        c["ANIMAL"] = c.get("ANIMAL", 0) + n
        c[a] = c.get(a, 0) + n
    for pl in plans.values():
        c[pl["item"]] = c.get(pl["item"], 0) + 1
        if pl["kind"] == "animal":
            c["ANIMAL"] = c.get("ANIMAL", 0) + 1
    return c



# Planting windows copied from 10 top-team farms (what they plant by day):
# strawberries days 2-7, melons days 0-6, tomatoes 8-19, carrots 10+,
# animals until day 10, wheat any time.
def _in_window(item, day):
    if not PARAMS["windows"]:
        return True
    w = PARAMS["win_" + item.lower()]
    return w[0] <= day <= w[1]



# Crop mix of new plantings by day on top-team farms (wheat/carrot/tomato/
# strawberry shares), days 8..27.
MIX = {8: (86, 0, 2, 7), 9: (77, 0, 15, 8), 10: (72, 8, 13, 6), 11: (64, 23, 11, 2),
       12: (67, 10, 6, 17), 13: (70, 14, 3, 13), 14: (73, 23, 3, 1), 15: (42, 30, 6, 22),
       16: (54, 35, 11, 0), 17: (69, 21, 10, 0), 18: (49, 21, 29, 1), 19: (48, 40, 12, 0)}
MIX_LATE = (55, 45, 0, 0)
MIX_CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY")


def _imitate(tiles, plans, free_empty, free_struct, cnt, T, own, day, plant_day, labor, budget):
    """Top-team macro policy: animals and early strawberries/melons to the
    template counts, then fill tiles with the day's crop mix. The market model
    only picks the animal type."""
    def assign(o, pos):
        nonlocal budget
        best = dict(o)
        best["day"] = day
        best["adj"] = 100.0
        plans[pos] = best
        budget -= o["cost"]
        traj_add_events(T, o["events"], day)
        own.add(o["events"])
        own.refresh(T, {e[0] for e in o["events"]})
        cnt[o["item"]] = cnt.get(o["item"], 0) + 1
        if o["kind"] == "animal":
            cnt["ANIMAL"] = cnt.get("ANIMAL", 0) + 1

    # 1) animals
    guard = 0
    while guard < 40 and day <= PARAMS["imit_animal_last"]:
        guard += 1
        want_goose = cnt.get("GOOSE", 0) < _template_target("GOOSE", day)
        if cnt.get("ANIMAL", 0) >= _template_target("ANIMAL", day) and not want_goose:
            break
        if want_goose:
            kinds = ["GOOSE"]
        else:
            vals = []
            for a in ANIMALS:
                if a == "GOOSE" and not PARAMS["imit_goose_free"]:
                    continue
                v = value_animal_T(a, day, T, labor, own)
                if v:
                    vals.append((v["value"], a))
            vals.sort(reverse=True)
            kinds = [a for _, a in vals]
        placed = False
        for a in kinds:
            v = value_animal_T(a, day, T, labor, own)
            if not v or v["cost"] > budget:
                continue
            skind = ANIMALS[a]["structure"]
            pos = next((p for p in free_struct if tiles[p[1]][p[0]]["kind"] == skind), None)
            if pos is not None:
                free_struct.remove(pos)
            elif free_empty:
                pos = free_empty.pop(0)
            else:
                break
            assign(v, pos)
            placed = True
            break
        if not placed:
            break
    if day > LAST_DAY - 2:
        return
    # 2) premium crops to the template (early game)
    for c in ("STRAWBERRY", "MELON"):
        while free_empty and cnt.get(c, 0) < _template_target(c, day):
            v = value_crop_T(c, plant_day, T, labor, day, own)
            if not v or v["cost"] > budget or v["value"] < PARAMS["imit_floor"]:
                break
            assign(v, free_empty.pop())
    # 3) fill with the day's mix
    mix = MIX.get(day, MIX_LATE if day > 19 else (100, 0, 0, 0))
    placed_today = {c: 0 for c in MIX_CROPS}
    while free_empty:
        n = sum(placed_today.values()) + 1
        # crop whose share is most below its target share
        best_c, best_gap = None, -1e9
        for c, sh in zip(MIX_CROPS, mix):
            if sh <= 0:
                continue
            gap = sh / 100.0 - placed_today[c] / n
            if gap > best_gap:
                best_c, best_gap = c, gap
        v = value_crop_T(best_c, plant_day, T, labor, day, own)
        if not v or v["value"] < PARAMS["imit_floor"]:
            v = value_crop_T("WHEAT", plant_day, T, labor, day, own)
            if not v:
                break
        if v["cost"] > budget:
            break
        placed_today[v["item"]] = placed_today.get(v["item"], 0) + 1
        assign(v, free_empty.pop())



def _cluster_pick(free_empty, tiles, plans, crop, day):
    """Tile for a new crop: next to the same crop planted/planned today (same
    watering and harvest days = contiguous sweeps), else next to the same crop,
    else far from the shed (animals keep the ring around the shed)."""
    best, best_s = None, -1e9
    for p in free_empty:
        x, y = p
        sc = 0.15 * min(manhattan(p, s_) for s_ in SHED_TILES)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < BOARD and 0 <= ny < BOARD):
                continue
            pl = plans.get((nx, ny))
            if pl is not None and pl["kind"] == "crop" and pl["item"] == crop:
                sc += PARAMS["clu_same_day"]
                continue
            t = tiles[ny][nx]
            if is_plant(t) and t["crop"] == crop:
                sc += PARAMS["clu_same_day"] if t["planted_day"] == day else PARAMS["clu_same"]
            elif t is None or is_weed(t):
                sc += PARAMS["clu_free"]
        if sc > best_s:
            best, best_s = p, sc
    return best


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


def _eod(day):
    """Last hour whose actions still count today (23; 22 on the final day)."""
    if not PARAMS["use_hour23"]:
        return LAST_ACT_HOUR
    return LAST_ACT_HOUR if day >= LAST_DAY else TPD - 1


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

    T = traj_new(market, shops, day)
    own = Own(day)
    my_ev = farm_events(me, day, private)
    traj_add_events(T, my_ev, day)
    own.add(my_ev)
    traj_add_events(T, farm_events(opp, day), day, PARAMS["opp_weight"])
    for pl in plans.values():
        if pl.get("events"):
            traj_add_events(T, pl["events"], day)
            own.add(pl["events"])
    if PARAMS["drift_w"]:
        _drift_correct(T, market, day, hour)
    traj_calibrate(T, day)
    traj_calibrate_opp(T, day)
    own.refresh(T)

    # committed spend for existing unexecuted plans
    for pl in plans.values():
        budget -= pl["cost"]

    free_empty, free_struct = [], []
    for y in range(BOARD):
        for x in range(BOARD):
            t = tiles[y][x]
            if t == "LOCKED" or (x, y) in plans:
                continue
            if t is None or is_weed(t):
                free_empty.append((x, y))
            elif is_empty_struct(t):
                free_struct.append((x, y))
    sd = lambda p: min(manhattan(p, s_) for s_ in SHED_TILES)
    free_empty.sort(key=sd)          # animals take the near end, crops the far end
    free_struct.sort(key=sd)

    late = hour > _eod(day) - 2
    labor = PARAMS["labor_per_action"]
    lam = lam_mult * PARAMS["cap_lambda"] * min(PARAMS["lam_max"], max(0.1, PARAMS["cap_ref"] / max(1.0, float(me["money"]))))
    if PARAMS["npv_r0"] > 0:
        _DISC[0] = PARAMS["npv_r0"] * min(PARAMS["npv_max"], max(PARAMS["npv_min"], PARAMS["npv_ref"] / max(1.0, float(me["money"]))))
        lam = 0.0
    else:
        _DISC[0] = 0.0
    plant_day = day if not late else day + 1
    minr = PARAMS["min_rate"]
    save_ratio = PARAMS["save_ratio"]
    starved = 0
    blocked = set()
    reserve_left = None
    tcnt = _template_counts(me, private, plans) if PARAMS["tmpl"] else None
    global _MIRROR
    _MIRROR = _opp_counts(opp) if PARAMS["mirror"] else None
    _ROUTE.clear()
    if PARAMS["router"] and len(shops) >= 2 and day <= PARAMS["router_last"]:
        _ROUTE.update(_router_targets(shops, day))
    if PARAMS["imitate"]:
        _imitate(tiles, plans, free_empty, free_struct, _template_counts(me, private, plans),
                 T, own, day, plant_day, labor, budget)
        return plans, 0
    guard = 0
    while (free_empty or free_struct) and guard < 200:
        guard += 1
        opts = []
        if free_empty or free_struct:
            for a in ANIMALS:
                if a in blocked or not _in_window(a, day):
                    continue
                has_struct = any(tiles[p[1]][p[0]]["kind"] == ANIMALS[a]["structure"] for p in free_struct)
                if not has_struct and not free_empty:
                    continue
                v = value_animal_T(a, day, T, labor, own)
                if v:
                    v["adj"] = adjusted_rate(v, lam)
                    v["struct"] = has_struct
                    opts.append(v)
        if free_empty:
            for c in CROPS:
                if c in blocked or not _in_window(c, plant_day):
                    continue
                v = value_crop_T(c, plant_day, T, labor, day, own)
                if v:
                    v["adj"] = adjusted_rate(v, lam)
                    if c == "STRAWBERRY" and day <= PARAMS["early_straw_day"] and v["value"] > 0:
                        v["adj"] = max(v["adj"], v["rate"]) * PARAMS["early_straw_mult"]
                    opts.append(v)
        if tcnt is not None:
            for o in opts:
                cat = "ANIMAL" if o["kind"] == "animal" else o["item"]
                if o["value"] > 0 and tcnt.get(cat, 0) < _template_target(cat, day):
                    o["adj"] = o["adj"] * PARAMS["tmpl_mult"] + PARAMS["tmpl_add"]
        forced = None
        if tcnt is not None and PARAMS["tmpl_force"] and day <= PARAMS["tmpl_force_days"]:
            for cat in (("SHEEP", "COW", "GOOSE", "STRAWBERRY", "MELON", "TOMATO", "CARROT", "WHEAT")
                        if _MIRROR is not None else PARAMS["tmpl_cats"]):
                if tcnt.get(cat, 0) >= _template_target(cat, day):
                    continue
                cands = [o for o in opts
                         if (o["kind"] == "animal" if cat == "ANIMAL" else o["item"] == cat)
                         and o["value"] > PARAMS["tmpl_floor_" + ("animal" if cat in ("ANIMAL", "GOOSE", "SHEEP", "COW") else "crop")]
                         and o["cost"] <= budget]
                if cands:
                    forced = max(cands, key=lambda o: o["value"])
                    break
        if forced is not None:
            forced["adj"] = max(forced["adj"], 1e6)
            opts = [forced]
        opts = [o for o in opts if o["adj"] >= minr]
        if not opts:
            break
        opts.sort(key=lambda o: -o["adj"])
        best_aff = None
        for o in opts:
            if o["cost"] <= budget:
                best_aff = o
                break
        top = opts[0]
        if top["cost"] > budget:
            starved += 1
            if (top["kind"] == "animal" and best_aff is not None and PARAMS["reserve_tiles"]
                    and top["adj"] >= save_ratio * best_aff["adj"]):
                # keep near-shed tiles free for the animals our coming income will buy
                if reserve_left is None:
                    inc = 0.0
                    for P in PRODUCTS:
                        for d in range(day, min(DAYS, day + PARAMS["reserve_days"])):
                            if own.O[P][d]:
                                inc += own.O[P][d] * price(P, T[P][d])
                    reserve_left = int(max(0.0, inc + budget) // top["cost"])
                st_kind = ANIMALS[top["item"]]["structure"]
                spos = next((p for p in free_struct if tiles[p[1]][p[0]]["kind"] == st_kind), None)
                if reserve_left > 0 and (spos is not None or free_empty):
                    reserve_left -= 1
                    if spos is not None:
                        free_struct.remove(spos)
                    else:
                        free_empty.pop(0)
                    traj_add_events(T, top["events"], day)
                    continue
                blocked.add(top["item"])
                continue
            if best_aff is None or top["adj"] >= save_ratio * best_aff["adj"]:
                break                       # save cash for the much better use
            blocked.add(top["item"])
            continue
        o = top
        if o["kind"] == "animal":
            st_kind = ANIMALS[o["item"]]["structure"]
            pos = None
            for p in free_struct:
                if tiles[p[1]][p[0]]["kind"] == st_kind:
                    pos = p
                    break
            if pos is not None:
                free_struct.remove(pos)
            else:
                pos = free_empty.pop(0)
        else:
            if PARAMS["cluster_crops"]:
                pos = _cluster_pick(free_empty, tiles, plans, o["item"], day)
                free_empty.remove(pos)
            else:
                pos = free_empty.pop()
        best = dict(o)
        best["day"] = day
        plans[pos] = best
        if tcnt is not None:
            tcnt[o["item"]] = tcnt.get(o["item"], 0) + 1
            if o["kind"] == "animal":
                tcnt["ANIMAL"] = tcnt.get("ANIMAL", 0) + 1
        budget -= best["cost"]
        traj_add_events(T, best["events"], day)
        own.add(best["events"])
        own.refresh(T, {e[0] for e in best["events"]})
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
            crit = False
            if is_plant(t):
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                pr = prices[t["crop"]]
                if not cd["ongoing"]:
                    inc = one_time_remaining_increments(t, day)
                    ws = (cd["myd"] + 1) // 2
                    in_window = ws <= age <= cd["myd"]
                    if (PARAMS["fert_onetime"] and in_window and not t["watered_today"] and not last_day
                            and t.get("fertilized_until_day", -1) < day):
                        t2 = dict(t)
                        t2["fertilized_until_day"] = day + 2
                        g = one_time_remaining_increments(t2, day) - inc
                        if g > 0:
                            fv = g * pr - PARAMS["fert_internal_mult"] * prices["FERTILIZER"]
                            if fv > PARAMS["fert_onetime_min"]:
                                acts.append((["FERTILIZE"], fv, "FERTILIZER"))
                    if not t["watered_today"]:
                        wv = 0
                        if in_window and inc > 0:
                            wv += pr * (2 if t.get("fertilized_until_day", -1) >= day else 1)
                        if t["consecutive_unwatered"] >= 1 and not last_day:
                            # dies tonight if not watered: worth the whole plant
                            wv += 20 + pr * expected_one_time_yield(t, day)
                            crit = True
                        if wv > 0:
                            acts.append((["WATER"], wv, None))
                    if age >= cd["fyd"] and (inc == 0 or last_day or age >= cd["myd"]):
                        acts.append((["HARVEST"], 40 + pr * t["yield_units"] * 0.5, None))
                        if PARAMS["harvest_crit"] and age >= cd["myd"]:
                            crit = True     # decays from tomorrow 00:00 (or already decaying)
                    elif age >= cd["fyd"] and t["watered_today"] and inc == 0:
                        acts.append((["HARVEST"], 40, None))
                    elif (day <= PARAMS["early_harvest_day"] and age >= cd["myd"] - 1 and t["watered_today"]
                          and t["crop"] == "WHEAT"):
                        acts.append((["HARVEST"], 60, None))     # free the tile a day early (top teams do)
                else:
                    done = productions_done(t["crop"], t["planted_day"], day)
                    finished = done >= cd["max"]
                    if not t["watered_today"] and not finished and not last_day:
                        fert_active = t.get("fertilized_until_day", -1) >= day
                        if t["consecutive_unwatered"] >= 1 or fert_active:
                            left = t["yield_units"] + (cd["max"] - done)
                            crit = crit or t["consecutive_unwatered"] >= 1
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
                    crit = t["consecutive_unfed"] >= 1
                if not t["cared_today"] and day < LAST_DAY - 1:
                    acts.append((["CARE"], PARAMS["care_mult"] * prices[a["product"]], None))
                    if PARAMS["care_crit"] and prices[a["product"]] >= PARAMS["care_crit_price"]:
                        crit = True
                if t.get("fertilizer_available"):
                    acts.append((["COLLECT_FERTILIZER"], PARAMS["collect_base"] + PARAMS["collect_mult"] * prices["FERTILIZER"], None))
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
                acts.append((["PLACE", a], PARAMS["place_value"], a))
            elif (t is None or is_weed(t) or is_empty_struct(t)) and (x, y) in plans:
                pl = plans[(x, y)]
                if is_weed(t) or (pl["kind"] == "crop" and is_empty_struct(t)):
                    acts.append((["DIG"], 30, None))
                elif pl["kind"] == "crop":
                    if hour <= _eod(day) - 1 and seeds_avail.get(pl["item"], 0) > 0:
                        acts.append((["PLANT", pl["item"]], 50 + max(0, pl.get("adj", 0)), None))
                elif pl["kind"] == "animal":
                    struct = ANIMALS[pl["item"]]["structure"]
                    if t is None:
                        acts.append((["BUILD_" + struct], 45, None))
                    elif is_empty_struct(t) and t["kind"] == struct:
                        acts.append((["PLACE", pl["item"]], PARAMS["place_value"], pl["item"]))
            if PARAMS["job_min"] > 0:
                acts = [a_ for a_ in acts if a_[1] >= PARAMS["job_min"]]
            if acts:
                jobs[(x, y)] = {"acts": acts, "crit": crit}
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
        if it == "WHEAT":
            n -= PARAMS["carry_wheat_keep"]      # a few units are feed
        if it in prices and n > 0:
            v += prices[it] * n
    return v



# ----------------------------------------------------------------------------
# Opening tape: DECEM (ladder #1) plays an identical day 0 in every game.
# Each step is guarded: the tape is used only while our farm matches the
# recorded farm (tile signature + unit positions); otherwise we hand over.
# ----------------------------------------------------------------------------
OPENING_TAPE = [{"a":{"farmer":["PASS"],"hands":[],"market":[["BUY_ANIMAL","COW",1],["BUY_ANIMAL","SHEEP",1],["BUY_PRODUCT","WHEAT",5]]},"sig":".....#####.....#####.....#####.....#####.....#######################################################","pos":[[4,4]]},{"a":{"farmer":["PICKUP","COW",1],"hands":[],"market":[["SELL","WHEAT",1],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_ANIMAL","COW",1],["BUY_ANIMAL","SHEEP",1]]},"sig":".....#####.....#####.....#####.....#####.....#######################################################","pos":[[4,4]]},{"a":{"farmer":["BUILD_PASTURE"],"hands":[["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","COW",1],["PICKUP","SHEEP",1]],"market":[["SELL","WHEAT",1]]},"sig":".....#####.....#####.....#####.....#####.....#######################################################","pos":[[4,4],[5,4],[4,5],[5,5],[4,4]]},{"a":{"farmer":["PLACE","COW",1],"hands":[["NORTH"],["NORTH"],["NORTH"],["WEST"]],"market":[["SELL","WHEAT",1],["BUY_PRODUCT","WHEAT",1]]},"sig":".....#####.....#####.....#####.....#####....P#######################################################","pos":[[4,4],[5,4],[4,5],[5,5],[4,4]]},{"a":{"farmer":["PICKUP","WHEAT",3],"hands":[["WEST"],["WEST"],["NORTH"],["WEST"]],"market":[["BUY_ANIMAL","SHEEP",1]]},"sig":".....#####.....#####.....#####.....#####....C#######################################################","pos":[[4,4],[5,3],[4,4],[5,4],[3,4]]},{"a":{"farmer":["PICKUP","SHEEP",1],"hands":[["BUILD_PASTURE"],["BUILD_PASTURE"],["NORTH"],["WEST"]],"market":[]},"sig":".....#####.....#####.....#####.....#####....C#######################################################","pos":[[4,4],[4,3],[3,4],[5,3],[2,4]]},{"a":{"farmer":["NORTH"],"hands":[["PLACE","SHEEP",1],["PLACE","SHEEP",1],["PASS"],["NORTH"]],"market":[["BUY_PRODUCT","WHEAT",2]]},"sig":".....#####.....#####.....#####....P#####...PC#######################################################","pos":[[4,4],[4,3],[3,4],[5,2],[1,4]]},{"a":{"farmer":["WEST"],"hands":[["CARE"],["CARE"],["WEST"],["PASS"]],"market":[["BUY_SEED","MELON",2]]},"sig":".....#####.....#####.....#####....S#####...SC#######################################################","pos":[[4,3],[4,3],[3,4],[5,2],[1,3]]},{"a":{"farmer":["BUILD_PASTURE"],"hands":[["SOUTH"],["WEST"],["BUILD_PASTURE"],["PLANT","MELON"]],"market":[["BUY_SEED","MELON",2]]},"sig":".....#####.....#####.....#####....S#####...SC#######################################################","pos":[[3,3],[4,3],[3,4],[4,2],[1,3]]},{"a":{"farmer":["PLACE","SHEEP",1],"hands":[["PICKUP","WHEAT",1],["WEST"],["PLACE","COW",1],["WATER"]],"market":[]},"sig":".....#####.....#####....P#####.m.PS#####...SC#######################################################","pos":[[3,3],[4,4],[2,4],[4,2],[1,3]]},{"a":{"farmer":["FEED"],"hands":[["NORTH"],["PLANT","MELON"],["WEST"],["NORTH"]],"market":[]},"sig":".....#####.....#####....C#####.m.SS#####...SC#######################################################","pos":[[3,3],[4,4],[1,4],[4,2],[1,3]]},{"a":{"farmer":["CARE"],"hands":[["FEED"],["WATER"],["PLANT","MELON"],["PLANT","MELON"]],"market":[["BUY_SEED","MELON",2]]},"sig":".....#####.....#####....C#####.m.SS#####.m.SC#######################################################","pos":[[3,3],[4,3],[1,4],[3,2],[1,2]]},{"a":{"farmer":["SOUTH"],"hands":[["NORTH"],["WEST"],["WATER"],["WATER"]],"market":[["BUY_SEED","WHEAT",3]]},"sig":".....#####.....#####.m.mC#####.m.SS#####.m.SC#######################################################","pos":[[3,3],[4,3],[1,4],[3,2],[1,2]]},{"a":{"farmer":["FEED"],"hands":[["NORTH"],["PLANT","MELON"],["NORTH"],["NORTH"]],"market":[["BUY_SEED","WHEAT",1]]},"sig":".....#####.....#####.m.mC#####.m.SS#####.m.SC#######################################################","pos":[[3,4],[4,2],[0,4],[3,2],[1,2]]},{"a":{"farmer":["EAST"],"hands":[["PLANT","WHEAT"],["WATER"],["PLANT","WHEAT"],["PLANT","WHEAT"]],"market":[["BUY_SEED","WHEAT",1]]},"sig":".....#####.....#####.m.mC#####.m.SS#####mm.SC#######################################################","pos":[[3,4],[4,1],[0,4],[3,1],[1,1]]},{"a":{"farmer":["DROP"],"hands":[["WATER"],["NORTH"],["WATER"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},"sig":".....#####.w.ww#####.m.mC#####.m.SS#####mm.SC#######################################################","pos":[[4,4],[4,1],[0,4],[3,1],[1,1]]},{"a":{"farmer":["WEST"],"hands":[["NORTH"],["PLANT","MELON"],["WEST"],["NORTH"]],"market":[["SELL","WHEAT",2]]},"sig":".....#####.w.ww#####.m.mC#####.m.SS#####mm.SC#######################################################","pos":[[4,4],[4,1],[0,3],[3,1],[1,1]]},{"a":{"farmer":["WEST"],"hands":[["PLANT","WHEAT"],["WATER"],["PLANT","WHEAT"],["PLANT","WHEAT"]],"market":[["BUY_SEED","WHEAT",3]]},"sig":".....#####.w.ww#####.m.mC#####mm.SS#####mm.SC#######################################################","pos":[[3,4],[4,0],[0,3],[2,1],[1,0]]},{"a":{"farmer":["PLANT","WHEAT"],"hands":[["WATER"],["NORTH"],["WATER"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},"sig":".w..w#####.wwww#####.m.mC#####mm.SS#####mm.SC#######################################################","pos":[[2,4],[4,0],[0,3],[2,1],[1,0]]},{"a":{"farmer":["WATER"],"hands":[["WEST"],["PLANT","WHEAT"],["SOUTH"],["WEST"]],"market":[["BUY_SEED","WHEAT",1]]},"sig":".w..w#####.wwww#####.m.mC#####mm.SS#####mmwSC#######################################################","pos":[[2,4],[4,0],[0,2],[2,1],[1,0]]},{"a":{"farmer":["NORTH"],"hands":[["PLANT","WHEAT"],["WATER"],["PLANT","WHEAT"],["PLANT","WHEAT"]],"market":[["BUY_SEED","WHEAT",1]]},"sig":".w..w#####.wwww#####wm.mC#####mm.SS#####mmwSC#######################################################","pos":[[2,4],[3,0],[0,2],[2,2],[0,0]]},{"a":{"farmer":["PLANT","WHEAT"],"hands":[["WATER"],["NORTH"],["WATER"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},"sig":"ww.ww#####.wwww#####wmwmC#####mm.SS#####mmwSC#######################################################","pos":[[2,3],[3,0],[0,2],[2,2],[0,0]]},{"a":{"farmer":["WATER"],"hands":[["WEST"],["PLANT","WHEAT"],["SOUTH"],["SOUTH"]],"market":[]},"sig":"ww.ww#####.wwww#####wmwmC#####mmwSS#####mmwSC#######################################################","pos":[[2,3],[3,0],[0,1],[2,2],[0,0]]},{"a":{"farmer":["SOUTH"],"hands":[["SOUTH"],["WATER"],["SOUTH"],["SOUTH"]],"market":[]},"sig":"ww.ww#####wwwww#####wmwmC#####mmwSS#####mmwSC#######################################################","pos":[[2,3],[2,0],[0,1],[2,3],[0,1]]}]


def _tile_sig(tiles):
    out = []
    for row in tiles:
        for x in row:
            if x is None:
                out.append('.')
            elif x == "LOCKED":
                out.append('#')
            elif x.get("kind") == "PLANT":
                out.append(x["crop"][0].lower())
            elif x.get("animal"):
                out.append(x["animal"][0])
            else:
                out.append(x["kind"][0])
    return ''.join(out)


def _tape_action(obs, st, step):
    if not PARAMS.get("use_tape", True) or not st.get("tape_ok", True):
        return None
    if step >= len(OPENING_TAPE):
        return None
    rec = OPENING_TAPE[step]
    me = obs["farms"][obs["player"]]
    pos = [list(me["farmer"])] + [list(h) for h in me["hands"]]
    if _tile_sig(me["tiles"]) != rec["sig"] or pos != rec["pos"]:
        st["tape_ok"] = False
        return None
    a = rec["a"]
    return {"farmer": list(a["farmer"]), "hands": [list(h) for h in a["hands"]],
            "market": [list(m) for m in a["market"]]}

# ----------------------------------------------------------------------------
# main agent
# ----------------------------------------------------------------------------
def _agent(obs):
    player = obs["player"]
    _CUR[0] = player
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
    tape = _tape_action(obs, st, step)
    if tape is not None:
        return tape

    me = dict(me)
    me["tiles"] = [list(r) for r in me["tiles"]]   # never mutate the observation
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
    budget = money - PARAMS["op_reserve"] - (n_animals * PARAMS["animal_reserve"] if day < LAST_DAY else 0)
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
        if buy > 0 and hour <= _eod(day) - 2:
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
        dv = PARAMS["drop_value"]
        if money < PARAMS["poor_money"]:
            dv = PARAMS["poor_drop_value"]
        if (not last_day and hour >= PARAMS["evening_hour"] and cv >= PARAMS["evening_value"]
                and hour + shed_dist(pos) <= TPD - 1):
            return True        # sell tonight, before the opponent's morning dump
        return must or cv >= dv

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
        if (PARAMS["resticky"] and tpos != "SHED" and isinstance(tpos, tuple)
                and manhattan(units[ui], tpos) > PARAMS["stick_dist"]):
            del targets[ui]       # re-decide far targets every turn
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

    # 1b) urgent critical jobs (a plant dies / an animal escapes tonight) preempt
    #     the nearest unit when the deadline is close
    if PARAMS["preempt"]:
        deadline = LAST_ACT_HOUR if last_day else TPD - 1
        owner = {tp: u for u, tp in targets.items() if isinstance(tp, tuple)}
        urgent = []
        for tpos, job in jobs.items():
            if not job.get("crit"):
                continue
            n_acts = len(job["acts"])
            ds = sorted((manhattan(units[u], tpos), u) for u in range(len(units)))
            dmin = ds[0][0]
            slack = deadline - (hour + dmin + n_acts)
            if slack > PARAMS["preempt_slack"]:
                continue
            cur = owner.get(tpos)
            if cur is not None and manhattan(units[cur], tpos) <= dmin + 1:
                continue
            urgent.append((slack, tpos, ds))
        urgent.sort()
        for slack, tpos, ds in urgent:
            for dist, u in ds:
                if dist > ds[0][0] + 2:
                    break
                tu = targets.get(u)
                if tu == "SHED" and not last_day:
                    continue
                if isinstance(tu, tuple) and (jobs.get(tu, {}).get("crit") or tu == units[u]):
                    continue
                v, pick = job_value_for(jobs[tpos], invs[u], shed)
                if v <= 0 or pick:
                    continue
                cur = owner.get(tpos)
                if cur is not None:
                    targets.pop(cur, None)
                    taken.discard(tpos)
                if isinstance(tu, tuple):
                    taken.discard(tu)
                    owner.pop(tu, None)
                targets[u] = tpos
                owner[tpos] = u
                taken.add(tpos)
                break

    # 1c) animals waiting in the shed: a unit at the shed carries each one to
    #     its empty coop/pasture (a placed animal earns every day)
    if PARAMS["dispatch_place"]:
        owner = {tp: u for u, tp in targets.items() if isinstance(tp, tuple)}
        avail = {a: shed.get(a, 0) for a in ANIMALS}
        for u in range(len(units)):
            for a in ANIMALS:
                avail[a] -= invs[u].get(a, 0) if False else 0
        for tpos, job in jobs.items():
            if tpos in taken:
                continue
            pl = next((a_ for a_ in job["acts"] if a_[0][0] == "PLACE"), None)
            if pl is None or avail.get(pl[2], 0) <= 0:
                continue
            best_u, best_d = None, 99
            for u in range(len(units)):
                if shed_dist(units[u]) > 1:
                    continue
                tu = targets.get(u)
                if tu == "SHED" or (isinstance(tu, tuple) and (jobs.get(tu, {}).get("crit") or tu == units[u])):
                    continue
                if isinstance(tu, tuple) and any(a_[0][0] == "PLACE" for a_ in jobs.get(tu, {"acts": []})["acts"]):
                    continue
                dd = manhattan(units[u], tpos)
                if dd < best_d:
                    best_u, best_d = u, dd
            if best_u is None:
                continue
            tu = targets.get(best_u)
            if isinstance(tu, tuple):
                taken.discard(tu)
            targets[best_u] = tpos
            taken.add(tpos)
            avail[pl[2]] -= 1

    # 2) assign free units greedily (best score first)
    free_units = [ui for ui in range(len(units)) if ui not in targets]
    zone_of = _zones(st, jobs, tiles, len(units), day) if PARAMS["use_zones"] else None
    crit_dmin = {}
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
            sc = (v ** PARAMS["val_pow"]) / ((d + PARAMS["dist_offset"]) ** PARAMS["dist_pow"] + n_acts)
            if job.get("crit"):
                dmin = crit_dmin.get(tpos)
                if dmin is None:
                    dmin = min(manhattan(u, tpos) for u in units)
                    crit_dmin[tpos] = dmin
                deadline = LAST_ACT_HOUR if last_day else TPD - 1
                if (hour >= PARAMS["crit_hour"] or
                        (hour + dmin + n_acts >= deadline - PARAMS["crit_slack"] and d <= dmin + PARAMS["crit_near"])):
                    sc *= 1e12
            if zone_of is not None and zone_of(tpos) == ui:
                sc *= PARAMS["zone_mult"]
            pairs.append((sc, ui, tpos))
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
    done_here = set()
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
            act = ["DROP"] if pos == s_ else (_en_route(pos, jobs, inv, done_here) or move_toward(pos, s_))
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
                    act = _en_route(pos, jobs, inv, done_here) or move_toward(pos, tpos)
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
    wheat_keep = (n_animals + sum(want_animals.values())) * min(PARAMS["feed_days"], LAST_DAY - day) + PARAMS["wheat_buffer"] + 2 if day < LAST_DAY else 0
    wheat_keep = min(wheat_keep, 60)
    ext = farm_pending(opp, day)
    fert_keep = 0
    if PARAMS["use_fert"] and not last_day:
        for row in tiles:
            for t in row:
                if is_plant(t) and CROPS[t["crop"]]["ongoing"]:
                    eves = production_eves(t["crop"], t["planted_day"])
                    if any(day <= D <= day + PARAMS['fert_keep_days'] for D in eves):
                        fert_keep += 1
        if PARAMS["fert_onetime"]:
            fert_keep += PARAMS["fert_keep_extra"] + sum(
                1 for j in jobs.values() for a_ in j["acts"]
                if a_[0][0] == "FERTILIZE" and is_plant(tiles[0][0]) is not None)
        fert_keep = max(0, fert_keep - sum(i.get("FERTILIZER", 0) for i in invs))
    fert_orders = []
    if (PARAMS["fert_buy_price"] > 0 and fert_keep > 0 and hour <= PARAMS["hire_last_hour"]
            and prices["FERTILIZER"] <= PARAMS["fert_buy_price"]):
        short = fert_keep - shed.get("FERTILIZER", 0) - incoming.get("FERTILIZER", 0)
        if short > 0 and money > 200:
            fert_orders.append(["BUY_PRODUCT", "FERTILIZER", int(short)])
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
    if PARAMS["sell_phase"] >= 0 and not last_day and step % 4 != PARAMS["sell_phase"]:
        keep = [o for o in sell_orders if o[1] in ("WHEAT", "FERTILIZER") or money < 300]
        sell_orders = keep
    if PARAMS["sell_premium_first"]:
        if PARAMS["sell_sort"] == "value":
            sell_orders.sort(key=lambda o: -prices.get(o[1], 0) * o[2])
        elif PARAMS["sell_sort"] == "slope":
            sell_orders.sort(key=lambda o: -(price_f(o[1], market["inventory"][o[1]]) - price_f(o[1], market["inventory"][o[1]] + o[2])))
        else:
            sell_orders.sort(key=lambda o: -prices.get(o[1], 0))

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
        fl = PARAMS["hire_floor"]
        if fl:
            want_hands = max(want_hands, int(round(fl[min(day, len(fl) - 1)] * PARAMS["hire_floor_scale"])))
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
    all_orders = sell_orders + land_orders + wheat_orders + hire_orders + animal_orders + seed_orders + fert_orders
    all_orders = all_orders[:10]

    return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": all_orders}



def _angle(p):
    return math.atan2(p[1] - (HALF - 0.5), p[0] - (HALF - 0.5))


def _zones(st, jobs, tiles, n_units, day):
    """Split the farm into n_units angular sectors around the shed with equal
    expected work; unit i owns sector i. Recomputed when the crew changes."""
    key = (day, n_units)
    z = st.get("zones")
    if z is None or z[0] != key:
        items = []
        for y in range(BOARD):
            for x in range(BOARD):
                t = tiles[y][x]
                if t == "LOCKED":
                    continue
                w = 0.0
                if (x, y) in jobs:
                    w += len(jobs[(x, y)]["acts"])
                if is_animal(t):
                    w += 2.0
                elif is_plant(t):
                    w += 0.5
                elif t is not None:
                    w += 0.3
                if w > 0:
                    items.append((_angle((x, y)), w))
        items.sort()
        tot = sum(w for _, w in items) or 1.0
        bounds = []
        acc = 0.0
        k = 1
        for a, w in items:
            acc += w
            while k < n_units and acc >= tot * k / n_units:
                bounds.append(a)
                k += 1
        while len(bounds) < n_units - 1:
            bounds.append(math.pi)
        z = (key, bounds)
        st["zones"] = z
    bounds = z[1]

    def zone_of(p):
        a = _angle(p)
        i = 0
        while i < len(bounds) and a > bounds[i]:
            i += 1
        return i
    return zone_of



def _en_route(pos, jobs, inv, done_here):
    """Do a critical action on the tile we are walking over (costs one step,
    saves a later trip)."""
    mode = PARAMS["enroute"]
    if not mode or pos in done_here:
        return None
    job = jobs.get(pos)
    if job is None or (mode == "crit" and not job.get("crit")):
        return None
    for a_, val, need in job["acts"]:
        if a_[0] in ("PLANT", "PLACE", "BUILD_COOP", "BUILD_PASTURE", "DIG"):
            continue
        if need and inv.get(need, 0) <= 0:
            continue
        if mode == "crit" and a_[0] not in ("WATER", "FEED"):
            continue
        if val < PARAMS["enroute_min"]:
            continue
        done_here.add(pos)
        return a_
    return None

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
