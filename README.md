# Kaggriculture agent + local lab

## Setup
```bash
pip install -U kaggle-environments   # >= 1.32 ships envs/kaggriculture (the real game engine)
```
The engine source is at `site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`.
Read it: several rules differ from the web page (see "Engine facts" below).

## Files
| file | purpose |
|---|---|
| `main.py` | the agent (single file, no dependencies). Submit this. |
| `run_match.py` | tournament: `python run_match.py main.py starter -n 10` (each seed played twice, sides swapped) |
| `trace.py` | per-day summary of one game: money, tile mix, hands, prices |
| `losses.py` | counts dead crops / escaped animals: `python losses.py main.py starter 2000 3` |
| `variant.py`, `sweep.sh` | parameter A/B: `./sweep.sh 6 7000 'name:{"cap_lambda":0.3}'` (variants vs `var/base.py`) |

## Workflow
1. `cp main.py var/base.py` (freeze the current best)
2. edit `main.py`
3. `python run_match.py main.py var/base.py -n 8` → keep the change if the win rate is clearly > 50%
4. `kaggle competitions submit kaggriculture -f main.py -m "vX"`

## How the agent works
1. **Market model**: exact price curves, expected town consumption (known shops + expected future shops),
   and pending supply from *both* farms (the opponent's farm is visible).
2. **Tile planner**: every free tile gets the crop/animal with the best value per tile-day at projected
   prices, after a capital charge; cash is held back so every tile gets planted.
3. **Jobs**: each tile lists its needed actions with a $ value (a plant that dies tonight if not watered
   is worth its whole expected yield; an unfed animal that escapes tonight is worth ~$490).
4. **Scheduler**: sticky assignment; free units pick the job with the highest value per turn spent.
   Units grab feed wheat at the shed at the start of the day.
5. **Market**: sell every turn (hold only when a price is forecast to rise), hire by workload,
   buy land when the current land is full, buy animals only for built structures, fertilize tomatoes and
   strawberries on production days.

## Engine facts that matter (from the source)
- Kaggle calls the **last callable** in `main.py` → `agent` must be defined last.
- Step 718 (day 29, hour 22) is the last processed step. End-of-day drops never happen on the last day.
- One-time crops start decaying at age `max_yield_day + 1`. Melon can't be harvested before age 10,
  so fertilizing melons gains nothing.
- The ongoing-crop held cap is `max_yield` (4). Harvest often when fertilized.
- FEED needs wheat in *that unit's* inventory. PLACE needs the animal in the inventory. SELL only sells from the shed.
- A DROP and a SELL in the same turn work (unit actions are processed before market orders).
- Every animal yields 1 fertilizer/day (~$90 early) → animals pay back fast.

## Results (local)
- vs built-in `starter`: 100% wins, ~$80–110k vs ~$3.5k
- vs its own earlier versions: v8 beats v4 11/1

## Next ideas (highest expected value first)
1. Download top-ladder replays (`kaggle competitions replay <id>`), run `trace.py`-style analysis on them, copy what works.
2. Opponent-aware selling: sell premium goods (melon/milk/wool/strawberry) *before* the opponent's visible harvests land.
3. Zone-based routing (fixed sweeps per unit) to cut walking further; ~50% of actions are still moves.
4. Tune `PARAMS` with larger sweeps (≥20 seeds per variant; 10-game results are noise).
5. Grow our own feed wheat instead of buying it when the wheat price is high.

## Learning from top teams (Ice & Fire)
```bash
kaggle competitions replay <EPISODE_ID> -p replays          # collect 20-50 episodes of one team
python ghost.py "replays/*.json" main.py --summary           # race your agent vs their recorded games
python icefire.py "replays/*.json" --team Majkel1337 --mine main.py --games 12   # ice/fire, land timing, money curve: theirs vs yours
python icefire.py "replays/*.json" --team Majkel1337 --export-opening opening.json --min-share 0.6
python icefire.py --embed opening.json main.py main_opening.py   # play their opening, then hand over to your agent
python run_match.py main_opening.py main.py -n 10              # keep it only if it wins
```
Local test: a 97-step extracted opening lifted the old v4 agent 12/0 vs plain v4.
Copying Majkel1337's land timing alone (PARAMS land_steps=[150,218,222]) LOST 1/11: copy coherent packages, then verify.
