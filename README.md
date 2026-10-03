# KaggriCulture

A single-file Python agent for Kaggle's [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) simulation, plus a local lab for testing it.

Kaggriculture is a two-player, 30-day farming and trading game (720 turns). Each player grows crops, raises animals, hires hands and sells into a shared market whose prices react to supply. The player with more coins after the last turn wins. Leaderboard rating counts only wins, losses and ties, not the margin.

## Quick start

```bash
pip install -U kaggle-environments          # version 1.32 or later ships the game engine
python run_match.py main.py starter -n 10    # 10 seeds, each played twice with sides swapped
kaggle competitions submit kaggriculture -f main.py -m "my agent"
```

The engine source, `kaggle_environments/envs/kaggriculture/kaggriculture.py`, is the ground truth for the rules. Several details differ from the competition web page.

## Files

| File | Purpose |
|---|---|
| `main.py` | The agent. One file, no dependencies. This is what you submit. |
| `sim.py` | Fast simulator that drives the real engine without Kaggle's per-step copies. About 4× faster than `kaggle_environments`, with identical results. |
| `ev.py` | Benchmark. Ghost races against recorded top-team games plus head-to-head games against a reference agent, reported with standard errors. |
| `ghost.py` | Ghost race against one recorded replay, using the recorded shops and weeds. |
| `run_match.py` | Head-to-head tournament on the full Kaggle environment, sides swapped. |
| `variant.py`, `sweep.sh` | Make agent variants with `PARAMS` overrides and sweep them. |
| `trace.py`, `losses.py`, `icefire.py` | Per-day game summaries, dead crops and escaped animals, and top-team analysis. |
| `1153*.json`, `replays/` | Recorded ladder games used for ghost races. |

## How the agent works

`main.py` has three layers. Every tunable knob lives in the `PARAMS` dict at the top of the file.

### 1. Market model

- **Exact prices.** It uses the engine's price curves.
- **Projected inventory.** For every product and every remaining day, it projects market inventory from three things: town demand, both farms' production schedules, and the shops that are open or still expected. The opponent's farm is visible, so its supply is included.
- **New production.** A new crop or animal is valued at the projected price on the days it actually sells. It is also charged for how much its sales will lower the price of our own later sales.

### 2. Planner

- **Choosing what to plant.** Each free tile gets the crop or animal with the best value per tile-day after a capital charge. The planner saves cash when an unaffordable option is much better.
- **Land.** Land is bought on a fixed schedule: steps 147, 198 and 250.
- **Hands.** Hiring follows a daily minimum schedule. The marginal hand is capped at $150 a day because hand costs grow as a Fibonacci sequence.
- **Early strawberries.** Strawberries are pushed onto tiles freed by the early wheat harvest on days 2–7.

### 3. Scheduler and market orders

- **Nearest job first.** Each worker takes the nearest useful job (score = value / distance¹⁰). This roughly halved the walking of the earlier value-per-turn scheduler.
- **Urgent jobs.** A crop that would die tonight or an animal that would escape tonight pulls in the nearest worker when 2 hours or less of slack remain. Workers also water or feed urgent tiles they walk over.
- **Animals in the shed.** Animals waiting in the shed are carried straight to their empty coop or pasture.
- **Pocket goods.** Goods stay in workers' pockets, since the engine drops them into the shed at night. Workers walk them back early only when cash is short, or after 5 pm to sell them that evening.
- **Sell orders.** Both players' orders are processed in parallel by their position in the order list, and each sale lowers the price. The agent lists first the goods whose price would fall the most from its own sale, so it sells before the opponent does.

## Results (local)

| Test | Result |
|---|---|
| vs the original agent in this repo, head-to-head | 40/40 wins, $99.7k vs $83.1k on average |
| vs the version just before, head-to-head | 44/80 wins, +$0.6k on average |
| Ghost race vs 5 DECEM (ladder #1) replays | wins most games, but DECEM's recorded moves go off track against a new opponent, so this overstates our strength |
| Ghost race vs 5 non-drifting top-team replays (DSM, Yizhou, Boey) | ~$88k vs ~$130k, still loses |

## Testing a change

```bash
cp main.py var/base.py                                     # freeze the current best
# edit main.py, or make a variant:
python variant.py var/try.py '{"evening_hour": 15}' main.py
python ev.py var/try.py --ghost clean -w 2 --h2h var/base.py -n 40
```

Keep a change only if it clearly wins. Head-to-head differences under about $1.5k, or ghost differences under about $3k, are noise.

## Rules of the engine that matter

- **Last callable.** Kaggle calls the last callable in `main.py`, so `agent` must be defined last. It must never raise.
- **Last step.** Step 718 (day 29, hour 22) is the last processed step. Unsold goods are worth nothing.
- **Hour 23.** Actions at hour 23 count on every day except the last.
- **Crop decay.** One-time crops start decaying at age `max_yield_day + 1`. Fertilizing melons gains nothing.
- **Inventories.** FEED needs wheat in that unit's own inventory. PLACE needs the animal in the inventory. SELL only sells from the shed.
- **Same-turn sales.** Unit actions resolve before market orders, so a DROP and a SELL in the same turn both work.
- **Fertilizer.** Every animal yields 1 fertilizer per day. Fertilizer has no town demand, so its price only falls as both players sell it.

## Known gap and next steps

The top teams out-earn this agent mainly through timing and plans chosen for each set of opening shops, not through faster work. When our agent was given a top team's exact planting plan, it still earned about $98k against their $155k. They spend nearly all their cash in days 1–10 and pick their animals and crops from the first two shops. They also switch to stable-priced crops (tomatoes, eggs, wheat, carrots) once premium prices crash.

The most promising next steps:

1. **Early cash-flow model.** Rank purchases by payback so cash compounds in days 1–10.
2. **Strategies per opening-shop pair.** Tune a separate strategy offline for each pair of opening shops and select it on day 6.
3. **Opponent-aware forecast.** Predict the opponent's future plantings so we stop over-planting crops that are about to crash.
