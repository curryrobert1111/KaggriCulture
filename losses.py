"""Count plant deaths and animal escapes for player 0 across seeds."""
import sys
from collections import Counter
from run_match import load_agent
from kaggle_environments import make
tot = Counter()
for seed in range(int(sys.argv[3]), int(sys.argv[3]) + int(sys.argv[4])):
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.run([load_agent(sys.argv[1], "A"), load_agent(sys.argv[2], "B")])
    for i in range(1, len(env.steps)):
        a = env.steps[i-1][0].observation["farms"][0]["tiles"]; b = env.steps[i][0].observation["farms"][0]["tiles"]
        hr = env.steps[i-1][0].observation["hour"]
        for y in range(10):
            for x in range(10):
                ta, tb = a[y][x], b[y][x]
                if not isinstance(ta, dict): continue
                if ta.get("kind") == "PLANT" and isinstance(tb, dict) and tb.get("kind") == "WEED":
                    tot[(ta["crop"], "unwatered" if hr == 23 and not ta["watered_today"] else "decay")] += 1
                if ta.get("animal") and isinstance(tb, dict) and not tb.get("animal"):
                    tot[(ta["animal"], "escaped")] += 1
    tot["reward"] += env.steps[-1][0].reward
print(dict(tot))
