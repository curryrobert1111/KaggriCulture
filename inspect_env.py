# inspect_env.py
import json
from kaggle_environments import make

def inspect():
    env = make("kaggriculture", configuration={"seed": 42})
    state = env.reset()
    obs = state[0].observation
    
    player_id = obs.get("player", 0)
    my_farm = obs.get("farms", [{}, {}])[player_id]
    
    print("================ My Farm 數據結構 ================")
    print(json.dumps(my_farm, indent=2, default=str))
    print("==================================================")

if __name__ == "__main__":
    inspect()