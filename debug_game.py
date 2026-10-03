# debug_game.py
from kaggle_environments import make
from main import agent

def run_debug():
    env = make("kaggriculture", configuration={"seed": 42})
    state = env.reset()

    print("=== 前 30 回合單步執行日誌 ===")
    for step in range(30):
        my_action = agent(state[0].observation, env.configuration)
        state = env.step([my_action, {"farmer": ["PASS"], "hands": [], "market": []}])
        
        obs = state[0].observation
        p_id = obs.get("player", 0)
        my_farm = obs["farms"][p_id]
        
        money = my_farm.get("money")
        farmer = my_farm.get("farmer")
        
        # 統計 NW 區域開墾/種植狀態
        tiles = my_farm.get("tiles", [])
        active_tiles = []
        if tiles:
            for r in range(5):
                for c in range(5):
                    if tiles[r][c] is not None and tiles[r][c] != "LOCKED":
                        active_tiles.append((r, c, tiles[r][c]))

        print(f"Step {step:02d} | Money: {money} | Farmer Pos: {farmer} | Executed Action: {my_action['farmer']}")
        if my_action['market']:
            print(f"        | Market Action: {my_action['market']}")
        if active_tiles:
            print(f"        | Active Tiles: {active_tiles}")

if __name__ == "__main__":
    run_debug()