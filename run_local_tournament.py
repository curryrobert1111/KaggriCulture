import pandas as pd
from kaggle_environments import make

# 匯入你的 Agent 進入點
from main import agent as my_agent


# =====================================================================
# 1. 具備地圖移動能力之 Baseline Opponent
# =====================================================================
def baseline_bot(obs, config):
    player_id = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_id] if len(farms) > player_id else {}

    money = my_farm.get("money", 3000)
    tiles = my_farm.get("tiles", [])
    farmer_pos = my_farm.get("farmer", [4, 4])
    shed = my_farm.get("shed", my_farm.get("inventory", {}))

    market_cmds = []
    farmer_cmd = ["PASS"]

    # 1. 買種子與賣貨
    if isinstance(shed, dict):
        for item, qty in shed.items():
            if isinstance(qty, (int, float)) and qty > 0 and not str(item).endswith("_seed"):
                market_cmds.append(["SELL", str(item), int(qty)])

    wheat_seeds = shed.get("wheat_seed", 0) if isinstance(shed, dict) else 0
    if wheat_seeds == 0 and money >= 50:
        market_cmds.append(["BUY_SEED", "wheat", 5])

    # 2. 地圖尋路移動 (前往 [0, 0] 種田)
    target = (0, 0)
    if farmer_pos == [0, 0]:
        farmer_cmd = ["PLANT", "wheat"]
    else:
        r, c = farmer_pos[0], farmer_pos[1]
        if r > 0:
            farmer_cmd = ["NORTH"]
        elif c > 0:
            farmer_cmd = ["WEST"]

    return {
        "farmer": farmer_cmd,
        "hands": [],
        "market": market_cmds
    }


# =====================================================================
# 2. 錦標賽測試主程式
# =====================================================================
def start_local_tournament(num_games: int = 10):
    print(f"=== 開始進行 {num_games} 局 1v1 本地對決 ===\n")
    results = []

    for game_idx in range(num_games):
        seed = 1000 + game_idx
        is_even = (game_idx % 2 == 0)

        p0_agent = my_agent if is_even else baseline_bot
        p1_agent = baseline_bot if is_even else my_agent

        env = make("kaggriculture", configuration={"seed": seed})
        env.run([p0_agent, p1_agent])

        p0_score = env.state[0].reward or 0
        p1_score = env.state[1].reward or 0

        my_score = float(p0_score if is_even else p1_score)
        opp_score = float(p1_score if is_even else p0_score)
        margin = my_score - opp_score
        is_win = "勝利" if margin > 0 else ("平手" if margin == 0 else "敗北")

        results.append({
            "對局": f"Match #{game_idx + 1:02d}",
            "Seed": seed,
            "我方立場": "P0 (先手)" if is_even else "P1 (後手)",
            "我的得分": round(my_score, 1),
            "對手得分": round(opp_score, 1),
            "分差": f"{margin:+.1f}",
            "結果": is_win
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    start_local_tournament(num_games=10)