from kaggle_environments import make
from main import agent as my_agent

# 1. 在本檔案中直接定義對手的 baseline_bot（避免跨檔案 import）
def baseline_bot(obs, config):
    player_id = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}]) if "farms" in obs else [{}, {}]
    my_farm = farms[player_id] if len(farms) > player_id else {}
    shed = my_farm.get("shed", {})
    plots = my_farm.get("plots", [])

    farmer_actions = []
    market_actions = []

    for i, plot in enumerate(plots):
        if plot.get("crop") is None:
            farmer_actions.append({"action": "plant", "plot_index": i, "crop": "wheat"})
        elif plot.get("stage") == "mature":
            farmer_actions.append({"action": "harvest", "plot_index": i})

    for item, qty in shed.items():
        if qty > 0:
            market_actions.append({"action": "sell", "item": item, "quantity": qty, "target": "market"})

    return {
        "farmer": farmer_actions,
        "hands": [],
        "market": market_actions
    }

# 2. 診斷主程式
def run_debug():
    print("=== 開始診斷 Kaggriculture 環境 ===")
    
    # 建立環境
    env = make("kaggriculture", configuration={"seed": 1000})

    # 查看官方規格要求
    print("\n--- 1. 官方環境要求的 Action 格式規範 (Specification) ---")
    print(env.specification.action)

    # 執行對決
    env.run([my_agent, baseline_bot])

    # 查看結束狀態
    print("\n--- 2. 遊戲結束狀態 (Status & Score) ---")
    print(f"P0 (my_agent) Status: {env.state[0].status} | Score: {env.state[0].reward}")
    print(f"P1 (baseline) Status: {env.state[1].status} | Score: {env.state[1].reward}")

    # 查看第一回合觀察值結構
    print("\n--- 3. Step 0 實際收到的 Observation 欄位 ---")
    if env.steps and len(env.steps) > 0:
        first_obs = env.steps[0][0]["observation"]
        print("Obs 欄位:", list(first_obs.keys()))

    # 查看錯誤訊息 Log
    print("\n--- 4. 系統 Log / 錯誤訊息 ---")
    has_logs = False
    if hasattr(env, "logs") and env.logs:
        for step_idx, log in enumerate(env.logs):
            if any(log):
                print(f"Step {step_idx}:", log)
                has_logs = True
    if not has_logs:
        print("沒有列印出錯誤 log。")

if __name__ == "__main__":
    run_debug()