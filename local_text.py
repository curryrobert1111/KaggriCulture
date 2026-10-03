import json
import os
from kaggle_environments import make
from main3 import agent  # 匯入你的 agent
from main1 import agent1  # 匯入另一個 agent
# 建立模擬環境
env = make("kaggriculture")

# 讓兩個 Agent 對決（或一個你寫的，一個預設的）
env.run([agent, agent1])

# 印出終局比分與結果
print("遊戲結果：")
for i, player in enumerate(env.state):
    print(f"Player {i}: reward={player.reward}, status={player.status}")

# 確保 replays 資料夾存在並存檔
os.makedirs("replays", exist_ok=True)
with open("replays/replay.json", "w", encoding="utf-8") as f:
    json.dump(env.toJSON(), f)

print("\n測試完成！對戰紀錄已儲存至 replays/replay.json")