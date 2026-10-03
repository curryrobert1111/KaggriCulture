import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from main4 import LGBMConfidenceGateEngine

# =====================================================================
# 1. 具備獨立 Seed 與決策風格的對手 Bot (Opponent Bot)
# =====================================================================
class DynamicOpponentBot:
    def __init__(self, bot_seed: int, noise_rate: float = 0.3):
        self.bot_seed = bot_seed
        self.noise_rate = noise_rate
        self.rng = np.random.RandomState(bot_seed)
        
        # 對手獨特的預測週期與盲區 (由 bot_seed 決定)
        self.wheat_cycle = self.rng.choice([5, 6, 7, 8])
        self.herd_cycle = self.rng.choice([9, 10, 11, 12])
        self.fake_cycle = self.rng.choice([6, 7, 8, 9])

    def reset(self):
        self.rng = np.random.RandomState(self.bot_seed)

    def get_actions(self, step: int) -> Dict[Tuple[int, str], int]:
        """對手 Bot 根據自己的 Seed 產生預測/發單訊號"""
        raw_preds = {}
        
        # 根據對手的週期產生訊號
        if step % self.wheat_cycle == 0: 
            raw_preds[(step + 3, "WHEAT")] = 2
        if step % self.herd_cycle == 0: 
            raw_preds[(step + 3, "HERD_A")] = 2
            
        # 根據對手的噪音率產生假訊號 (Noise)
        if self.rng.rand() < self.noise_rate and step % self.fake_cycle == 0:
            raw_preds[(step + 3, "HERD_B")] = 2
            
        return raw_preds

# =====================================================================
# 2. 你的 ML Agent 封裝
# =====================================================================
class MyMLAgent:
    def __init__(self, ml_threshold: float = 0.65):
        self.engine = LGBMConfidenceGateEngine(ml_threshold=ml_threshold)

    def reset(self):
        self.engine = LGBMConfidenceGateEngine(ml_threshold=self.engine.ml_threshold)

    def get_actions(self, step: int, actual_events: List[str], raw_preds: Dict[Tuple[int, str], int]) -> List[str]:
        self.engine.update_actual_events(step, actual_events)
        actions = []
        for item in ["WHEAT", "HERD_A", "HERD_B"]:
            qty = self.engine.get_gated_forecast(step, item, raw_preds)
            if qty >= 2:
                actions.append(item)
        return actions

# =====================================================================
# 3. 雙種子對戰模擬器 (Env Seed vs Bot Seed)
# =====================================================================
def run_pvp_match(my_agent: MyMLAgent, opponent: DynamicOpponentBot, env_seed: int) -> Tuple[int, int]:
    env_rng = np.random.RandomState(env_seed)
    my_agent.reset()
    opponent.reset()

    my_score, opp_score = 1000, 1000

    for step in range(720):
        # 1. 環境真實需求 (由 env_seed 決定)
        actual_events = []
        if (step + 3) % 10 == 0: actual_events.append("WHEAT")
        if (step + 2) % 15 == 0: actual_events.append("HERD_A")

        # 2. 對手 Bot 產出訊號 (由 opponent.bot_seed 決定)
        opp_raw_preds = opponent.get_actions(step)

        # 3. 兩邊做出的決策
        my_actions = my_agent.get_actions(step, actual_events, opp_raw_preds)
        
        # 對手 Bot (不開風控，盲目採信自己的訊號)
        opp_actions = [item for (t, item), qty in opp_raw_preds.items() if qty >= 2]

        # 4. 結算對戰得分
        for item in my_actions:
            my_score += 12 if item in actual_events else -15
            
        for item in opp_actions:
            opp_score += 12 if item in actual_events else -15

        # 微小環境波動
        my_score += env_rng.randint(-1, 2)
        opp_score += env_rng.randint(-1, 2)

    return my_score, opp_score

# =====================================================================
# 4. 對戰測試主程式
# =====================================================================
def start_bot_tournament(num_opponents: int = 20):
    my_agent = MyMLAgent(ml_threshold=0.65)
    
    # 產生不同的對手 Seed 與環境 Seed
    np.random.seed(2026)
    bot_seeds = np.random.randint(1000, 9999, size=num_opponents)
    env_seeds = np.random.randint(10000, 99999, size=num_opponents)

    results = []
    print(f"=== 開始與 {num_opponents} 位不同 Seed 的對手 Bot 進行 1v1 對戰 ===\n")

    for i, (b_seed, e_seed) in enumerate(zip(bot_seeds, env_seeds)):
        # 建立具備獨特行為模式的對手 Bot
        opponent = DynamicOpponentBot(bot_seed=b_seed, noise_rate=0.35)
        
        my_score, opp_score = run_pvp_match(my_agent, opponent, env_seed=e_seed)
        margin = my_score - opp_score
        is_win = "勝利" if margin > 0 else "敗北"

        results.append({
            "對局": f"Match #{i+1:02d}",
            "對手 Seed": b_seed,
            "環境 Seed": e_seed,
            "我的得分": my_score,
            "對手得分": opp_score,
            "分差 (Margin)": f"{margin:+d}",
            "結果": is_win
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # 彙總戰績
    wins = sum(1 for r in results if r["結果"] == "勝利")
    avg_margin = np.mean([int(r["分差 (Margin)"]) for r in results])
    print("\n" + "="*50)
    print(f"總戰績: {wins} 勝 / {num_opponents - wins} 負 (勝率: {wins / num_opponents:.1%})")
    print(f"平均每局領先分差: {avg_margin:+.2f} 分")
    print("="*50)

if __name__ == "__main__":
    start_bot_tournament(num_opponents=20)