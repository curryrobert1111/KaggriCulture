import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make



class KaggricultureEnv(gym.Env):
    """
    將 Kaggle Kaggriculture 包裝為 Gymnasium 相容環境
    - Observation Space: 1D Float32 Numpy Array (歸一化特徵向量)
    - Action Space: Discrete(5) 高階巨集指令
    """
    metadata = {"render_modes": []}

    def __init__(self, opponent_agent=None):
        super().__init__()
        self.raw_env = make("kaggriculture")
        # 若未指定對手，預設使用 random 或者是你寫好的 FSM agent
        self.opponent_agent = opponent_agent if opponent_agent else "random"
        
        # 1. 定義巨集動作空間 (Macro Action Space)
        # 0: 採收優先 | 1: 澆水優先 | 2: 播種優先 | 3: 除草與擴張 | 4: 出售庫存變現
        self.action_space = spaces.Discrete(5)
        
        # 2. 定義觀察空間 (Observation Space)
        # 包含：全域時間 (3) + 玩家狀態 (3) + 倉庫與種子 (10) + 市場價格 (5) + 地圖網格 (10x10x4 = 400)
        self.OBS_DIM = 3 + 3 + 10 + 5 + 400
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        
        self.trainer = None
        self.current_obs = None
        self.prev_money = 3000.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 利用 Kaggle 自帶的 train 介面建立單人對決環境
        self.trainer = self.raw_env.train([None, self.opponent_agent])
        raw_obs = self.trainer.reset()
        
        self.current_obs = raw_obs
        self.prev_money = raw_obs["farms"][raw_obs["player"]]["money"]
        
        obs_vector = self._extract_features(raw_obs)
        info = {}
        return obs_vector, info

    def step(self, action):
        # 1. 將 RL 輸出的巨集動作 (0~4) 轉化為 Kaggle 底層指令
        kaggle_action = self._macro_to_kaggle_action(action, self.current_obs)
        
        # 2. 向 Kaggle 環境推進一步
        raw_obs, reward_val, terminated, info = self.trainer.step(kaggle_action)
        self.current_obs = raw_obs
        
        # 3. 提取特徵向量
        obs_vector = self._extract_features(raw_obs)
        
        # 4. 獎勵函數設計 (Reward Shaping)
        current_money = raw_obs["farms"][raw_obs["player"]]["money"]
        money_delta = current_money - self.prev_money
        
        # 將金錢變動歸一化作為獎勵 (賺 100 元得 +1.0 獎勵，賠錢扣分)
        reward = money_delta / 100.0
        self.prev_money = current_money
        
        # 遊戲回合結束 (720 步)
        truncated = False
        
        return obs_vector, reward, terminated, truncated, info

    def _extract_features(self, obs):
        """將 Kaggle nested dict 轉化為神經網路看得懂的 1D 向量"""
        features = []
        
        # === A. 全域時間特徵 (歸一化至 0~1) ===
        day = obs.get("day", 0) / 30.0
        hour = obs.get("hour", 0) / 24.0
        step = obs.get("step", 0) / 720.0
        features.extend([day, hour, step])
        
        # === B. 我方農場狀態 ===
        player_id = obs["player"]
        my_farm = obs["farms"][player_id]
        money = my_farm["money"] / 10000.0  # 假設最大資金 10000
        fx, fy = my_farm["farmer"]
        features.extend([money, fx / 10.0, fy / 10.0])
        
        # === C. 私有倉庫與種子 (精選核心 5 種作物) ===
        items = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "WATERMELON"]
        shed = obs["private"]["shed"]
        seeds = obs["private"]["seeds"]
        
        for item in items:
            features.append(shed.get(item, 0) / 50.0)   # 倉庫庫存歸一化
        for item in items:
            features.append(seeds.get(item, 0) / 20.0)  # 種子數量歸一化
            
        # === D. 市場價格 ===
        prices = obs["market"]["prices"]
        for item in items:
            features.append(prices.get(item, 0) / 300.0) # 價格歸一化
            
        # === E. 10x10 農田狀態 (每格 4 個特徵) ===
        tiles = my_farm["tiles"]
        grid_features = []
        for y in range(10):
            for x in range(10):
                tile = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else "LOCKED"
                
                is_empty = 1.0 if tile is None else 0.0
                is_weed = 1.0 if isinstance(tile, dict) and tile.get("kind") == "WEED" else 0.0
                is_plant = 1.0 if isinstance(tile, dict) and tile.get("kind") == "PLANT" else 0.0
                
                watered = 0.0
                if is_plant and tile.get("watered_today"):
                    watered = 1.0
                    
                grid_features.extend([is_empty, is_weed, is_plant, watered])
                
        features.extend(grid_features)
        
        return np.array(features, dtype=np.float32)

    def _macro_to_kaggle_action(self, macro_action, obs):
        """將高階決策轉化為底層可執行的 command (簡單 FSM 轉譯器)"""
        player_id = obs["player"]
        my_farm = obs["farms"][player_id]
        fx, fy = my_farm["farmer"]
        pos = (fx, fy)
        tiles = my_farm["tiles"]
        money = my_farm["money"]
        seeds = obs["private"]["seeds"]
        shed = obs["private"]["shed"]
        
        market_actions = []
        farmer_action = ["PASS"]
        
        # 1. 巨集動作 4: 拋售所有庫存
        if macro_action == 4:
            for item, count in shed.items():
                if count > 0: market_actions.append(["SELL", item, count])

        # 2. 尋找目標格子 (簡單尋路輔助)
        def get_nearest(target_list):
            if not target_list: return None
            return min(target_list, key=lambda t: abs(t[0]-pos[0]) + abs(t[1]-pos[1]))

        def step_toward(target):
            tx, ty = target
            if pos[0] < tx: return ["EAST"]
            if pos[0] > tx: return ["WEST"]
            if pos[1] < ty: return ["SOUTH"]
            if pos[1] > ty: return ["NORTH"]
            return ["PASS"]

        # 掃描地圖任務
        harvests, waters, plants, weeds = [], [], [], []
        for y in range(len(tiles)):
            for x in range(len(tiles[y])):
                t = tiles[y][x]
                if t is None: plants.append((x, y))
                elif isinstance(t, dict):
                    if t.get("kind") == "WEED": weeds.append((x, y))
                    elif t.get("kind") == "PLANT":
                        if obs["day"] - t.get("planted_day", obs["day"]) >= 2:
                            harvests.append((x, y))
                        elif not t.get("watered_today", False):
                            waters.append((x, y))

        # 根據 RL 的巨集動作選擇行為
        if macro_action == 0 and harvests: # 採收優先
            target = get_nearest(harvests)
            farmer_action = ["HARVEST"] if pos == target else step_toward(target)
        elif macro_action == 1 and waters: # 澆水優先
            target = get_nearest(waters)
            farmer_action = ["WATER"] if pos == target else step_toward(target)
        elif macro_action == 2 and plants: # 播種優先
            if sum(seeds.values()) == 0 and money >= 10:
                market_actions.append(["BUY_SEED", "WHEAT", 2])
            target = get_nearest(plants)
            farmer_action = ["PLANT", "WHEAT"] if pos == target else step_toward(target)
        elif macro_action == 3 and weeds:  # 除草優先
            target = get_nearest(weeds)
            farmer_action = ["DIG"] if pos == target else step_toward(target)

        return {"farmer": farmer_action, "hands": [], "market": market_actions[:10]}