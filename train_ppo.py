import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from kaggriculture_env import KaggricultureEnv

def make_env():
    """環境實例化工廠函數"""
    def _thunk():
        # 初步訓練對手先使用 "random"，建立基礎盈利能力
        return KaggricultureEnv(opponent_agent="random")
    return _thunk

if __name__ == "__main__":
    # 1. 建立向量化環境 (Parallelized Environments)
    # 透過多個環境並行收集數據，能大幅提升 PPO 收斂速度
    NUM_ENVS = 4
    vec_env = DummyVecEnv([make_env() for _ in range(NUM_ENVS)])

    # 2. 配置 PPO 模型超參數
    model = PPO(
        policy="MlpPolicy",            # 使用多層感知器 (MLP) 處理 1D 向量
        env=vec_env,
        learning_rate=3e-4,             # 經典 PPO 學習率
        n_steps=2048,                   # 每個環境每次採樣步數 (2048 * 4 = 8192 步更新一次)
        batch_size=128,                 # 梯度下降批次大小
        n_epochs=10,                    # 每次採樣數據重複訓練輪數
        gamma=0.99,                     # 折扣因子 (重視遠期獎勵)
        gae_lambda=0.95,                # GAE 平滑因子
        clip_range=0.2,                 # PPO 裁切範圍
        ent_coef=0.01,                  # 熵係數 (鼓勵探索，防止過早收斂到局部最佳)
        verbose=1,
        tensorboard_log="./tb_logs/"    # Tensorboard 數據監控路徑
    )

    print("=== 開始 PPO 模型訓練 ===")
    # 建議初步訓練設定 200,000 步 (約需 5~15 分鐘，視 CPU 性能而定)
    TOTAL_TIMESTEPS = 200_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 3. 儲存模型權重
    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_kaggriculture_v1"
    model.save(model_path)
    print(f"\n模型已成功儲存至: {model_path}.zip")

    # 4. 進行 10 場對局評估
    print("\n=== 開始評估模型勝率與收益 ===")
    eval_env = KaggricultureEnv(opponent_agent="random")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"10 場平均單步 Reward: {mean_reward:.3f} +/- {std_reward:.3f}")