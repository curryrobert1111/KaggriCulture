from kaggriculture_env import KaggricultureEnv

env = KaggricultureEnv()
obs, info = env.reset()

print("Observation Vector Shape:", obs.shape)
print("Observation Sample (First 10 values):", obs[:10])

# 隨機執行 5 步
for _ in range(5):
    action = env.action_space.sample()  # 隨機選擇 0~4
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward:.3f}, Terminated: {terminated}")