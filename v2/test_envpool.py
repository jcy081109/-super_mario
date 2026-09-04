import sys
sys.path.insert(0, '.')
import time
from mario_rl.env_wrappers import make_env

print("=== 测试环境池版 RandomLevelEnv ===")

# 创建多关卡环境（24关）
print("创建多关卡环境（24关环境池）...")
t0 = time.time()
env = make_env(multi_level=True, multi_worlds=(1,2,3,4,5,6,7,8), multi_stages=(1,2,3), seed=42)
t1 = time.time()
print(f"  创建耗时: {t1-t0:.2f}s")
print(f"  观测空间: {env.observation_space.shape}")
print(f"  动作空间: {env.action_space.n}")

# 测试多次 reset 和 step，统计耗时
print("\n测试 100 次 reset + 100 步...")
t0 = time.time()
for i in range(100):
    obs = env.reset()
    for _ in range(10):
        obs, r, done, info = env.step(1)
        if done:
            break
t1 = time.time()
print(f"  100次reset+1000步总耗时: {t1-t0:.2f}s")
print(f"  平均每次reset+10步: {(t1-t0)/100*1000:.1f}ms")

# 测试单关卡环境作为对比
print("\n对比：单关卡环境...")
env_single = make_env(world=1, stage=1, seed=42)
t0 = time.time()
for i in range(100):
    obs = env_single.reset()
    for _ in range(10):
        obs, r, done, info = env_single.step(1)
        if done:
            break
t1 = time.time()
print(f"  100次reset+1000步总耗时: {t1-t0:.2f}s")
print(f"  平均每次reset+10步: {(t1-t0)/100*1000:.1f}ms")

env.close()
env_single.close()
print("\n环境池版 RandomLevelEnv 测试通过！")
