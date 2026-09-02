"""
阶段三管线验证：环境 Wrapper → PPO 初始化 → 极短训练 → 保存/加载 → 推理。
不追求训练效果，只验证全流程能跑通。
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mario_rl.config import config
from mario_rl.env_wrappers import make_env
from mario_rl.model import get_policy_kwargs
from mario_rl.utils import get_device, set_seed

print("=" * 60)
print("阶段三：管线验证")
print("=" * 60)

# 1. 设备
device = get_device()
print(f"\n[1/6] 设备: {device}")
set_seed(42)

# 2. 环境 Wrapper 管线
print(f"\n[2/6] 环境 Wrapper 管线验证")
env = make_env(seed=42)
obs = env.reset()
print(f"  reset 观测形状: {obs.shape}, dtype: {obs.dtype}")
print(f"  像素范围: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"  动作空间: {env.action_space.n} 个动作")
assert obs.shape == (4, 84, 84), f"期望 (4,84,84), 实际 {obs.shape}"
assert obs.dtype == np.float32
assert 0.0 <= obs.min() and obs.max() <= 1.0

# 连续 step
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
print(f"  连续 10 步 step 无异常，最后 x_pos={info.get('x_pos')}")
env.close()

# 3. PPO 模型初始化
print(f"\n[3/6] PPO 模型初始化")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = DummyVecEnv([lambda: make_env(seed=42)])
model = PPO(
    "CnnPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=64,
    batch_size=16,
    n_epochs=2,
    verbose=0,
    device=device,
    policy_kwargs=get_policy_kwargs(),
)
n_params = sum(p.numel() for p in model.policy.parameters())
print(f"  模型参数量: {n_params:,}")
print(f"  策略网络结构:")
for name, module in model.policy.named_children():
    print(f"    {name}: {module.__class__.__name__}")

# 4. 极短训练闭环
print(f"\n[4/6] 极短训练闭环（128 步，验证不报错）")
model.learn(total_timesteps=128, log_interval=None)
print(f"  训练 128 步完成，无异常")

# 5. 模型保存与加载
print(f"\n[5/6] 模型保存与加载")
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / "test_model"
    model.save(str(model_path))
    print(f"  模型已保存: {model_path}.zip")

    loaded_model = PPO.load(str(model_path), device=device)
    print(f"  模型已加载")

    # 6. 推理验证
    print(f"\n[6/6] 推理验证")
    eval_env = make_env(seed=99)
    eval_obs = eval_env.reset()
    action, _ = loaded_model.predict(eval_obs, deterministic=True)
    action = int(action)  # model.predict 返回 ndarray，JoypadSpace 需要 int
    print(f"  推理输出动作: {action} (0-6)")
    assert 0 <= action <= 6, f"动作应在 0-6 范围内，实际 {action}"

    # 连续推理 20 步
    total_reward = 0
    for _ in range(20):
        action, _ = loaded_model.predict(eval_obs, deterministic=True)
        eval_obs, reward, done, info = eval_env.step(int(action))
        total_reward += reward
        if done:
            eval_obs = eval_env.reset()
    print(f"  连续推理 20 步无异常，累计奖励: {total_reward:.1f}")
    eval_env.close()

vec_env.close()

print("\n" + "=" * 60)
print("管线验证全部通过！")
print("=" * 60)
