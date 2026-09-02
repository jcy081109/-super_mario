"""快速验证可视化功能：随机策略玩游戏，测试渲染和录屏"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio
import numpy as np

from mario_rl.config import config
from mario_rl.env_wrappers import make_env

print("=== 可视化功能验证 ===")

# 1. 创建环境
env = make_env(seed=42)
obs = env.reset()
print(f"观测形状: {obs.shape}, dtype: {obs.dtype}")

# 2. 测试 rgb_array 渲染（录屏用，不需要显示窗口）
print("\n测试 rgb_array 渲染...")
frame = env.render(mode='rgb_array')
print(f"渲染帧形状: {frame.shape}, dtype: {frame.dtype}, 范围: [{frame.min()}, {frame.max()}]")
assert frame.shape == (240, 256, 3), f"期望 (240,256,3), 实际 {frame.shape}"
print("rgb_array 渲染正常 ✓")

# 3. 测试 human 渲染（显示窗口，可能在无头环境失败）
print("\n测试 human 渲染...")
try:
    env.render(mode='human')
    print("human 渲染正常 ✓（窗口已显示）")
except Exception as e:
    print(f"human 渲染跳过（无头环境或无显示）: {e}")

# 4. 随机策略玩 100 步并录屏
print("\n随机策略玩 100 步并录屏...")
frames = []
total_reward = 0
for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    frame = env.render(mode='rgb_array')
    frames.append(frame)
    if done:
        print(f"  第 {i+1} 步死亡，重置")
        obs = env.reset()

print(f"  100 步完成，累计奖励: {total_reward:.1f}, 最终 x_pos: {info.get('x_pos')}")

# 5. 保存视频
video_dir = config.paths.video_dir
video_dir.mkdir(parents=True, exist_ok=True)
video_path = video_dir / "visual_test_random.mp4"
imageio.mimwrite(str(video_path), frames, fps=30, quality=8)
print(f"\n视频已保存: {video_path}")
print(f"视频帧数: {len(frames)}, 每帧形状: {frames[0].shape}")

env.close()
print("\n=== 可视化功能验证全部通过 ===")
