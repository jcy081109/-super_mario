"""阶段二环境验证：包导入 + 马里奥环境创建 + 随机策略闭环"""
import sys

print("=== 1. 核心包导入 ===")
import torch
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import numpy as np
print(f"numpy: {np.__version__}")

import gym
print(f"gym: {gym.__version__}")

import gym_super_mario_bros
from importlib.metadata import version as _pkg_version
print(f"gym_super_mario_bros: {_pkg_version('gym-super-mario-bros')}")

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
print(f"SIMPLE_MOVEMENT: {len(SIMPLE_MOVEMENT)} actions -> {SIMPLE_MOVEMENT}")

from stable_baselines3 import PPO
print(f"stable_baselines3 PPO imported OK")

import cv2
print(f"opencv: {cv2.__version__}")

print("\n=== 2. 环境创建 ===")
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(f"action_space: {env.action_space}")
print(f"observation_space: {env.observation_space}")

print("\n=== 3. 随机策略闭环（50步） ===")
state = env.reset()
print(f"reset -> state shape: {state.shape}, dtype: {state.dtype}, min/max: {state.min()}/{state.max()}")

total_reward = 0
episode_count = 0
for i in range(50):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        episode_count += 1
        print(f"  step {i+1}: episode finished, total_reward={total_reward:.1f}, x_pos={info.get('x_pos')}, flag_get={info.get('flag_get')}")
        state = env.reset()
        total_reward = 0
    else:
        state = next_state

print(f"\nFinal info: x_pos={info.get('x_pos')}, y_pos={info.get('y_pos')}, "
      f"coins={info.get('coins')}, score={info.get('score')}, "
      f"flag_get={info.get('flag_get')}, time={info.get('time')}")
print(f"Episodes finished during 50 steps: {episode_count}")

env.close()
print("\n=== ENV VERIFICATION PASSED ===")
