"""工具函数：随机种子设置、设备检测、日志"""
import random

import numpy as np
import torch

from .config import config


def set_seed(seed: int = None):
    """固定所有随机种子，保证可复现。"""
    seed = seed if seed is not None else config.env.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """检测可用设备：优先 CUDA，否则 CPU。"""
    if config.ppo.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.ppo.device)


def print_env_info(env):
    """打印环境关键信息，用于调试。"""
    print(f"  观测空间: {env.observation_space}")
    print(f"  观测形状: {env.observation_space.shape}, dtype: {env.observation_space.dtype}")
    print(f"  动作空间: {env.action_space}")
    print(f"  动作数: {env.action_space.n}")
