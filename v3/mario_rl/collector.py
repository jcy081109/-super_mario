"""
数据收集器：多环境并行收集 rollout 数据。
支持 SubprocVecEnv 真并行，记录 episode 统计。

收集流程：
  1. 模型 eval 模式，no_grad 推理
  2. 每步：模型推理出动作 → 环境 step → 存储到 buffer
  3. 收集 n_steps 后，计算最后一个状态的价值（用于 GAE bootstrap）
  4. 返回 buffer、last_values、收集时间、收集步数
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .model import ActorCritic
from .ppo import RolloutBuffer
from .utils import RewardNormalizer


# ─── 统计结果 ───────────────────────────────────────────────────────

@dataclass
class EpisodeStatsResult:
    """Episode 统计结果。

    Attributes:
        ep_rew_mean: 平均 episode 奖励。
        ep_len_mean: 平均 episode 长度。
        ep_x_pos_mean: 平均最终 x 坐标。
        ep_win_rate: 通关率（0.0~1.0）。
        n_episodes: 统计的 episode 数。
    """
    ep_rew_mean: float
    ep_len_mean: float
    ep_x_pos_mean: float
    ep_win_rate: float
    n_episodes: int

    def to_dict(self) -> dict:
        """转换为字典（兼容旧代码）。"""
        return {
            "ep_rew_mean": self.ep_rew_mean,
            "ep_len_mean": self.ep_len_mean,
            "ep_x_pos_mean": self.ep_x_pos_mean,
            "ep_win_rate": self.ep_win_rate,
            "n_episodes": self.n_episodes,
        }


@dataclass
class CollectResult:
    """一次收集的结果。

    Attributes:
        buffer: 收集好数据的 RolloutBuffer。
        last_values: (n_envs,) 最后一个状态的价值（用于 GAE bootstrap）。
        collect_time: 收集耗时（秒）。
        n_collected: 收集的总步数（n_steps × n_envs）。
    """
    buffer: RolloutBuffer
    last_values: np.ndarray
    collect_time: float
    n_collected: int


# ─── Episode 统计 ───────────────────────────────────────────────────

class EpisodeStats:
    """Episode 统计记录器，滑动窗口记录最近 episode 的统计。

    Attributes:
        ep_rew: 最近 episode 奖励。
        ep_len: 最近 episode 长度。
        ep_x_pos: 最近 episode 最终 x 坐标。
        ep_win: 最近 episode 是否通关（1.0/0.0）。
    """

    def __init__(self, window_size: int = 100) -> None:
        self.ep_rew: deque = deque(maxlen=window_size)
        self.ep_len: deque = deque(maxlen=window_size)
        self.ep_x_pos: deque = deque(maxlen=window_size)
        self.ep_win: deque = deque(maxlen=window_size)

    def update_from_infos(self, infos: list, dones: np.ndarray) -> None:
        """从环境 info 中提取 episode 统计。

        当某个环境 done 时，从其 info 中提取 Monitor 记录的 episode 奖励/长度，
        以及 x_pos 和 flag_get。

        Args:
            infos: 各环境的 info 字典列表。
            dones: 各环境是否结束的布尔数组。
        """
        for i, info in enumerate(infos):
            if dones[i]:
                ep_info = info.get("episode", {})
                if ep_info:
                    self.ep_rew.append(ep_info.get("r", 0))
                    self.ep_len.append(ep_info.get("l", 0))
                # 从 info 中提取 x_pos 和 flag_get（Monitor wrapper 记录）
                if "x_pos" in info:
                    self.ep_x_pos.append(info["x_pos"])
                if "flag_get" in info:
                    self.ep_win.append(1.0 if info["flag_get"] else 0.0)

    def get_stats(self) -> EpisodeStatsResult:
        """获取当前统计。

        Returns:
            EpisodeStatsResult 统计结果。
        """
        return EpisodeStatsResult(
            ep_rew_mean=float(np.mean(self.ep_rew)) if self.ep_rew else 0.0,
            ep_len_mean=float(np.mean(self.ep_len)) if self.ep_len else 0.0,
            ep_x_pos_mean=float(np.mean(self.ep_x_pos)) if self.ep_x_pos else 0.0,
            ep_win_rate=float(np.mean(self.ep_win)) if self.ep_win else 0.0,
            n_episodes=len(self.ep_rew),
        )


# ─── 数据收集器 ─────────────────────────────────────────────────────

class Collector:
    """多环境数据收集器。

    用当前策略收集 n_steps 步数据到 RolloutBuffer。

    Args:
        model: ActorCritic 模型（收集时使用 eval 模式）。
        envs: 向量化环境（VecEnv）。
        n_envs: 并行环境数。
        n_steps: 每环境收集步数。
        obs_shape: 观测空间形状。
        device: 计算设备。
        normalize_reward: 是否启用奖励归一化（SB3 VecNormalize 风格）。
        gamma: 折扣因子（奖励归一化时用于计算 discounted return）。

    Attributes:
        obs: 当前观测，初始化时 reset。
        stats: EpisodeStats 统计器（记录原始奖励，不归一化）。
        reward_normalizer: 奖励归一化器（启用时存在）。
    """

    def __init__(
        self,
        model: ActorCritic,
        envs,
        n_envs: int,
        n_steps: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        normalize_reward: bool = False,
        gamma: float = 0.99,
    ) -> None:
        self.model = model
        self.envs = envs
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.device = device
        self.stats = EpisodeStats()
        self.reward_normalizer = RewardNormalizer(gamma=gamma) if normalize_reward else None

        # 初始化环境
        self.obs = envs.reset()

    def collect(self, buffer: RolloutBuffer) -> CollectResult:
        """收集 n_steps 步数据。

        模型 eval 模式，no_grad 推理，每步存储到 buffer。
        收集完后计算最后一个状态的价值（用于 GAE bootstrap）。

        Args:
            buffer: 要写入的 RolloutBuffer（会被 reset 并填充）。

        Returns:
            CollectResult 收集结果（buffer、last_values、耗时、步数）。
        """
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            for step in range(self.n_steps):
                # 1. 模型推理
                obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
                actions, log_probs, values, _ = self.model.get_action(obs_tensor)

                # 2. 环境 step
                actions_np = actions.cpu().numpy()
                next_obs, rewards, dones, infos = self.envs.step(actions_np)

                # 3. 记录 episode 统计（用原始奖励，不归一化）
                self.stats.update_from_infos(infos, dones)

                # 4. 奖励归一化（启用时，归一化后的奖励存入 buffer 用于训练）
                store_rewards = rewards
                if self.reward_normalizer is not None:
                    store_rewards = self.reward_normalizer.normalize(rewards, dones)

                # 5. 存储到 buffer
                buffer.add(
                    obs=self.obs,
                    action=actions_np,
                    reward=store_rewards,
                    done=dones,
                    value=values.detach().cpu().numpy(),
                    log_prob=log_probs.detach().cpu().numpy(),
                )

                # 6. 更新 obs
                self.obs = next_obs

            # 收集完后，计算最后一个状态的价值（用于 GAE bootstrap）
            obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
            last_values = self.model.get_value(obs_tensor).cpu().numpy()

        collect_time = time.time() - start_time
        n_collected = self.n_steps * self.n_envs
        return CollectResult(
            buffer=buffer,
            last_values=last_values,
            collect_time=collect_time,
            n_collected=n_collected,
        )

    def get_stats(self) -> EpisodeStatsResult:
        """获取 episode 统计。

        Returns:
            EpisodeStatsResult 统计结果。
        """
        return self.stats.get_stats()

    def close(self) -> None:
        """关闭环境。"""
        self.envs.close()
