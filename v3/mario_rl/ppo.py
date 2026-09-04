"""
PPO 核心算法：RolloutBuffer + GAE + PPO Clip 损失 + 更新逻辑。
从零实现，不依赖 SB3 算法部分（仅用其 VecEnv 环境封装）。

算法流程：
  1. 收集 n_steps × n_envs 步数据到 RolloutBuffer
  2. 计算 GAE 优势函数和回报
  3. 优势归一化
  4. 数据打乱，分小批量，多 epoch 更新
  5. 损失 = PPO Clip 策略损失 + vf_coef × 价值损失 + ent_coef × 熵损失
  6. 梯度裁剪 + Adam 优化器更新
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .config import config
from .model import ActorCritic
from .utils import explained_variance


# ─── 更新指标 ───────────────────────────────────────────────────────

@dataclass
class UpdateMetrics:
    """PPO 一次更新的训练指标。

    Attributes:
        policy_loss: 策略损失（PPO Clip）。
        value_loss: 价值损失（MSE）。
        entropy: 策略熵（越高探索越多）。
        approx_kl: 近似 KL 散度（新旧策略差异）。
        clip_fraction: 被裁剪的比率（0~1，越高说明更新越激进）。
        loss: 总损失。
        explained_variance: 价值网络解释方差（1.0=完美，0=均值，负=很差）。
        n_updates: 累计梯度更新次数。
        update_time: 更新耗时（秒），由 pipeline 填充。
    """
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    loss: float
    explained_variance: float
    n_updates: int
    update_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转换为字典（兼容旧代码）。"""
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "loss": self.loss,
            "explained_variance": self.explained_variance,
            "n_updates": self.n_updates,
            "update_time": self.update_time,
        }


# ─── 滚动缓冲区 ─────────────────────────────────────────────────────

class RolloutBuffer:
    """滚动缓冲区，存储一轮收集的数据。

    支持多环境：每个环境独立存储，最后合并。
    数据按 (n_steps, n_envs, ...) 布局存储。

    Args:
        n_envs: 并行环境数。
        n_steps: 每环境收集步数。
        obs_shape: 观测空间形状（如 (4, 84, 84)）。
        device: 数据转换为 tensor 时使用的设备。

    Attributes:
        observations: (n_steps, n_envs, *obs_shape) 观测。
        actions: (n_steps, n_envs) 动作。
        rewards: (n_steps, n_envs) 奖励。
        dones: (n_steps, n_envs) 是否结束。
        values: (n_steps, n_envs) 状态价值。
        log_probs: (n_steps, n_envs) 动作对数概率。
        last_values: (n_envs,) 收集结束后最后一个状态的价值（用于 GAE bootstrap）。
        pos: 当前写入位置。
        full: 缓冲区是否已满。
    """

    def __init__(
        self,
        n_envs: int,
        n_steps: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
    ) -> None:
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.device = device
        self.reset()

    def reset(self) -> None:
        """清空缓冲区，重置写入位置。"""
        self.observations = np.zeros((self.n_steps, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.last_values: Optional[np.ndarray] = None
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """添加一步数据（所有环境同时）。

        Args:
            obs: (n_envs, *obs_shape) 当前观测。
            action: (n_envs,) 执行的动作。
            reward: (n_envs,) 获得的奖励。
            done: (n_envs,) 是否结束。
            value: (n_envs,) 状态价值。
            log_prob: (n_envs,) 动作对数概率。
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 GAE 优势函数和回报。

        公式：
          δ_t = r_t + γ × V(s_{t+1}) × (1-done_t) - V(s_t)
          A_t = δ_t + γλ × (1-done_t) × A_{t+1}
          returns_t = A_t + V(s_t)

        Args:
            last_values: (n_envs,) 最后一个状态的价值（用于 bootstrap）。
            gamma: 折扣因子。
            gae_lambda: GAE 平滑参数。

        Returns:
            advantages: (n_steps, n_envs) 优势函数。
            returns: (n_steps, n_envs) 回报（用于价值损失目标）。
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0

        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            advantages[step] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + self.values
        return advantages, returns

    def get(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """获取扁平化的训练数据（所有环境合并）。

        Args:
            advantages: (n_steps, n_envs) 优势函数。
            returns: (n_steps, n_envs) 回报。

        Returns:
            字典，每个值都是 (n_steps * n_envs, ...) 的 tensor：
            - observations: 观测
            - actions: 动作
            - old_values: 旧价值（用于价值损失裁剪，当前未使用）
            - old_log_probs: 旧对数概率（用于 PPO 比率）
            - advantages: 优势
            - returns: 回报
        """
        # 展平: (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        batch_obs = self.observations.reshape(-1, *self.obs_shape)
        batch_actions = self.actions.reshape(-1)
        batch_values = self.values.reshape(-1)
        batch_log_probs = self.log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)

        return {
            "observations": torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(batch_actions, dtype=torch.long, device=self.device),
            "old_values": torch.as_tensor(batch_values, dtype=torch.float32, device=self.device),
            "old_log_probs": torch.as_tensor(batch_log_probs, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(batch_advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(batch_returns, dtype=torch.float32, device=self.device),
        }


# ─── PPO 算法 ───────────────────────────────────────────────────────

class PPO:
    """PPO 算法类，负责模型管理、优化器、更新逻辑。

    Args:
        model: ActorCritic 模型。
        device: 计算设备。
        lr: Adam 学习率。
        clip_range: PPO 策略比率裁剪范围。
        ent_coef: 熵损失系数。
        vf_coef: 价值损失系数。
        max_grad_norm: 梯度裁剪最大范数。
        normalize_advantage: 是否对优势做批归一化。

    Attributes:
        model: ActorCritic 模型。
        optimizer: Adam 优化器。
        total_updates: 累计梯度更新次数。
    """

    def __init__(
        self,
        model: ActorCritic,
        device: torch.device,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        self.optimizer = Adam(model.parameters(), lr=lr, eps=1e-5)
        self.total_updates = 0

    def update(
        self,
        buffer: RolloutBuffer,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
        n_epochs: int,
        batch_size: int,
    ) -> UpdateMetrics:
        """用缓冲区数据更新策略。

        流程：计算 GAE → 优势归一化 → 数据打乱 → 多 epoch 小批量更新。

        Args:
            buffer: 收集好数据的 RolloutBuffer。
            last_values: (n_envs,) 最后一个状态的价值（用于 GAE bootstrap）。
            gamma: 折扣因子。
            gae_lambda: GAE 平滑参数。
            n_epochs: 每轮数据更新的 epoch 数。
            batch_size: 小批量大小。

        Returns:
            UpdateMetrics 更新指标。
        """
        # 1. 计算 GAE
        advantages, returns = buffer.compute_gae(last_values, gamma, gae_lambda)

        # 2. 归一化优势
        if self.normalize_advantage:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # 3. 获取扁平化数据
        batch_data = buffer.get(advantages, returns)
        n_samples = batch_data["observations"].shape[0]

        # 4. 多轮更新
        metrics_lists: Dict[str, list] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "loss": [],
        }

        for epoch in range(n_epochs):
            # 打乱索引
            indices = torch.randperm(n_samples, device=self.device)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]

                # 取小批量
                obs = batch_data["observations"][idx]
                actions = batch_data["actions"][idx]
                old_log_probs = batch_data["old_log_probs"][idx]
                old_values = batch_data["old_values"][idx]
                adv = batch_data["advantages"][idx]
                ret = batch_data["returns"][idx]

                # 重新评估动作
                new_log_probs, new_values, entropy = self.model.evaluate_actions(obs, actions)

                # 比率 r_t = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # 策略损失（PPO Clip）
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失（MSE）
                value_loss = nn.functional.mse_loss(new_values, ret)

                # 熵损失（鼓励探索）
                entropy_loss = -entropy.mean()

                # 总损失
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # 反向传播 + 梯度裁剪 + 优化器更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.total_updates += 1

                # 记录指标
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                metrics_lists["policy_loss"].append(policy_loss.item())
                metrics_lists["value_loss"].append(value_loss.item())
                metrics_lists["entropy"].append(entropy.mean().item())
                metrics_lists["approx_kl"].append(approx_kl.item())
                metrics_lists["clip_fraction"].append(clip_fraction.item())
                metrics_lists["loss"].append(loss.item())

        # 计算价值网络解释方差
        with torch.no_grad():
            pred_values = batch_data["old_values"].cpu().numpy()
            true_returns = batch_data["returns"].cpu().numpy()
            ev = explained_variance(pred_values, true_returns)

        return UpdateMetrics(
            policy_loss=float(np.mean(metrics_lists["policy_loss"])),
            value_loss=float(np.mean(metrics_lists["value_loss"])),
            entropy=float(np.mean(metrics_lists["entropy"])),
            approx_kl=float(np.mean(metrics_lists["approx_kl"])),
            clip_fraction=float(np.mean(metrics_lists["clip_fraction"])),
            loss=float(np.mean(metrics_lists["loss"])),
            explained_variance=float(ev),
            n_updates=self.total_updates,
        )

    def save(self, path: str) -> None:
        """保存模型和优化器状态。

        Args:
            path: 保存路径。
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }, path)

    def load(self, path: str) -> None:
        """加载模型和优化器状态。

        Args:
            path: checkpoint 文件路径。
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
