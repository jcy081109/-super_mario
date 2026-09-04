"""
模型定义：NatureCNN / LargeCNN 特征提取器 + ActorCritic 策略网络。

NatureCNN 与 SB3 结构一致（168万参数），用于对比；
LargeCNN 通道翻倍+更深（424万参数），用于多关卡训练增加容量。

两种模型共享相同的接口：forward / get_action / get_value / evaluate_actions。
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ─── 特征提取器 ────────────────────────────────────────────────────

class NatureCNN(nn.Module):
    """DeepMind Nature CNN 特征提取器。

    结构：3层卷积 (32→64→64) + 全连接 (3136→512)
    参数量：约 168 万

    Args:
        input_channels: 输入观测通道数（帧堆叠数，默认4）。

    Input:
        x: (batch, input_channels, 84, 84) float32 [0,1]

    Output:
        (batch, 512) 特征向量
    """

    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 计算 flatten 后的维度: 64 * 7 * 7 = 3136
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            n_flatten = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，提取特征。

        Args:
            x: (batch, input_channels, 84, 84) 观测。

        Returns:
            (batch, 512) 特征向量。
        """
        return self.fc(self.cnn(x))


class LargeCNN(nn.Module):
    """更大的 CNN 特征提取器，用于多关卡训练增加模型容量。

    结构：4层卷积 (64→128→128→128) + 两层全连接 (6272→1024→512)
    参数量：约 424 万（NatureCNN 的 2.5 倍）

    Args:
        input_channels: 输入观测通道数（帧堆叠数，默认4）。

    Input:
        x: (batch, input_channels, 84, 84) float32 [0,1]

    Output:
        (batch, 512) 特征向量
    """

    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 计算 flatten 后的维度: 128 * 5 * 5 = 3200
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            n_flatten = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，提取特征。

        Args:
            x: (batch, input_channels, 84, 84) 观测。

        Returns:
            (batch, 512) 特征向量。
        """
        return self.fc(self.cnn(x))


# ─── 演员-评论家网络 ────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """演员-评论家网络，共享特征提取层。

    策略头: 输出动作 logits（离散动作空间）
    价值头: 输出状态价值 V(s)

    Args:
        input_channels: 输入观测通道数（帧堆叠数）。
        n_actions: 动作空间大小。
        model_size: 模型大小，"nature"（168万，SB3兼容）或 "large"（424万）。

    Attributes:
        model_size: 模型大小标识。
        features_extractor: 特征提取网络（NatureCNN 或 LargeCNN）。
        actor: 策略头，512 → n_actions。
        critic: 价值头，512 → 1。
    """

    def __init__(
        self,
        input_channels: int = 4,
        n_actions: int = 7,
        model_size: str = "nature",
    ) -> None:
        super().__init__()
        self.model_size = model_size
        if model_size == "large":
            self.features_extractor = LargeCNN(input_channels)
        else:
            self.features_extractor = NatureCNN(input_channels)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        # 正交初始化（PPO 标准做法）
        self._init_weights()

    def _init_weights(self) -> None:
        """正交初始化所有权重，策略头和价值头使用特殊增益。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # 策略头用小增益（初始策略接近均匀分布）
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        # 价值头用增益1
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """前向传播，返回动作分布和状态价值。

        Args:
            obs: (batch, input_channels, 84, 84) 观测。

        Returns:
            dist: Categorical 动作分布。
            value: (batch, 1) 状态价值。
        """
        features = self.features_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        dist = Categorical(logits=action_logits)
        return dist, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作（收集阶段使用）。

        Args:
            obs: (batch, input_channels, 84, 84) 观测。
            deterministic: 是否使用确定性策略（取概率最大动作）。

        Returns:
            actions: (batch,) 采样的动作。
            log_probs: (batch,) 动作的对数概率。
            values: (batch,) 状态价值。
            entropy: (batch,) 动作分布熵。
        """
        dist, value = self.forward(obs)
        if deterministic:
            actions = torch.argmax(dist.probs, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, value.squeeze(-1), entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """只返回状态价值（收集结束时用于 GAE bootstrap）。

        Args:
            obs: (batch, input_channels, 84, 84) 观测。

        Returns:
            (batch,) 状态价值。
        """
        features = self.features_extractor(obs)
        return self.critic(features).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定动作的对数概率、价值和熵（PPO 更新阶段使用）。

        Args:
            obs: (batch, input_channels, 84, 84) 观测。
            actions: (batch,) 要评估的动作。

        Returns:
            log_probs: (batch,) 动作的对数概率。
            values: (batch,) 状态价值。
            entropy: (batch,) 动作分布熵。
        """
        dist, value = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy

    def save(self, path: str) -> None:
        """保存模型权重。

        Args:
            path: 保存路径。
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device) -> None:
        """加载模型权重。

        Args:
            path: 权重文件路径。
            device: 加载到的设备。
        """
        self.load_state_dict(torch.load(path, map_location=device))
