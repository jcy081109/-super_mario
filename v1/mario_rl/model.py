"""
自定义轻量 CNN 特征提取器。
SB3 内置 NatureCNN 为 3 层卷积 (32→64→64) + FC 512，已满足"2-3层卷积"要求。
若算力紧张，可切换为自定义轻量版 (16→32→32) + FC 256。
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .config import config


class CustomCNN(BaseFeaturesExtractor):
    """3 层轻量卷积网络，通道数和 FC 隐藏维可配置。

    输入: (4, 84, 84) 堆叠灰度帧
    输出: features_dim 维特征向量
    """

    def __init__(self, observation_space, features_dim: int = None, channels: tuple = None):
        cfg = config.model
        features_dim = features_dim or cfg.cnn_fc_hidden
        channels = channels or cfg.cnn_channels
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 堆叠帧数 = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, channels[0], kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 动态计算 flatten 后的维度
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def get_policy_kwargs():
    """根据 config 返回 PPO 的 policy_kwargs。
    use_custom_cnn=False → SB3 内置 NatureCNN（默认，稳定）
    use_custom_cnn=True  → 自定义轻量 CNN（更低算力）

    注意：normalize_images=False，因为 Wrapper 已将像素归一化到 [0,1] float32，
    SB3 不应再次归一化（否则会把 [0,1] 除以 255 变成接近 0）。
    """
    kwargs = {"normalize_images": False}
    if config.model.use_custom_cnn:
        kwargs.update({
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {
                "features_dim": config.model.cnn_fc_hidden,
                "channels": config.model.cnn_channels,
            },
        })
    # 使用 SB3 内置 NatureCNN（3 层卷积 32→64→64，FC 512）
    return kwargs
