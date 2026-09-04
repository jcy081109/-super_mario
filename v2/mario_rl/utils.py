"""
工具函数：随机种子、设备检测、日志、统计、学习率调度、模型评估。
所有通用工具集中在此，避免各模块重复实现。
"""
import glob
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .model import ActorCritic

from .config import config


# ─── 随机种子与设备 ───────────────────────────────────────────────

def set_seed(seed: Optional[int] = None) -> None:
    """固定所有随机种子，保证实验可复现。

    设置 Python random、NumPy、PyTorch（CPU+CUDA）的种子，
    并启用 cuDNN 确定性模式。

    Args:
        seed: 随机种子，为 None 时使用 config.env.seed。
    """
    seed = seed if seed is not None else config.env.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """检测可用计算设备，优先 CUDA，否则 CPU。

    Returns:
        torch.device: 可用设备。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 滑动窗口统计 ──────────────────────────────────────────────────

class SmoothedValue:
    """滑动窗口统计器，用于训练指标的平滑展示。

    维护一个固定长度的 deque，提供均值、中位数、全局平均等统计量。

    Attributes:
        deque: 存储最近 window_size 个值的双端队列。
        total: 所有历史值的总和。
        count: 所有历史值的计数。
    """

    def __init__(self, window_size: int = 20) -> None:
        self.deque: deque = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0

    def update(self, value: float) -> None:
        """添加一个新值。

        Args:
            value: 要记录的数值。
        """
        self.deque.append(value)
        self.total += value
        self.count += 1

    @property
    def median(self) -> float:
        """滑动窗口中位数。"""
        return float(np.median(self.deque)) if self.deque else 0.0

    @property
    def mean(self) -> float:
        """滑动窗口均值。"""
        return float(np.mean(self.deque)) if self.deque else 0.0

    @property
    def global_avg(self) -> float:
        """全部历史值的全局平均。"""
        return self.total / max(self.count, 1)

    @property
    def max(self) -> float:
        """滑动窗口最大值。"""
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        """最近一次添加的值。"""
        return self.deque[-1] if self.deque else 0.0


# ─── 训练日志 ──────────────────────────────────────────────────────

class TrainLogger:
    """训练日志记录器，支持终端打印和 TensorBoard。

    自动在 log_dir 下创建 run_1/、run_2/ 等子目录，
    避免多次训练的日志混在一起。

    Attributes:
        use_tensorboard: 是否启用 TensorBoard。
        writer: TensorBoard SummaryWriter 实例。
        run_dir: 当前训练的日志目录。
        metrics: 指标名 → SmoothedValue 的映射。
        start_time: 日志器创建时间。
    """

    def __init__(self, log_dir: str, use_tensorboard: bool = True) -> None:
        self.use_tensorboard = use_tensorboard
        self.writer = None

        # 自动创建带序号的子目录（run_1, run_2, ...）
        os.makedirs(log_dir, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(log_dir, "run_*")))
        next_num = len(existing) + 1
        self.run_dir = os.path.join(log_dir, f"run_{next_num}")
        os.makedirs(self.run_dir, exist_ok=True)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.run_dir)
            except ImportError:
                print("[警告] tensorboard 未安装，仅终端打印")
                self.use_tensorboard = False

        self.metrics: dict[str, SmoothedValue] = {}
        self.start_time = time.time()

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """记录一个标量指标。

        Args:
            tag: 指标名称（如 "rollout/ep_rew_mean"）。
            value: 指标数值。
            step: 训练步数。
        """
        if tag not in self.metrics:
            self.metrics[tag] = SmoothedValue()
        self.metrics[tag].update(float(value))
        if self.writer:
            self.writer.add_scalar(tag, float(value), step)

    def get(self, tag: str) -> float:
        """获取某个指标的滑动平均。

        Args:
            tag: 指标名称。

        Returns:
            滑动平均值，指标不存在时返回 0.0。
        """
        if tag in self.metrics:
            return self.metrics[tag].mean
        return 0.0

    def elapsed(self) -> float:
        """已用时间（秒）。"""
        return time.time() - self.start_time

    def close(self) -> None:
        """关闭 TensorBoard writer。"""
        if self.writer:
            self.writer.close()


# ─── 模型统计 ──────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数量。

    Args:
        model: PyTorch 模型。

    Returns:
        可训练参数总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """计算价值网络的解释方差。

    公式：1 - Var(y_true - y_pred) / Var(y_true)
    - 1.0 表示完美预测
    - 0.0 表示不如预测均值
    - 负数表示预测很差

    Args:
        y_pred: 预测值（价值网络输出）。
        y_true: 真实值（GAE 回报）。

    Returns:
        解释方差，范围 [-∞, 1.0]。
    """
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return 1.0 - np.var(y_true - y_pred) / var_y


# ─── 学习率调度 ────────────────────────────────────────────────────

def make_lr_schedule(
    initial_lr: float,
    lr_min: float = 0.0,
    schedule: str = "linear",
) -> Callable[[float], float]:
    """创建学习率调度函数。

    Args:
        initial_lr: 初始学习率。
        lr_min: 学习率下限（线性衰减时不低于此值）。
        schedule: 调度类型，"constant" 或 "linear"。

    Returns:
        接受 progress（1.0=开始，0.0=结束）返回学习率的函数。
    """
    if schedule == "constant":
        return lambda progress: initial_lr

    def linear_lr(progress: float) -> float:
        """线性衰减学习率。

        Args:
            progress: 剩余训练进度，1.0 表示刚开始，0.0 表示结束。

        Returns:
            当前学习率，不低于 lr_min。
        """
        return max(initial_lr * progress, lr_min)

    return linear_lr


# ─── 奖励归一化 ─────────────────────────────────────────────────────

class RunningMeanStd:
    """在线均值/方差统计（Welford's algorithm）。

    用于奖励归一化，维护一个 running mean 和 variance，
    随新数据不断更新，不需要存储所有历史数据。

    Attributes:
        mean: 当前均值。
        var: 当前方差。
        count: 累计样本数。
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """用新数据更新统计量。

        Args:
            x: 新数据，形状与初始化时的 shape 一致（或可广播）。
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """用批量统计量更新总体统计量（并行方差合并公式）。"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RewardNormalizer:
    """奖励归一化器（SB3 VecNormalize norm_reward 风格）。

    维护一个 discounted return 的 running mean/variance，
    用其标准差对原始奖励做缩放：normalized_reward = reward / (std + eps)。

    这种方式的好处：
    - 不需要减均值（保持奖励的正负方向）
    - 自动适应不同关卡/阶段的奖励量级
    - 稳定 PPO 的 advantage 估计

    Args:
        gamma: 折扣因子，用于计算 discounted return。
        epsilon: 数值稳定性常数，防止除零。
        clip: 归一化后奖励的裁剪范围（±clip），None 表示不裁剪。

    Attributes:
        ret_rms: discounted return 的 RunningMeanStd。
        returns: 各环境当前的 discounted return（用于累计）。
    """

    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: Optional[float] = 10.0,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.ret_rms = RunningMeanStd()
        self.returns: Optional[np.ndarray] = None

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """归一化一批奖励。

        Args:
            rewards: (n_envs,) 原始奖励。
            dones: (n_envs,) 是否结束（结束时重置该环境的 discounted return）。

        Returns:
            (n_envs,) 归一化后的奖励。
        """
        if self.returns is None:
            self.returns = np.zeros_like(rewards, dtype=np.float64)

        # 更新 discounted return
        self.returns = self.returns * self.gamma + rewards
        # 用 discounted return 更新 running statistics
        self.ret_rms.update(self.returns)
        # 归一化：只除以标准差，不减均值（保持奖励方向）
        std = np.sqrt(self.ret_rms.var + self.epsilon)
        normalized_rewards = rewards / std
        # 裁剪
        if self.clip is not None:
            normalized_rewards = np.clip(normalized_rewards, -self.clip, self.clip)
        # 结束的环境重置 discounted return
        self.returns[dones] = 0.0
        return normalized_rewards.astype(np.float32)

    def reset(self) -> None:
        """重置归一化器状态（新训练开始时调用）。"""
        self.ret_rms = RunningMeanStd()
        self.returns = None


# ─── 模型评估 ──────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """模型评估结果。

    Attributes:
        mean_reward: 平均 episode 奖励。
        std_reward: 奖励标准差。
        mean_length: 平均 episode 长度。
        mean_x_pos: 平均最终 x 坐标。
        win_rate: 通关率（0.0~1.0）。
        n_episodes: 评估局数。
    """
    mean_reward: float
    std_reward: float
    mean_length: float
    mean_x_pos: float
    win_rate: float
    n_episodes: int


def evaluate(
    model: "ActorCritic",
    envs,
    n_episodes: int = 5,
    device: Optional[torch.device] = None,
) -> EvalResult:
    """评估模型在给定环境上的表现。

    使用确定性策略（取概率最大的动作），运行 n_episodes 局，
    统计平均奖励、长度、x 坐标和通关率。

    Args:
        model: 要评估的 ActorCritic 模型。
        envs: 向量化环境（VecEnv）。
        n_episodes: 评估局数。
        device: 计算设备，为 None 时自动检测。

    Returns:
        EvalResult 评估结果。
    """
    if device is None:
        device = get_device()

    model.eval()
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_x_pos: list[float] = []
    episode_wins: list[float] = []

    obs = envs.reset()
    current_rewards = np.zeros(envs.num_envs)
    current_lengths = np.zeros(envs.num_envs, dtype=int)

    with torch.no_grad():
        while len(episode_rewards) < n_episodes:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions, _, _, _ = model.get_action(obs_tensor, deterministic=True)
            obs, rewards, dones, infos = envs.step(actions.cpu().numpy())

            current_rewards += rewards
            current_lengths += 1

            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(float(current_rewards[i]))
                    episode_lengths.append(int(current_lengths[i]))
                    if "x_pos" in infos[i]:
                        episode_x_pos.append(float(infos[i]["x_pos"]))
                    if "flag_get" in infos[i]:
                        episode_wins.append(1.0 if infos[i]["flag_get"] else 0.0)
                    current_rewards[i] = 0.0
                    current_lengths[i] = 0

    return EvalResult(
        mean_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        std_reward=float(np.std(episode_rewards)) if episode_rewards else 0.0,
        mean_length=float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        mean_x_pos=float(np.mean(episode_x_pos)) if episode_x_pos else 0.0,
        win_rate=float(np.mean(episode_wins)) if episode_wins else 0.0,
        n_episodes=len(episode_rewards),
    )
