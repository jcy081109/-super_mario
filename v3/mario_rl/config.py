"""
全局配置：环境、PPO 超参、流水线、训练、路径。
所有超参集中在此，便于调参与追溯。
训练入口通过命令行参数覆盖此处默认值。
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    """环境与预处理配置。

    Attributes:
        env_id: 默认关卡环境 ID（单关模式使用）。
        seed: 全局随机种子。
        frame_skip: 每 N 帧执行一次动作（动作重复 N 帧，奖励累加）。
        stack_size: 堆叠帧数，让智能体感知运动方向。
        image_size: 缩放后图像边长（正方形）。
    """
    env_id: str = "SuperMarioBros-1-1-v0"
    seed: int = 42
    frame_skip: int = 4
    stack_size: int = 4
    image_size: int = 84


@dataclass
class PPOConfig:
    """PPO 算法超参数。

    Attributes:
        learning_rate: Adam 优化器初始学习率。
        n_steps: 每个环境每轮收集的步数。
        batch_size: PPO 更新的小批量大小。
        n_epochs: 每轮数据更新的 epoch 数。
        gamma: 回报折扣因子。
        gae_lambda: GAE 优势估计的平滑参数。
        clip_range: PPO 策略比率裁剪范围 [1-clip, 1+clip]。
        ent_coef: 熵损失系数（鼓励探索）。
        vf_coef: 价值损失系数。
        max_grad_norm: 梯度裁剪最大范数。
        normalize_advantage: 是否对优势做批归一化。
    """
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


@dataclass
class PipelineConfig:
    """流水线配置（v2 核心特性）。

    双缓冲设计：两个 rollout buffer 交替使用，收集线程写 buffer_a，
    更新线程读 buffer_b，收集完交换。collect_model 为收集用快照，
    参数固定；train_model 为更新用模型，每轮同步到 collect_model。

    Attributes:
        enabled: 是否启用流水线（收集与更新并行）。
    """
    enabled: bool = True


@dataclass
class TrainConfig:
    """训练流程配置。

    Attributes:
        total_timesteps: 总训练步数。
        n_envs: 并行环境数量。
        save_freq: 每 N 步保存 checkpoint。
        eval_freq: 每 N 步评估一次。
        eval_episodes: 每次评估的局数。
        log_interval: 每 N 轮更新打印一次日志。
        use_amp: 是否使用混合精度训练（预留，暂未实现）。
    """
    total_timesteps: int = 10_000_000
    n_envs: int = 32
    save_freq: int = 200_000
    eval_freq: int = 200_000
    eval_episodes: int = 5
    log_interval: int = 10
    use_amp: bool = False


@dataclass
class PathConfig:
    """路径配置（基于项目根目录自动创建）。

    Attributes:
        project_root: 项目根目录（v2/）。
        checkpoint_dir: checkpoint 保存目录。
        log_dir: TensorBoard 日志目录。
        video_dir: 录制视频保存目录。
    """
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    video_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """初始化后自动创建目录。"""
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.log_dir = self.project_root / "logs"
        self.video_dir = self.project_root / "videos"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """总配置，聚合所有子配置。

    Attributes:
        env: 环境配置。
        ppo: PPO 超参配置。
        pipeline: 流水线配置。
        train: 训练流程配置。
        paths: 路径配置。
    """
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# 全局单例
config = Config()
