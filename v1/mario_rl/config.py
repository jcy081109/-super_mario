"""
全局配置：环境、PPO 超参、训练、路径。
所有超参集中在此，便于调参与追溯。
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    """环境与预处理配置"""
    env_id: str = "SuperMarioBros-1-1-v0"
    seed: int = 42
    frame_skip: int = 4          # 每 N 帧执行一次动作（减少计算量）
    stack_size: int = 4          # 堆叠帧数（体现运动）
    image_size: int = 84         # 缩放后图像边长
    # SIMPLE_MOVEMENT: 7 个动作 [NOOP, right, right+A, right+B, right+A+B, A, left]


@dataclass
class PPOConfig:
    """PPO 超参（基于 SB3 默认值，适配低算力）"""
    learning_rate: float = 3e-4
    n_steps: int = 1024           # 每次更新收集的步数；笔记本可降为 512
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01        # 熵系数，鼓励探索
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # 设备自动检测：有 GPU 用 cuda，否则 cpu
    device: str = "auto"


@dataclass
class ModelConfig:
    """模型结构配置"""
    # True: 使用自定义轻量 CNN；False: 使用 SB3 内置 NatureCNN（3层卷积 32→64→64）
    use_custom_cnn: bool = False
    # 自定义 CNN 通道数（仅 use_custom_cnn=True 时生效）
    cnn_channels: tuple = (16, 32, 32)
    cnn_fc_hidden: int = 256


@dataclass
class TrainConfig:
    """训练配置"""
    total_timesteps: int = 1_000_000   # 总训练步数（先 100 万看效果）
    save_freq: int = 50_000             # 每 N 步保存 checkpoint
    log_interval: int = 10              # 每 N 次更新打印日志
    eval_freq: int = 100_000            # 每 N 步评估一次
    eval_episodes: int = 5


@dataclass
class PathConfig:
    """路径配置（基于项目根目录）"""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    video_dir: Path = field(init=False)

    def __post_init__(self):
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.log_dir = self.project_root / "logs"
        self.video_dir = self.project_root / "videos"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """总配置"""
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# 全局单例
config = Config()
