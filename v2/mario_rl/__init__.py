"""
Mario RL v2 - 从零实现的流水线 PPO（不依赖 SB3 算法）。

核心特性：
- 双缓冲流水线收集+更新，CPU/GPU 最大化利用
- NatureCNN / LargeCNN 两种模型架构
- 自适应熵系数、奖励塑形、学习率调度
- 多关卡训练（训练集/测试集分离）
- 完整的类型注解和接口定义

模块结构：
- config: 全局配置
- utils: 工具函数（种子/设备/日志/学习率/评估）
- model: 模型定义（NatureCNN/LargeCNN/ActorCritic）
- env_wrappers: 环境 Wrapper 管线与工厂
- ppo: PPO 核心算法（RolloutBuffer/GAE/更新）
- collector: 数据收集器
- pipeline: 流水线调度器
- train: 训练入口
- watch: 观看/评估入口
"""
__version__ = "2.1.0"
