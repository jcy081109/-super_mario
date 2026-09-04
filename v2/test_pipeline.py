"""v2 流水线模式快速测试"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import torch
from mario_rl.config import config
from mario_rl.utils import set_seed, get_device, count_parameters
from mario_rl.model import ActorCritic
from mario_rl.ppo import PPO
from mario_rl.collector import Collector, make_vec_envs
from mario_rl.pipeline import PipelinePPO

print('=== 流水线模式测试 ===')
device = get_device()
set_seed(42)

# 小配置快速测试
n_envs = 2
n_steps = 8
batch_size = 4
n_epochs = 2

print(f'配置: {n_envs}环境 × {n_steps}步 × batch{batch_size} × {n_epochs}轮')

# 创建环境
envs = make_vec_envs(n_envs=n_envs, seed=42, vec_env_type='dummy', world=1, stage=1)
obs_shape = envs.observation_space.shape
n_actions = envs.action_space.n

# 创建模型和PPO
model = ActorCritic(input_channels=obs_shape[0], n_actions=n_actions).to(device)
print(f'模型参数量: {count_parameters(model):,}')

ppo = PPO(model=model, device=device, lr=3e-4)

# 创建收集器
collector = Collector(
    model=model, envs=envs, n_envs=n_envs, n_steps=n_steps,
    obs_shape=obs_shape, device=device
)

# 创建流水线调度器
pipeline = PipelinePPO(
    train_model=model, ppo=ppo, collector=collector,
    n_envs=n_envs, n_steps=n_steps, obs_shape=obs_shape,
    device=device, gamma=0.99, gae_lambda=0.95,
    n_epochs=n_epochs, batch_size=batch_size, enabled=True
)

print()
print('运行 3 轮流水线收集+更新...')
for i in range(3):
    stats = pipeline.collect_one_iteration()
    print(f'  第{i+1}轮: 总步数={stats["total_timesteps"]}, '
          f'收集时间={stats["collect_time"]:.3f}s, '
          f'更新指标={stats.get("update_metrics") is not None}')

pipe_stats = pipeline.get_pipeline_stats()
print()
print('流水线统计:')
print(f'  总轮数: {pipe_stats["n_iterations"]}')
print(f'  总步数: {pipe_stats["total_timesteps"]:,}')
print(f'  平均收集时间: {pipe_stats["avg_collect_time"]:.4f}s')
print(f'  平均更新时间: {pipe_stats["avg_update_time"]:.4f}s')
print(f'  流水线启用: {pipe_stats["pipeline_enabled"]}')

pipeline.close()
print()
print('=== 流水线模式测试通过！===')
