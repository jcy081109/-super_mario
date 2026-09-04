"""v2 集成测试：验证所有模块能正常导入和运行"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

print('=== 模块导入测试 ===')
from mario_rl.config import config
from mario_rl.utils import set_seed, get_device, count_parameters
from mario_rl.model import ActorCritic
from mario_rl.ppo import PPO, RolloutBuffer
from mario_rl.collector import Collector, make_vec_envs
from mario_rl.pipeline import PipelinePPO
from mario_rl.env_wrappers import make_env
print('所有模块导入成功！')

print()
print('=== 环境创建测试（2环境，dummy）===')
device = get_device()
set_seed(42)
envs = make_vec_envs(n_envs=2, seed=42, vec_env_type='dummy', world=1, stage=1)
obs = envs.reset()
print(f'观测形状: {obs.shape}, dtype: {obs.dtype}')
print(f'动作空间: {envs.action_space.n} 个动作')

print()
print('=== 模型创建测试 ===')
model = ActorCritic(input_channels=obs.shape[1], n_actions=envs.action_space.n).to(device)
print(f'模型参数量: {count_parameters(model):,}')

print()
print('=== 单步推理测试 ===')
import torch
obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
actions, log_probs, values, entropy = model.get_action(obs_tensor)
print(f'动作: {actions.cpu().numpy()}')
print(f'价值: {values.detach().cpu().numpy()}')
print(f'熵: {entropy.detach().cpu().numpy()}')

print()
print('=== 环境 step 测试 ===')
next_obs, rewards, dones, infos = envs.step(actions.cpu().numpy())
print(f'奖励: {rewards}, done: {dones}')

print()
print('=== RolloutBuffer 测试 ===')
buffer = RolloutBuffer(n_envs=2, n_steps=4, obs_shape=obs.shape[1:], device=device)
for i in range(4):
    buffer.add(obs=obs, action=actions.cpu().numpy(), reward=rewards,
               done=dones, value=values.detach().cpu().numpy(), log_prob=log_probs.detach().cpu().numpy())
    obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    actions, log_probs, values, _ = model.get_action(obs_tensor)
    next_obs, rewards, dones, infos = envs.step(actions.cpu().numpy())
    obs = next_obs
print(f'Buffer 填充完成，full={buffer.full}')

last_values = values.detach().cpu().numpy()
advantages, returns = buffer.compute_gae(last_values, gamma=0.99, gae_lambda=0.95)
print(f'GAE 计算完成，advantages shape={advantages.shape}')

print()
print('=== PPO 更新测试 ===')
ppo = PPO(model=model, device=device, lr=3e-4)
metrics = ppo.update(buffer=buffer, last_values=last_values, gamma=0.99,
                      gae_lambda=0.95, n_epochs=2, batch_size=4)
print('更新完成！')
print(f'  policy_loss: {metrics["policy_loss"]:.6f}')
print(f'  value_loss: {metrics["value_loss"]:.6f}')
print(f'  entropy: {metrics["entropy"]:.4f}')
print(f'  approx_kl: {metrics["approx_kl"]:.6f}')
print(f'  clip_fraction: {metrics["clip_fraction"]:.4f}')
print(f'  explained_variance: {metrics["explained_variance"]:.4f}')

envs.close()
print()
print('=== 所有测试通过！v2 代码可以正常运行 ===')
