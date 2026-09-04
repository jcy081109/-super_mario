import sys
sys.path.insert(0, '.')
from mario_rl.env_wrappers import make_env

print('测试多关卡环境创建（multi_worlds=None, multi_stages=None）...')
env = make_env(multi_level=True, multi_worlds=None, multi_stages=None, seed=42)
obs = env.reset()
print(f'  观测形状: {obs.shape}')
print(f'  动作空间: {env.action_space.n}')
obs, r, done, info = env.step(1)
xpos = info.get('x_pos', '?')
print(f'  step正常: reward={r}, x_pos={xpos}')
env.close()
print('多关卡环境创建成功！')
