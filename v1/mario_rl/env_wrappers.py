"""
环境 Wrapper 管线：
  原始帧 (240,256,3) uint8
  → SkipFrame(4)          每 4 帧执行一次动作
  → GrayScaleObservation   RGB → 灰度 (240,256) uint8
  → ResizeObservation(84)  缩放 (84,84) uint8
  → FrameStack(4)          堆叠 4 帧 (4,84,84) uint8
  → NormalizeObservation    像素 /255 → [0,1] float32

最终观测空间 (4, 84, 84) float32，符合 SB3 CnnPolicy 的 CHW 格式。
注意：归一化必须在 FrameStack 之后，否则 FrameStack 会强制转回 uint8。
"""
from collections import deque

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor

from .config import config


class SkipFrame(gym.Wrapper):
    """每 skip 帧执行一次动作，返回最后一帧的观测。
    减少环境交互次数，加速训练。"""

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    """用 opencv 将观测缩放到 shape×shape，保持灰度单通道。"""

    def __init__(self, env, shape: int = 84):
        super().__init__(env)
        self.shape = (shape, shape)
        # 灰度图无通道维，观测空间为 (shape, shape)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)


class NormalizeObservation(gym.ObservationWrapper):
    """像素归一化 [0,255] uint8 → [0,1] float32。
    必须正确更新 observation_space 的 dtype，否则后续 FrameStack 会强制转回 uint8。"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return (observation / 255.0).astype(np.float32)


class FrameStack(gym.Wrapper):
    """堆叠最近 num_stack 帧，输出 (num_stack, H, W)。
    让智能体感知运动方向（起跳、下落、敌人移动）。"""

    def __init__(self, env, num_stack: int = 4):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.array(self._frames, dtype=self.observation_space.dtype)


class RewardShaping(gym.RewardWrapper):
    """可选奖励塑形：时间惩罚 + 跳跃惩罚。

    - time_penalty：每步扣一点奖励，鼓励快速前进，避免原地磨蹭
    - jump_penalty：每次执行跳跃动作扣一点奖励， discouraging 过度跳跃
      （模型容易学会"一直右+跳"，导致遇到坑时跳跃时机不对掉下去）
      惩罚不宜过大（建议 0.01~0.05），否则模型会完全不跳，无法越过障碍

    SIMPLE_MOVEMENT 7 动作中包含跳跃的是：
      2=right+A, 4=right+A+B, 5=A（原地跳）
    不包含跳跃的是：0=NOOP, 1=right, 3=right+B, 6=left
    """

    # 包含跳跃按键 A 的动作索引（SIMPLE_MOVEMENT）
    JUMP_ACTIONS = {2, 4, 5}

    def __init__(
        self,
        env,
        time_penalty: float = 0.0,
        jump_penalty: float = 0.0,
        shaping: bool = False,
    ):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.jump_penalty = jump_penalty
        self.shaping = shaping
        self._last_action = None

    def step(self, action):
        """重写 step 以获取 action，用于跳跃惩罚判断。"""
        self._last_action = action
        return super().step(action)

    def reward(self, reward):
        if self.shaping:
            # 时间惩罚：每步都扣
            reward = reward - self.time_penalty
            # 跳跃惩罚：仅当执行了跳跃动作时扣
            if self._last_action in self.JUMP_ACTIONS:
                reward = reward - self.jump_penalty
        return reward


class RandomLevelEnv(gym.Env):
    """随机关卡环境：每次 reset 时随机选择一个关卡，强制模型学通用策略而非记忆单关布局。

    原理：内部维护当前环境实例，每次 reset 时随机选 world/stage 并重建环境。
    所有 step/render/seed/close 调用转发到当前环境。
    适用于多关卡联合训练，提升模型泛化能力。
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, worlds=(1,), stages=(1, 2, 3, 4), levels=None, seed: int = None):
        """
        Args:
            worlds: 可选的 world 列表，如 (1, 2) 表示 1-1~2-4（与 stages 笛卡尔积）
            stages: 可选的 stage 列表，如 (1,2,3,4)
            levels: 直接指定关卡列表，如 ((1,1), (1,2), (2,4))。优先级高于 worlds/stages。
                    用于灵活指定训练集/测试集分离，如7关训练+1关测试。
            seed: 随机关卡选择的随机种子
        """
        self.worlds = tuple(worlds)
        self.stages = tuple(stages)
        if levels is not None:
            self.levels = tuple(tuple(lv) for lv in levels)
        else:
            self.levels = tuple((w, s) for w in worlds for s in stages)
        self._rng = np.random.RandomState(seed)
        self.current_env = None
        self.current_world = None
        self.current_stage = None
        self.observation_space = None
        self.action_space = None
        self._make_env()

    def _make_env(self):
        """随机选一个关卡并创建原始环境 + JoypadSpace 包装。"""
        idx = self._rng.randint(len(self.levels))
        self.current_world, self.current_stage = self.levels[idx]
        env_id = f"SuperMarioBros-{self.current_world}-{self.current_stage}-v0"
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception:
                pass
        raw_env = gym_super_mario_bros.make(env_id)
        self.current_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space

    def reset(self):
        """每次 reset 都随机选新关卡并重建环境。"""
        self._make_env()
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode="human"):
        return self.current_env.render(mode)

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)
        return self.current_env.seed(seed)

    def close(self):
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception:
                pass

    def __str__(self):
        return f"RandomLevelEnv(current={self.current_world}-{self.current_stage}, worlds={self.worlds}, stages={self.stages})"


def make_env(
    env_id: str = None,
    seed: int = None,
    frame_skip: int = None,
    stack_size: int = None,
    image_size: int = None,
    shaping: bool = False,
    time_penalty: float = 0.0,
    jump_penalty: float = 0.0,
    world: int = None,
    stage: int = None,
    multi_level: bool = False,
    multi_worlds: tuple = (1,),
    multi_stages: tuple = (1, 2, 3, 4),
    levels: tuple = None,
):
    """创建并组装完整的马里奥环境 Wrapper 管线。
    参数为 None 时使用 config 中的默认值。
    world/stage 指定单关卡（优先级高于 env_id）。
    multi_level=True 时开启随机关卡模式，每次 reset 随机选关卡，强制模型学通用策略。
    levels 直接指定关卡列表（如 ((1,1),(1,2),(2,4))），优先级高于 multi_worlds/multi_stages。"""
    cfg = config.env
    seed = seed if seed is not None else cfg.seed
    frame_skip = frame_skip if frame_skip is not None else cfg.frame_skip
    stack_size = stack_size if stack_size is not None else cfg.stack_size
    image_size = image_size if image_size is not None else cfg.image_size

    # 1. 创建原始环境 + 动作空间压缩（SIMPLE_MOVEMENT 7 动作）
    if multi_level:
        # 多关卡模式：RandomLevelEnv 内部已包含 JoypadSpace，每次 reset 随机选关卡
        env = RandomLevelEnv(worlds=multi_worlds, stages=multi_stages, levels=levels, seed=seed)
    else:
        # 单关卡模式
        if world is not None and stage is not None:
            env_id = f"SuperMarioBros-{world}-{stage}-v0"
        else:
            env_id = env_id or cfg.env_id
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env.seed(seed)

    # 2. 帧跳过
    env = SkipFrame(env, skip=frame_skip)

    # 3. 灰度化（keep_dim=False → (H, W)）
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)

    # 4. 缩放（opencv，(84, 84) uint8）
    env = ResizeObservation(env, shape=image_size)

    # 5. 可选奖励塑形（在 uint8 阶段即可，不影响观测）
    if shaping:
        env = RewardShaping(
            env,
            time_penalty=time_penalty,
            jump_penalty=jump_penalty,
            shaping=True,
        )

    # 6. 帧堆叠（4, 84, 84）uint8 —— 必须在归一化之前，确保 dtype 一致
    env = FrameStack(env, num_stack=stack_size)

    # 7. 像素归一化 [0,1] float32 —— 必须在 FrameStack 之后，正确更新 dtype
    env = NormalizeObservation(env)

    # 8. Monitor 包装 —— 记录 episode 奖励/长度，供 SB3 记录 rollout/ep_rew_mean
    #    info_keywords 把 info 中的 x_pos（最终前进距离）和 flag_get（是否通关）
    #    也记录到 TensorBoard，方便直接看模型走了多远、有没有通关
    #    allow_early_resets=True 避免 SB3 提前 reset 时报错
    env = Monitor(env, allow_early_resets=True, info_keywords=("x_pos", "flag_get"))

    return env
