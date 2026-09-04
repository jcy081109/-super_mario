"""
环境 Wrapper 管线与工厂函数。

预处理管线：
  原始帧 (240,256,3) uint8
  → SkipFrame(4) / RandomFrameSkip(3~5)  帧跳过（固定或随机）
  → RandomStartPosition(可选)             随机初始位置（从关卡任意 x_pos 开始）
  → GrayScaleObservation                   RGB → 灰度 (240,256) uint8
  → ResizeObservation(84)                  缩放 (84,84) uint8
  → RewardShaping                           可选奖励塑形（时间惩罚+跳跃惩罚）
  → FrameStack(4)                           堆叠 4 帧 (4,84,84) uint8
  → NormalizeObservation                    像素 /255 → [0,1] float32
  → Monitor                                  记录 episode 统计

最终观测空间 (4, 84, 84) float32。
注意：归一化必须在 FrameStack 之后，否则 FrameStack 会强制转回 uint8。
"""
from collections import deque
from typing import Optional, Sequence, Tuple

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .config import config


# ─── 预处理 Wrapper ─────────────────────────────────────────────────

class SkipFrame(gym.Wrapper):
    """每 skip 帧执行一次动作，返回最后一帧的观测。

    动作重复 skip 帧，奖励累加，减少环境交互次数，加速训练。

    Args:
        env: 内层环境。
        skip: 动作重复帧数。
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作 skip 帧，返回最后一帧。

        Args:
            action: 离散动作索引。

        Returns:
            (obs, total_reward, done, info) 最后一帧的观测和累加奖励。
        """
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class RandomFrameSkip(gym.Wrapper):
    """随机帧跳过：每步在 [min_skip, max_skip] 之间随机选择跳过帧数。

    替代固定的 SkipFrame，模拟不同的游戏速度/输入延迟，防止模型记住
    精确的时间节奏（比如"按跳后 exactly 4 帧落地"），提升策略鲁棒性。

    每步动作重复随机帧数，奖励累加，返回最后一帧观测。

    Args:
        env: 内层环境。
        min_skip: 最小跳过帧数。
        max_skip: 最大跳过帧数。
    """

    def __init__(self, env: gym.Env, min_skip: int = 3, max_skip: int = 5) -> None:
        super().__init__(env)
        self.min_skip = min_skip
        self.max_skip = max_skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作随机 skip 帧，返回最后一帧。

        Args:
            action: 离散动作索引。

        Returns:
            (obs, total_reward, done, info) 最后一帧的观测和累加奖励。
        """
        skip = np.random.randint(self.min_skip, self.max_skip + 1)
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    """用 OpenCV 将观测缩放到 shape×shape，保持灰度单通道。

    Args:
        env: 内层环境。
        shape: 缩放后的图像边长（正方形）。
    """

    def __init__(self, env: gym.Env, shape: int = 84) -> None:
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """缩放观测。

        Args:
            observation: 原始观测 (H, W) uint8。

        Returns:
            缩放后的观测 (shape, shape) uint8。
        """
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)


class NormalizeObservation(gym.ObservationWrapper):
    """像素归一化 [0,255] uint8 → [0,1] float32。

    必须正确更新 observation_space 的 dtype，否则后续 FrameStack 会强制转回 uint8。

    Args:
        env: 内层环境。
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """归一化像素到 [0,1]。

        Args:
            observation: (..., H, W) uint8 观测。

        Returns:
            (..., H, W) float32 归一化观测。
        """
        return (observation / 255.0).astype(np.float32)


class FrameStack(gym.Wrapper):
    """堆叠最近 num_stack 帧，输出 (num_stack, H, W)。

    让智能体感知运动方向（起跳、下落、敌人移动）。

    Args:
        env: 内层环境。
        num_stack: 堆叠帧数。
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        super().__init__(env)
        self._num_stack = num_stack
        self._frames: deque = deque(maxlen=num_stack)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self) -> np.ndarray:
        """重置环境，用初始帧填充堆叠。

        Returns:
            (num_stack, H, W) 堆叠观测。
        """
        obs = self.env.reset()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步，更新堆叠帧。

        Args:
            action: 离散动作索引。

        Returns:
            (obs, reward, done, info) 堆叠后的观测。
        """
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """从帧队列生成堆叠观测。"""
        return np.array(self._frames, dtype=self.observation_space.dtype)


# ─── 奖励塑形 ───────────────────────────────────────────────────────

class RewardShaping(gym.RewardWrapper):
    """可选奖励塑形：时间惩罚 + 跳跃惩罚。

    - time_penalty：每步扣一点奖励，鼓励快速前进，避免原地磨蹭。
    - jump_penalty：每次执行跳跃动作扣一点奖励，防止过度跳跃。
      模型容易学会"一直右+跳"，导致遇到坑时跳跃时机不对掉下去。
      惩罚不宜过大（建议 0.01~0.05），否则模型会完全不跳，无法越过障碍。

    SIMPLE_MOVEMENT 7 动作中包含跳跃的是：2=right+A, 4=right+A+B, 5=A
    不包含跳跃的是：0=NOOP, 1=right, 3=right+B, 6=left

    Args:
        env: 内层环境。
        time_penalty: 每步时间惩罚。
        jump_penalty: 每次跳跃惩罚。
        shaping: 是否启用塑形（False 时不修改奖励）。
    """

    # 包含跳跃按键 A 的动作索引（SIMPLE_MOVEMENT）
    JUMP_ACTIONS = {2, 4, 5}

    def __init__(
        self,
        env: gym.Env,
        time_penalty: float = 0.0,
        jump_penalty: float = 0.0,
        shaping: bool = False,
    ) -> None:
        super().__init__(env)
        self.time_penalty = time_penalty
        self.jump_penalty = jump_penalty
        self.shaping = shaping
        self._last_action: Optional[int] = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """重写 step 以获取 action，用于跳跃惩罚判断。

        Args:
            action: 离散动作索引。

        Returns:
            (obs, reward, done, info)
        """
        self._last_action = action
        return super().step(action)

    def reward(self, reward: float) -> float:
        """修改奖励（gym.RewardWrapper 钩子）。

        Args:
            reward: 内层环境返回的原始奖励。

        Returns:
            修改后的奖励。
        """
        if self.shaping:
            # 时间惩罚：每步都扣
            reward = reward - self.time_penalty
            # 跳跃惩罚：仅当执行了跳跃动作时扣
            if self._last_action in self.JUMP_ACTIONS:
                reward = reward - self.jump_penalty
        return reward


# ─── 域随机化 ────────────────────────────────────────────────────────

class RandomStartPosition(gym.Wrapper):
    """随机初始位置：reset 时随机选择一个 x_pos，先自动向右走到该位置再开始。

    缓解过拟合地图的核心手段：模型无法依赖"从起点出发的固定路线"，必须学习
    从关卡任意位置开始的通用策略（遇到坑就跳、遇到怪物就踩/躲），而不是
    记住"在 x=320 跳、在 x=500 蹲"这种位置相关的动作序列。

    实现方式：reset 后随机生成目标 x_pos（0~max_start_x），自动执行"向右走"
    动作直到达到目标位置，然后将该状态作为 episode 的初始状态交给模型。
    自动走的过程不记录奖励和长度（Monitor 在本 wrapper 之外）。

    放置位置：RandomFrameSkip 之后、GrayScaleObservation 之前（需要读取 info['x_pos']）。

    Args:
        env: 内层环境（info 中需包含 x_pos 字段）。
        max_start_x: 随机初始位置的最大 x_pos（0 到 max_start_x 之间均匀随机）。
            建议设为关卡长度的 50~70%，如 1-1 长度约 3200，设 2000 较合适。
        right_action: 向右走的动作索引（SIMPLE_MOVEMENT 中为 1=right）。
        max_walk_steps: 自动走的最大步数，防止死循环。
        max_retries: 自动走时死亡后的最大重试次数。

    Attributes:
        current_start_x: 当前 episode 的初始 x_pos。
    """

    def __init__(
        self,
        env: gym.Env,
        max_start_x: int = 2000,
        right_action: int = 1,
        max_walk_steps: int = 5000,
        max_retries: int = 10,
    ) -> None:
        super().__init__(env)
        self.max_start_x = max_start_x
        self.right_action = right_action
        self.max_walk_steps = max_walk_steps
        self.max_retries = max_retries
        self.current_start_x: int = 0

    def reset(self, **kwargs) -> np.ndarray:
        """重置环境，随机选择初始位置并自动走到该位置。

        Returns:
            随机初始位置的观测帧（未灰度/缩放，与内层 env.reset 返回格式一致）。
        """
        for _ in range(self.max_retries):
            obs = self.env.reset(**kwargs)
            # 随机选择目标 x_pos（0=从起点开始）
            target_x = int(np.random.randint(0, self.max_start_x + 1))
            self.current_start_x = target_x
            if target_x == 0:
                return obs
            # 自动向右走到目标位置
            for _ in range(self.max_walk_steps):
                obs, _reward, done, info = self.env.step(self.right_action)
                if info.get("x_pos", 0) >= target_x:
                    return obs
                if done:
                    break  # 走到目标前死了，重新 reset 重试
        # 重试次数用完，直接从起点开始
        self.current_start_x = 0
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """透传 step（自动走只在 reset 时发生）。"""
        return self.env.step(action)


# ─── 多关卡环境池 ───────────────────────────────────────────────────

class RandomLevelEnv(gym.Env):
    """随机关卡环境：每次 reset 时随机选择一个关卡。

    强制模型学通用策略而非记忆单关布局，适用于多关卡联合训练。

    优化版（环境池）：启动时预创建所有关卡的环境实例，reset 时直接切换
    并调用 reset，避免每次重建 NES 模拟器 + JoypadSpace + Wrapper 管线的开销。
    所有 step/render/seed/close 调用转发到当前环境。

    Args:
        worlds: 可选的 world 列表，如 (1, 2) 表示 1-1~2-4（与 stages 笛卡尔积）。
        stages: 可选的 stage 列表，如 (1,2,3,4)。
        levels: 直接指定关卡列表，如 ((1,1), (1,2), (2,4))。优先级高于 worlds/stages。
        seed: 随机关卡选择的随机种子。

    Attributes:
        levels: 所有可选关卡元组列表。
        env_pool: (world, stage) → 环境实例 的映射。
        current_env: 当前激活的环境实例。
        current_world: 当前关卡的 world。
        current_stage: 当前关卡的 stage。
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        worlds: Sequence[int] = (1,),
        stages: Sequence[int] = (1, 2, 3, 4),
        levels: Optional[Sequence[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.worlds = tuple(worlds)
        self.stages = tuple(stages)
        if levels is not None:
            self.levels = tuple(tuple(lv) for lv in levels)
        else:
            self.levels = tuple((w, s) for w in worlds for s in stages)
        self._rng = np.random.RandomState(seed)
        self._base_seed = seed if seed is not None else 42

        # 预创建所有关卡环境池（key=(world, stage), value=JoypadSpace环境）
        self.env_pool: dict[Tuple[int, int], gym.Env] = {}
        for i, (w, s) in enumerate(self.levels):
            env_id = f"SuperMarioBros-{w}-{s}-v0"
            raw_env = gym_super_mario_bros.make(env_id)
            env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
            env.seed(self._base_seed + i * 1000)
            self.env_pool[(w, s)] = env

        # 所有关卡的 observation_space 和 action_space 相同，取第一个设置
        first_env = self.env_pool[self.levels[0]]
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space

        # 随机选一个初始关卡
        idx = self._rng.randint(len(self.levels))
        self.current_world, self.current_stage = self.levels[idx]
        self.current_env = self.env_pool[(self.current_world, self.current_stage)]

    def reset(self) -> np.ndarray:
        """随机选一个关卡，从环境池取出并 reset（不重建环境）。

        Returns:
            初始观测。
        """
        idx = self._rng.randint(len(self.levels))
        self.current_world, self.current_stage = self.levels[idx]
        self.current_env = self.env_pool[(self.current_world, self.current_stage)]
        return self.current_env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """转发到当前环境。"""
        return self.current_env.step(action)

    def render(self, mode: str = "human"):
        """转发到当前环境。"""
        return self.current_env.render(mode)

    def seed(self, seed: Optional[int] = None) -> list:
        """重新设置随机种子和所有池子环境的种子。"""
        self._rng = np.random.RandomState(seed)
        self._base_seed = seed if seed is not None else 42
        for i, (w, s) in enumerate(self.levels):
            self.env_pool[(w, s)].seed(self._base_seed + i * 1000)
        return self.current_env.seed(seed)

    def close(self) -> None:
        """关闭池子里所有环境。"""
        for env in self.env_pool.values():
            try:
                env.close()
            except Exception:
                pass

    def __str__(self) -> str:
        return (f"RandomLevelEnv(current={self.current_world}-{self.current_stage}, "
                f"pool_size={len(self.env_pool)})")


# ─── 环境工厂 ───────────────────────────────────────────────────────

def make_env(
    env_id: Optional[str] = None,
    seed: Optional[int] = None,
    frame_skip: Optional[int] = None,
    stack_size: Optional[int] = None,
    image_size: Optional[int] = None,
    shaping: bool = False,
    time_penalty: float = 0.0,
    jump_penalty: float = 0.0,
    world: Optional[int] = None,
    stage: Optional[int] = None,
    multi_level: bool = False,
    multi_worlds: Optional[Sequence[int]] = None,
    multi_stages: Optional[Sequence[int]] = None,
    levels: Optional[Sequence[Tuple[int, int]]] = None,
    random_start: bool = False,
    max_start_x: int = 2000,
    random_frame_skip: bool = False,
    min_frame_skip: int = 3,
    max_frame_skip: int = 5,
) -> gym.Env:
    """创建并组装完整的马里奥环境 Wrapper 管线。

    参数为 None 时使用 config 中的默认值。
    world/stage 指定单关卡（优先级高于 env_id）。
    multi_level=True 时开启随机关卡模式，每次 reset 随机选关卡。
    levels 直接指定关卡列表（如 ((1,1),(1,2),(2,4))），优先级高于 multi_worlds/multi_stages。
    random_start=True 时启用随机初始位置，reset 时从关卡任意 x_pos 开始。
    random_frame_skip=True 时启用随机帧跳过，每步在 [min_frame_skip, max_frame_skip] 间随机。

    Args:
        env_id: 环境 ID（单关模式，world/stage 未指定时使用）。
        seed: 随机种子。
        frame_skip: 固定帧跳过数（random_frame_skip=False 时使用）。
        stack_size: 帧堆叠数。
        image_size: 图像缩放边长。
        shaping: 是否启用奖励塑形。
        time_penalty: 时间惩罚（每步）。
        jump_penalty: 跳跃惩罚（每次跳跃）。
        world: 单关模式的 world。
        stage: 单关模式的 stage。
        multi_level: 是否启用多关卡随机关卡。
        multi_worlds: 多关卡模式的 world 列表。
        multi_stages: 多关卡模式的 stage 列表。
        levels: 直接指定关卡列表。
        random_start: 是否启用随机初始位置。
        max_start_x: 随机初始位置的最大 x_pos。
        random_frame_skip: 是否启用随机帧跳过。
        min_frame_skip: 随机帧跳过的最小值。
        max_frame_skip: 随机帧跳过的最大值。

    Returns:
        组装好完整 Wrapper 管线的环境。
    """
    cfg = config.env
    seed = seed if seed is not None else cfg.seed
    frame_skip = frame_skip if frame_skip is not None else cfg.frame_skip
    stack_size = stack_size if stack_size is not None else cfg.stack_size
    image_size = image_size if image_size is not None else cfg.image_size

    # 1. 创建原始环境 + 动作空间压缩（SIMPLE_MOVEMENT 7 动作）
    if multi_level:
        # 多关卡模式：RandomLevelEnv 内部已包含 JoypadSpace
        if multi_worlds is None:
            multi_worlds = (1, 2, 3, 4, 5, 6, 7, 8)
        if multi_stages is None:
            multi_stages = (1, 2, 3)
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

    # 2. 帧跳过（固定或随机）
    if random_frame_skip:
        env = RandomFrameSkip(env, min_skip=min_frame_skip, max_skip=max_frame_skip)
    else:
        env = SkipFrame(env, skip=frame_skip)

    # 2.5 随机初始位置 —— 在帧跳过之后、灰度化之前（需要读取 info['x_pos']）
    if random_start:
        env = RandomStartPosition(env, max_start_x=max_start_x)

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

    # 8. Monitor 包装 —— 记录 episode 奖励/长度
    env = Monitor(env, allow_early_resets=True, info_keywords=("x_pos", "flag_get"))

    return env


class EnvFactory:
    """可序列化的环境工厂类，用于 SubprocVecEnv 多进程真并行。

    Windows 上 pickle 无法序列化 lambda/局部函数，因此用模块级类代替。
    每个子进程调用 __call__() 创建独立环境实例。

    Args:
        seed: 基础随机种子（子进程 rank 会叠加）。
        shaping: 是否启用奖励塑形。
        time_penalty: 时间惩罚。
        jump_penalty: 跳跃惩罚。
        multi_level: 是否多关卡模式。
        multi_worlds: 多关卡 world 列表。
        multi_stages: 多关卡 stage 列表。
        levels: 直接指定关卡列表。
        frame_skip: 固定帧跳过数。
        stack_size: 帧堆叠数。
        image_size: 图像缩放边长。
        world: 单关 world。
        stage: 单关 stage。
        random_start: 是否启用随机初始位置。
        max_start_x: 随机初始位置的最大 x_pos。
        random_frame_skip: 是否启用随机帧跳过。
        min_frame_skip: 随机帧跳过最小值。
        max_frame_skip: 随机帧跳过最大值。
    """

    def __init__(
        self,
        seed: int = 42,
        shaping: bool = False,
        time_penalty: float = 0.0,
        jump_penalty: float = 0.0,
        multi_level: bool = False,
        multi_worlds: Optional[Sequence[int]] = None,
        multi_stages: Optional[Sequence[int]] = None,
        levels: Optional[Sequence[Tuple[int, int]]] = None,
        frame_skip: int = 4,
        stack_size: int = 4,
        image_size: int = 84,
        world: Optional[int] = None,
        stage: Optional[int] = None,
        random_start: bool = False,
        max_start_x: int = 2000,
        random_frame_skip: bool = False,
        min_frame_skip: int = 3,
        max_frame_skip: int = 5,
    ) -> None:
        self.seed = seed
        self.shaping = shaping
        self.time_penalty = time_penalty
        self.jump_penalty = jump_penalty
        self.multi_level = multi_level
        self.multi_worlds = multi_worlds
        self.multi_stages = multi_stages
        self.levels = levels
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.image_size = image_size
        self.world = world
        self.stage = stage
        self.random_start = random_start
        self.max_start_x = max_start_x
        self.random_frame_skip = random_frame_skip
        self.min_frame_skip = min_frame_skip
        self.max_frame_skip = max_frame_skip

    def __call__(self, rank: int = 0) -> gym.Env:
        """创建一个环境实例（子进程调用）。

        Args:
            rank: 子进程编号，用于设置不同种子。

        Returns:
            组装好的环境实例。
        """
        return make_env(
            seed=self.seed + rank,
            frame_skip=self.frame_skip,
            stack_size=self.stack_size,
            image_size=self.image_size,
            shaping=self.shaping,
            time_penalty=self.time_penalty,
            jump_penalty=self.jump_penalty,
            world=self.world,
            stage=self.stage,
            multi_level=self.multi_level,
            multi_worlds=self.multi_worlds,
            multi_stages=self.multi_stages,
            levels=self.levels,
            random_start=self.random_start,
            max_start_x=self.max_start_x,
            random_frame_skip=self.random_frame_skip,
            min_frame_skip=self.min_frame_skip,
            max_frame_skip=self.max_frame_skip,
        )


def make_vec_envs(
    n_envs: int = 4,
    seed: int = 42,
    vec_env_type: str = "subproc",
    world: int = 1,
    stage: int = 1,
    multi_level: bool = False,
    multi_worlds: Optional[Sequence[int]] = None,
    multi_stages: Optional[Sequence[int]] = None,
    levels: Optional[Sequence[Tuple[int, int]]] = None,
    frame_skip: int = 4,
    stack_size: int = 4,
    image_size: int = 84,
    shaping: bool = False,
    time_penalty: float = 0.0,
    jump_penalty: float = 0.0,
    random_start: bool = False,
    max_start_x: int = 2000,
    random_frame_skip: bool = False,
    min_frame_skip: int = 3,
    max_frame_skip: int = 5,
):
    """创建向量化环境（多环境并行）。

    Args:
        n_envs: 并行环境数量。
        seed: 基础随机种子。
        vec_env_type: "subproc"（多进程真并行）或 "dummy"（单进程串行）。
        world: 单关模式的 world。
        stage: 单关模式的 stage。
        multi_level: 是否多关卡模式。
        multi_worlds: 多关卡 world 列表。
        multi_stages: 多关卡 stage 列表。
        levels: 直接指定关卡列表。
        frame_skip: 固定帧跳过数。
        stack_size: 帧堆叠数。
        image_size: 图像缩放边长。
        shaping: 是否启用奖励塑形。
        time_penalty: 时间惩罚。
        jump_penalty: 跳跃惩罚。
        random_start: 是否启用随机初始位置。
        max_start_x: 随机初始位置的最大 x_pos。
        random_frame_skip: 是否启用随机帧跳过。
        min_frame_skip: 随机帧跳过最小值。
        max_frame_skip: 随机帧跳过最大值。

    Returns:
        VecEnv 向量化环境实例。

    Raises:
        ValueError: vec_env_type 不是 "subproc" 或 "dummy"。
    """
    factory = EnvFactory(
        seed=seed,
        shaping=shaping,
        time_penalty=time_penalty,
        jump_penalty=jump_penalty,
        multi_level=multi_level,
        multi_worlds=multi_worlds,
        multi_stages=multi_stages,
        levels=levels,
        frame_skip=frame_skip,
        stack_size=stack_size,
        image_size=image_size,
        world=world,
        stage=stage,
        random_start=random_start,
        max_start_x=max_start_x,
        random_frame_skip=random_frame_skip,
        min_frame_skip=min_frame_skip,
        max_frame_skip=max_frame_skip,
    )

    if vec_env_type == "subproc":
        envs = SubprocVecEnv([factory for _ in range(n_envs)])
    elif vec_env_type == "dummy":
        envs = DummyVecEnv([factory for _ in range(n_envs)])
    else:
        raise ValueError(f"未知 vec_env_type: {vec_env_type}，可选 'subproc' 或 'dummy'")

    return envs
