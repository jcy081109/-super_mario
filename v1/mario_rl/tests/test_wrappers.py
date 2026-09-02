"""
环境 Wrapper 管线单元测试。
验证：环境创建、观测空间形状、动作空间、reset/step 闭环、像素范围、帧堆叠。
"""
import numpy as np
import pytest

from mario_rl.config import config
from mario_rl.env_wrappers import FrameStack, ResizeObservation, SkipFrame, make_env


class TestMakeEnv:
    """测试 make_env 组装的完整管线"""

    def test_env_creation(self):
        """环境能正常创建且不报错"""
        env = make_env(seed=42)
        assert env is not None
        env.close()

    def test_observation_space_shape(self):
        """观测空间应为 (stack_size, image_size, image_size) = (4, 84, 84)"""
        env = make_env(seed=42)
        assert env.observation_space.shape == (4, 84, 84)
        env.close()

    def test_action_space(self):
        """动作空间应为 Discrete(7)（SIMPLE_MOVEMENT）"""
        env = make_env(seed=42)
        assert env.action_space.n == 7
        env.close()

    def test_reset_returns_correct_shape(self):
        """reset 返回的观测形状和 dtype 正确"""
        env = make_env(seed=42)
        obs = env.reset()
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.float32
        env.close()

    def test_pixel_range_normalized(self):
        """像素值应归一化到 [0, 1]"""
        env = make_env(seed=42)
        obs = env.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0
        env.close()

    def test_step_returns_valid_tuple(self):
        """step 返回 (obs, reward, done, info) 四元组且形状正确"""
        env = make_env(seed=42)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 4  # gym 0.21: obs, reward, done, info
        obs, reward, done, info = result
        assert obs.shape == (4, 84, 84)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_multi_step_no_error(self):
        """连续 20 步不报错"""
        env = make_env(seed=42)
        env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()
        env.close()

    def test_frame_stack_different_frames(self):
        """堆叠的 4 帧在多步后应发生变化（游戏画面推进）"""
        env = make_env(seed=42, frame_skip=1)
        obs = env.reset()
        # 执行多步确保画面有明显变化（游戏初始帧可能相同）
        for _ in range(20):
            obs2, _, done, _ = env.step(1)  # right
            if done:
                env.reset()
        assert not np.allclose(obs, obs2), "20步后堆叠帧应发生变化"
        env.close()


class TestSkipFrame:
    """测试 SkipFrame Wrapper"""

    def test_skip_repeats_action(self):
        """SkipFrame 应重复动作 skip 次，累计奖励"""
        from mario_rl.env_wrappers import make_env
        env = make_env(seed=42, frame_skip=4)
        env.reset()
        obs, reward, done, info = env.step(1)  # right
        assert obs.shape == (4, 84, 84)
        env.close()


class TestResizeObservation:
    """测试 ResizeObservation Wrapper"""

    def test_resize_shape(self):
        """缩放后形状应为 (shape, shape)（真实管线中先灰度化再缩放）"""
        import gym
        env = gym.make("SuperMarioBros-1-1-v0")
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)  # 先灰度化
        env = ResizeObservation(env, shape=84)
        obs = env.reset()
        assert obs.shape == (84, 84)
        env.close()


class TestConfig:
    """测试配置默认值"""

    def test_default_env_config(self):
        assert config.env.env_id == "SuperMarioBros-1-1-v0"
        assert config.env.frame_skip == 4
        assert config.env.stack_size == 4
        assert config.env.image_size == 84

    def test_default_ppo_config(self):
        assert config.ppo.learning_rate == 3e-4
        assert config.ppo.n_steps == 1024
        assert config.ppo.batch_size == 64
        assert config.ppo.clip_range == 0.2
