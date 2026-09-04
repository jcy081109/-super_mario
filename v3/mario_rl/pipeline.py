"""
流水线 PPO 调度器（v2 核心特性）。

双缓冲 + 模型快照，收集和更新并行，满足 on-policy 约束。

设计原理：
- collect_model：收集用模型（快照），推理时参数固定
- train_model：训练用模型，更新时参数变化
- 每轮收集开始前，collect_model 同步为 train_model 的最新参数
- 收集第N轮（用θ_{N-1}）和更新第N-1轮（θ_{N-2}→θ_{N-1}）并行
- 每批数据来自同一个固定策略，满足 on-policy 约束

双缓冲：
- buffer_a / buffer_b 交替使用
- 收集线程写 write_buffer，更新线程读 read_buffer
- 收集完后交换缓冲区，通知更新线程
"""
import copy
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Optional

import numpy as np
import torch

from .collector import Collector, CollectResult
from .model import ActorCritic
from .ppo import PPO, RolloutBuffer, UpdateMetrics


# ─── 流水线统计 ─────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    """流水线运行统计。

    Attributes:
        n_iterations: 已完成的迭代轮数。
        total_timesteps: 累计收集步数。
        avg_collect_time: 最近20轮平均收集时间（秒）。
        avg_update_time: 最近20轮平均更新时间（秒）。
        pipeline_enabled: 流水线是否启用。
        speedup: 理论加速比（串行时间/收集时间）。
    """
    n_iterations: int
    total_timesteps: int
    avg_collect_time: float
    avg_update_time: float
    pipeline_enabled: bool
    speedup: float

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "n_iterations": self.n_iterations,
            "total_timesteps": self.total_timesteps,
            "avg_collect_time": self.avg_collect_time,
            "avg_update_time": self.avg_update_time,
            "pipeline_enabled": self.pipeline_enabled,
            "speedup": self.speedup,
        }


# ─── 流水线 PPO 调度器 ──────────────────────────────────────────────

class PipelinePPO:
    """流水线 PPO 训练器。

    收集线程和更新线程并行，双缓冲交替。

    Args:
        train_model: 训练用模型（更新时参数变化）。
        ppo: PPO 算法实例。
        collector: 数据收集器。
        n_envs: 并行环境数。
        n_steps: 每环境每轮收集步数。
        obs_shape: 观测空间形状。
        device: 计算设备。
        gamma: 折扣因子。
        gae_lambda: GAE 平滑参数。
        n_epochs: 每轮数据更新 epoch 数。
        batch_size: 小批量大小。
        enabled: 是否启用流水线（False 时串行收集更新）。

    Attributes:
        collect_model: 收集用模型快照（参数固定）。
        buffer_a/b: 双缓冲 RolloutBuffer。
        write_buffer: 当前写缓冲区。
        read_buffer: 当前读缓冲区。
        total_timesteps: 累计收集步数。
        n_iterations: 已完成迭代轮数。
    """

    def __init__(
        self,
        train_model: ActorCritic,
        ppo: PPO,
        collector: Collector,
        n_envs: int,
        n_steps: int,
        obs_shape: tuple,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_epochs: int = 8,
        batch_size: int = 512,
        enabled: bool = True,
    ) -> None:
        self.train_model = train_model
        self.ppo = ppo
        self.collector = collector
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.enabled = enabled

        # 收集用模型快照（参数固定，不受更新影响）
        self.collect_model = copy.deepcopy(train_model)
        self.collect_model.eval()

        # 双缓冲
        self.buffer_a = RolloutBuffer(n_envs, n_steps, obs_shape, device)
        self.buffer_b = RolloutBuffer(n_envs, n_steps, obs_shape, device)
        self.write_buffer = self.buffer_a  # 当前写缓冲区
        self.read_buffer: Optional[RolloutBuffer] = None  # 当前读缓冲区

        # 同步事件
        self.collect_ready = threading.Event()  # 收集完成，数据可读
        self.update_ready = threading.Event()   # 更新完成，可以同步参数
        self.stop_event = threading.Event()      # 停止信号

        # 结果队列：更新线程把训练指标放这里
        self.result_queue: Queue = Queue(maxsize=4)

        # 统计
        self.n_iterations = 0
        self.total_timesteps = 0
        self.collect_times: list[float] = []
        self.update_times: list[float] = []

        # 更新线程
        self.update_thread: Optional[threading.Thread] = None
        if enabled:
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def _sync_collect_model(self) -> None:
        """把训练模型参数同步到收集模型（快照）。"""
        self.collect_model.load_state_dict(self.train_model.state_dict())
        self.collect_model.eval()

    def _update_loop(self) -> None:
        """更新线程循环：等待收集完成 → 更新 → 通知。"""
        while not self.stop_event.is_set():
            # 等待收集完成
            if not self.collect_ready.wait(timeout=1.0):
                continue
            self.collect_ready.clear()

            if self.read_buffer is None or not self.read_buffer.full:
                continue

            # 更新模型
            start_time = time.time()
            # 从 buffer 的正式属性获取 last_values
            last_values = self.read_buffer.last_values
            if last_values is None:
                # fallback：用最后一步的 value（不应该发生）
                last_values = self.read_buffer.values[-1]

            metrics: UpdateMetrics = self.ppo.update(
                buffer=self.read_buffer,
                last_values=last_values,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
            )
            update_time = time.time() - start_time
            self.update_times.append(update_time)
            metrics.update_time = update_time

            # 把结果放入队列
            if not self.result_queue.full():
                self.result_queue.put(metrics)

            # 通知收集线程：更新完成，可以同步参数
            self.update_ready.set()

    def collect_one_iteration(self) -> dict:
        """收集一轮数据（主线程调用）。

        如果流水线启用，收集和上一轮更新并行。
        流程：等待上一轮更新完成 → 同步参数 → 收集 → 交换缓冲区 → 通知更新。

        Returns:
            字典，包含：
            - iteration: 当前迭代轮数
            - total_timesteps: 累计步数
            - collect_time: 本轮收集时间
            - n_collected: 本轮收集步数
            - update_metrics: 上一轮更新指标（UpdateMetrics.to_dict()），可能为 None
            - ep_rew_mean, ep_len_mean, ep_x_pos_mean, ep_win_rate: episode 统计
        """
        # 如果启用流水线，等待上一轮更新完成，然后同步参数
        if self.enabled and self.n_iterations > 0:
            self.update_ready.wait()
            self.update_ready.clear()
            self._sync_collect_model()

        # 重置写缓冲区
        self.write_buffer.reset()

        # 收集数据（用 collect_model，参数固定）
        # 临时替换 collector 的模型为 collect_model
        original_model = self.collector.model
        self.collector.model = self.collect_model

        start_time = time.time()
        result: CollectResult = self.collector.collect(self.write_buffer)
        collect_time = time.time() - start_time

        # 恢复 collector 模型
        self.collector.model = original_model

        # 存储 last_values 到 buffer 的正式属性
        self.write_buffer.last_values = result.last_values

        self.collect_times.append(collect_time)
        self.total_timesteps += result.n_collected
        self.n_iterations += 1

        # 交换缓冲区
        self.read_buffer = self.write_buffer
        self.write_buffer = self.buffer_b if self.write_buffer is self.buffer_a else self.buffer_a

        # 通知更新线程
        if self.enabled:
            self.collect_ready.set()

        # 获取更新结果（如果有）
        update_metrics: Optional[dict] = None
        if not self.result_queue.empty():
            metrics = self.result_queue.get_nowait()
            update_metrics = metrics.to_dict() if isinstance(metrics, UpdateMetrics) else metrics

        stats = self.collector.get_stats()
        result_dict = {
            "iteration": self.n_iterations,
            "total_timesteps": self.total_timesteps,
            "collect_time": collect_time,
            "n_collected": result.n_collected,
            "update_metrics": update_metrics,
        }
        result_dict.update(stats.to_dict())
        return result_dict

    def collect_serial(self) -> dict:
        """串行模式（不启用流水线时使用）：收集 → 更新 → 同步。

        Returns:
            同 collect_one_iteration 的返回格式。
        """
        # 同步收集模型
        self._sync_collect_model()

        # 重置缓冲区
        self.write_buffer.reset()

        # 收集
        original_model = self.collector.model
        self.collector.model = self.collect_model
        start_time = time.time()
        result: CollectResult = self.collector.collect(self.write_buffer)
        collect_time = time.time() - start_time
        self.collector.model = original_model

        self.total_timesteps += result.n_collected
        self.n_iterations += 1

        # 更新（串行，直接调用）
        start_time = time.time()
        metrics: UpdateMetrics = self.ppo.update(
            buffer=self.write_buffer,
            last_values=result.last_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
        )
        update_time = time.time() - start_time
        metrics.update_time = update_time

        stats = self.collector.get_stats()
        result_dict = {
            "iteration": self.n_iterations,
            "total_timesteps": self.total_timesteps,
            "collect_time": collect_time,
            "update_time": update_time,
            "n_collected": result.n_collected,
            "update_metrics": metrics.to_dict(),
        }
        result_dict.update(stats.to_dict())
        return result_dict

    def get_pipeline_stats(self) -> PipelineStats:
        """获取流水线统计信息。

        Returns:
            PipelineStats 统计结果。
        """
        avg_collect = float(np.mean(self.collect_times[-20:])) if self.collect_times else 0.0
        avg_update = float(np.mean(self.update_times[-20:])) if self.update_times else 0.0

        if self.enabled and len(self.collect_times) > 5 and len(self.update_times) > 5:
            speedup = (avg_collect + avg_update) / max(avg_collect, 1e-8)
        else:
            speedup = 1.0

        return PipelineStats(
            n_iterations=self.n_iterations,
            total_timesteps=self.total_timesteps,
            avg_collect_time=avg_collect,
            avg_update_time=avg_update,
            pipeline_enabled=self.enabled,
            speedup=float(speedup),
        )

    def close(self) -> None:
        """停止更新线程，关闭环境。"""
        self.stop_event.set()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        self.collector.close()
