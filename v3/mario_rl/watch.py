"""
观看 AI 玩马里奥（v2 版）。

支持加载 checkpoint、渲染、遍历所有关卡、录制视频。

用法：
    python -m mario_rl.watch --world 1 --stage 1
    python -m mario_rl.watch --all-levels --deterministic
    python -m mario_rl.watch --model checkpoints/best_model.pt --speed 2.0
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import config
from .env_wrappers import make_env
from .model import ActorCritic
from .utils import get_device


# ─── 观看结果 ───────────────────────────────────────────────────────

@dataclass
class WatchResult:
    """一局游戏的观看结果。

    Attributes:
        reward: 总奖励。
        steps: 步数。
        x_pos: 最终 x 坐标。
        flag_get: 是否通关。
    """
    reward: float
    steps: int
    x_pos: int
    flag_get: bool


# ─── 工具函数 ───────────────────────────────────────────────────────

def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """查找最新的 checkpoint 文件（按修改时间排序）。

    优先查找 mario_ppo_*_steps.pt，找不到则用 mario_ppo_final.pt。

    Args:
        checkpoint_dir: checkpoint 目录。

    Returns:
        最新 checkpoint 的路径。

    Raises:
        FileNotFoundError: 目录中没有任何 checkpoint。
    """
    checkpoints = list(checkpoint_dir.glob("mario_ppo_*_steps.pt"))
    if not checkpoints:
        # 尝试 final
        final = checkpoint_dir / "mario_ppo_final.pt"
        if final.exists():
            return final
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到 checkpoint")
    # 按修改时间排序，最新的在前
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def generate_all_levels() -> List[Tuple[int, int]]:
    """生成所有 32 关列表（1-1 到 8-4）。

    Returns:
        (world, stage) 元组列表，共 32 个。
    """
    levels = []
    for world in range(1, 9):
        for stage in range(1, 5):
            levels.append((world, stage))
    return levels


# ─── 观看一局 ───────────────────────────────────────────────────────

def watch_episode(
    model: ActorCritic,
    env,
    device: torch.device,
    deterministic: bool = False,
    render: bool = True,
    speed: float = 1.0,
) -> WatchResult:
    """观看一局游戏。

    Args:
        model: ActorCritic 模型。
        env: 单环境实例。
        device: 计算设备。
        deterministic: 是否使用确定性策略（取概率最大动作）。
        render: 是否渲染画面。
        speed: 播放速度（1.0=正常，2.0=2倍速）。

    Returns:
        WatchResult 观看结果。
    """
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    x_pos = 0
    flag_get = False
    done = False

    model.eval()
    with torch.no_grad():
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = model.get_action(obs_tensor, deterministic=deterministic)
            action_np = int(action.item())

            obs, reward, done, info = env.step(action_np)
            total_reward += reward
            steps += 1

            if "x_pos" in info:
                x_pos = info["x_pos"]
            if "flag_get" in info:
                flag_get = info["flag_get"]

            if render:
                env.render()
                if speed > 0:
                    time.sleep(1.0 / (60.0 * speed))

    return WatchResult(
        reward=total_reward,
        steps=steps,
        x_pos=x_pos,
        flag_get=flag_get,
    )


# ─── 参数解析 ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description="观看 AI 玩马里奥（v2）")
    parser.add_argument("--model", type=str, default=None, help="模型路径（默认加载最新 checkpoint）")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=0, help="观看局数（0=无限）")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性策略（取概率最大动作）")
    parser.add_argument("--no-render", action="store_true", help="不渲染画面")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度（1.0=正常，2.0=2倍速）")
    parser.add_argument("--all-levels", action="store_true", help="按顺序玩所有 32 关")
    parser.add_argument("--record", action="store_true", help="录制视频（预留，暂未实现）")
    return parser.parse_args()


# ─── 主函数 ─────────────────────────────────────────────────────────

def main() -> None:
    """观看主函数。

    流程：
      1. 解析参数，检测设备
      2. 加载模型（指定路径或最新 checkpoint）
      3. 单关循环观看，或遍历所有 32 关
      4. 打印统计结果
    """
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print("=== 观看 AI 玩马里奥（v2）===")
    print("=" * 60)
    print(f"设备: {device}")

    # 加载模型
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_checkpoint(config.paths.checkpoint_dir)
    print(f"模型: {model_path}")

    # 获取观测空间和动作数（创建一个临时环境）
    temp_env = make_env(world=args.world, stage=args.stage)
    obs_shape = temp_env.observation_space.shape
    n_actions = temp_env.action_space.n
    temp_env.close()

    model = ActorCritic(input_channels=obs_shape[0], n_actions=n_actions).to(device)

    # 加载权重（v2 格式是 dict，包含 model_state_dict）
    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"模型已加载，参数量: {sum(p.numel() for p in model.parameters()):,}")

    render = not args.no_render

    if args.all_levels:
        # 遍历所有关卡
        levels = generate_all_levels()
        print(f"\n按顺序玩所有 {len(levels)} 关...")
        print("-" * 60)

        results: List[WatchResult] = []
        for i, (world, stage) in enumerate(levels):
            env = make_env(world=world, stage=stage, seed=42 + i)
            print(f"关卡 {world}-{stage}: ", end="", flush=True)

            result = watch_episode(model, env, device,
                                   deterministic=args.deterministic,
                                   render=render, speed=args.speed)
            results.append(result)

            status = "通关!" if result.flag_get else "未通关"
            print(f"奖励={result.reward:.0f}, 步数={result.steps}, "
                  f"x_pos={result.x_pos}, {status}")
            env.close()

        # 汇总
        print("\n" + "=" * 60)
        print("汇总统计:")
        print(f"  总关卡数: {len(results)}")
        print(f"  通关数: {sum(1 for r in results if r.flag_get)}")
        print(f"  通关率: {sum(1 for r in results if r.flag_get)/len(results)*100:.1f}%")
        print(f"  平均奖励: {np.mean([r.reward for r in results]):.1f}")
        print(f"  平均x_pos: {np.mean([r.x_pos for r in results]):.1f}")
        print(f"  平均步数: {np.mean([r.steps for r in results]):.1f}")

        # 各关明细
        print("\n各关明细:")
        print(f"{'关卡':<8} {'奖励':>8} {'步数':>6} {'x_pos':>7} {'通关':>6}")
        print("-" * 40)
        for i, (world, stage) in enumerate(levels):
            r = results[i]
            print(f"{world}-{stage:<5} {r.reward:>8.0f} {r.steps:>6} {r.x_pos:>7} "
                  f"{'✓' if r.flag_get else '✗':>6}")

    else:
        # 单关循环
        env = make_env(world=args.world, stage=args.stage)
        print(f"\n关卡: {args.world}-{args.stage}")
        print(f"策略: {'确定性' if args.deterministic else '随机'}")
        print(f"渲染: {'开启' if render else '关闭'}")
        print(f"速度: {args.speed}x")
        print("按 Ctrl+C 退出\n")
        print("-" * 60)

        episode = 0
        try:
            while args.episodes == 0 or episode < args.episodes:
                episode += 1
                result = watch_episode(model, env, device,
                                       deterministic=args.deterministic,
                                       render=render, speed=args.speed)
                status = "通关!" if result.flag_get else "未通关"
                print(f"第 {episode} 局: 奖励={result.reward:.0f}, "
                      f"步数={result.steps}, x_pos={result.x_pos}, {status}")
        except KeyboardInterrupt:
            print("\n\n用户中断，退出。")

        env.close()

    print("\n观看结束。")


if __name__ == "__main__":
    main()
