"""
实时观看 AI 玩马里奥。

功能：
- 自动加载 checkpoints/ 目录下最新模型（或用 --model 指定）
- 每局结束后自动检查并加载更新的 checkpoint（训练时可实时看最新效果）
- 支持实时渲染、慢放/快放、视频录制
- 支持单关模式（默认）和全关卡模式（--all-levels，按顺序玩1-1到8-4）
- 无限模式（默认）按 Ctrl+C 退出

用法:
    python -m mario_rl.watch                            # 自动加载最新 checkpoint，无限玩1-1
    python -m mario_rl.watch --all-levels               # 按顺序玩32关（1-1到8-4），循环
    python -m mario_rl.watch --all-levels --episodes 32 # 玩一遍32关后停止
    python -m mario_rl.watch --model xxx.zip            # 指定模型
    python -m mario_rl.watch --world 2 --stage 4        # 指定单关
    python -m mario_rl.watch --episodes 10 --record     # 玩10局并录屏
    python -m mario_rl.watch --speed 0.5                 # 慢放
"""
import argparse
import time
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

from .config import config
from .env_wrappers import make_env
from .utils import get_device


def find_latest_checkpoint() -> Path | None:
    """查找 checkpoints/ 目录下最新的 .zip 模型文件（按修改时间排序）"""
    ckpt_dir = config.paths.checkpoint_dir
    if not ckpt_dir.exists():
        return None
    zips = sorted(ckpt_dir.rglob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def generate_all_levels():
    """生成32关列表：1-1到8-4，按顺序"""
    return [(w, s) for w in range(1, 9) for s in range(1, 5)]


def parse_args():
    parser = argparse.ArgumentParser(description="实时观看 AI 玩马里奥")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径（不指定则自动加载 checkpoints/ 下最新 checkpoint）")
    parser.add_argument("--episodes", type=int, default=0,
                        help="玩多少局（0=无限模式，Ctrl+C 退出）")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="使用确定性策略（默认随机策略，能看到更多变化；想看最优表现加这个）")
    parser.add_argument("--shaping", action="store_true", help="开启奖励塑形（需与训练一致）")
    parser.add_argument("--no-render", action="store_true", help="关闭渲染（默认开启）")
    parser.add_argument("--record", action="store_true", help="录制视频到 videos/ 目录")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="渲染速度倍率（1.0=原速，0.5=慢放，2.0=快放）")
    parser.add_argument("--no-auto-reload", action="store_true",
                        help="关闭自动刷新模型（默认每局结束检查新 checkpoint）")
    parser.add_argument("--world", type=int, default=None,
                        help="关卡 world（1-8，不指定则用默认 1-1 关）")
    parser.add_argument("--stage", type=int, default=None,
                        help="关卡 stage（1-4，需与 --world 一起使用）")
    parser.add_argument("--all-levels", action="store_true",
                        help="全关卡模式：按顺序玩1-1到8-4共32关，玩完循环（--all-levels --episodes 32 则玩一遍停止）")
    return parser.parse_args()


def load_model(model_path: str, device, env) -> PPO:
    """加载模型并绑定环境（用 env=env 方式加载，自动适配环境数，
    避免模型训练时用了多环境但 watch 是单环境导致 n_envs 不匹配）"""
    return PPO.load(model_path, env=env, device=device)


def main():
    args = parse_args()
    device = get_device()
    render = not args.no_render
    auto_reload = not args.no_auto_reload

    # 确定关卡列表
    if args.all_levels:
        levels = generate_all_levels()
        level_desc = f"全关卡模式（{len(levels)}关，按顺序1-1到8-4，玩完循环）"
    elif args.world is not None and args.stage is not None:
        levels = [(args.world, args.stage)]
        level_desc = f"单关 {args.world}-{args.stage}"
    else:
        levels = [(1, 1)]
        level_desc = "单关 1-1（默认）"

    # 确定模型路径
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"错误：模型文件不存在: {model_path}")
            return
    else:
        model_path = find_latest_checkpoint()
        if model_path is None:
            print("错误：checkpoints/ 目录下没有找到 .zip 模型文件。")
            print("请先训练模型，或用 --model 指定模型路径。")
            return

    print(f"=== 观看 AI 玩马里奥 ===")
    print(f"模型: {model_path}")
    print(f"设备: {device}")
    print(f"关卡: {level_desc}")
    print(f"渲染: {'开启' if render else '关闭'}")
    print(f"自动刷新: {'开启（每局结束检查新 checkpoint）' if auto_reload else '关闭'}")
    print(f"局数: {'无限（Ctrl+C 退出）' if args.episodes == 0 else args.episodes}")
    print(f"策略: {'确定性（贪心，每局相同）' if args.deterministic else '随机（采样，每局不同）'}")

    results = []  # [(world, stage, reward, steps, x_pos, flag_get)]
    ep = 0
    level_idx = 0
    env = None
    current_model_mtime = model_path.stat().st_mtime

    try:
        while args.episodes == 0 or ep < args.episodes:
            ep += 1
            world, stage = levels[level_idx % len(levels)]
            level_idx += 1

            # 每局重建环境（切换关卡），并重新加载模型绑定新环境
            if env is not None:
                env.close()
            env = make_env(seed=args.seed + ep, shaping=args.shaping, world=world, stage=stage)
            model = load_model(str(model_path), device, env)

            obs = env.reset()
            total_reward = 0.0
            done = False
            steps = 0
            frames = [] if args.record else None

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, info = env.step(int(action))
                total_reward += reward
                steps += 1

                if render:
                    env.render(mode='human')
                    if args.speed != 1.0:
                        time.sleep(0.016 / args.speed)

                if args.record:
                    frames.append(env.render(mode='rgb_array'))

            final_x = info.get("x_pos", 0)
            flag_get = info.get("flag_get", False)
            results.append((world, stage, total_reward, steps, final_x, flag_get))

            status = "✓ 通关" if flag_get else f"x={final_x}"
            print(f"  Episode {ep} [{world}-{stage}]: reward={total_reward:.1f}, "
                  f"steps={steps}, {status}")

            # 保存视频
            if args.record and frames:
                video_dir = config.paths.video_dir
                video_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_path = video_dir / f"watch_{world}-{stage}_ep{ep}_reward{total_reward:.0f}_{timestamp}.mp4"
                imageio.mimwrite(str(video_path), frames, fps=args.fps, quality=8)
                print(f"    视频: {video_path}")

            # 自动刷新模型：每局结束后检查是否有更新的 checkpoint
            if auto_reload:
                latest = find_latest_checkpoint()
                if latest and latest.stat().st_mtime > current_model_mtime:
                    print(f"  >> 发现新 checkpoint: {latest.name}，下一局加载...")
                    model_path = latest
                    current_model_mtime = latest.stat().st_mtime

    except KeyboardInterrupt:
        print("\n用户中断，退出。")

    if env is not None:
        env.close()

    # 汇总统计
    if len(results) > 1:
        print(f"\n{'='*50}")
        print(f"=== 汇总（共 {len(results)} 局）===")
        win_count = sum(1 for r in results if r[5])
        avg_reward = np.mean([r[2] for r in results])
        avg_x = np.mean([r[4] for r in results])
        avg_steps = np.mean([r[3] for r in results])
        print(f"通关率: {win_count}/{len(results)} ({win_count/len(results):.1%})")
        print(f"平均奖励: {avg_reward:.1f}")
        print(f"平均 x_pos: {avg_x:.0f}")
        print(f"平均步数: {avg_steps:.0f}")

        # 全关卡模式下输出各关明细
        if args.all_levels:
            print(f"\n各关明细:")
            print(f"{'关卡':<6} {'奖励':>8} {'步数':>6} {'x_pos':>8} {'结果':<10}")
            print(f"{'-'*45}")
            for w, s, r, st, x, fg in results:
                status = "✓ 通关" if fg else "✗ 未通关"
                print(f"{w}-{s:<4} {r:>8.1f} {st:>6} {x:>8} {status:<10}")

    print("观看结束。")


if __name__ == "__main__":
    main()
