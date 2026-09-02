"""
评估/推理演示：加载训练好的模型，运行 N 个 episode，记录奖励与通关进度。
支持实时渲染游戏画面和视频录制。

用法:
    python -m mario_rl.evaluate --model checkpoints/mario_ppo_final.zip --render
    python -m mario_rl.evaluate --model checkpoints/best/best_model.zip --episodes 10 --render --record
    python -m mario_rl.evaluate --model xxx.zip --render --speed 0.5   # 慢放
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


def parse_args():
    parser = argparse.ArgumentParser(description="评估马里奥 PPO 智能体")
    parser.add_argument("--model", type=str, required=True, help="模型路径（.zip）")
    parser.add_argument("--episodes", type=int, default=5, help="评估 episode 数")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--shaping", action="store_true", help="与训练一致开启奖励塑形")
    parser.add_argument("--render", action="store_true", help="实时渲染游戏画面")
    parser.add_argument("--record", action="store_true", help="录制视频到 videos/ 目录")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="渲染速度倍率（1.0=原速，0.5=慢放，2.0=快放）")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(f"=== 马里奥 PPO 评估 ===")
    print(f"模型: {args.model}")
    print(f"设备: {device}")
    print(f"渲染: {'开启' if args.render else '关闭'}")
    print(f"录屏: {'开启' if args.record else '关闭'}")
    print(f"评估 episode 数: {args.episodes}")

    # 1. 创建环境
    env = make_env(seed=args.seed, shaping=args.shaping)
    print(f"观测空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space.n}")

    # 2. 加载模型（用 env=env 方式加载，自动适配环境数，
    #    避免模型训练时用了多环境但评估时是单环境导致 n_envs 不匹配）
    model = PPO.load(args.model, env=env, device=device)

    # 3. 运行评估
    rewards = []
    x_pos_list = []
    flag_get_count = 0

    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        frames = [] if args.record else None

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(int(action))  # predict 返回 ndarray，需转 int
            total_reward += reward
            steps += 1

            # 实时渲染
            if args.render:
                env.render(mode='human')
                if args.speed != 1.0:
                    time.sleep(0.016 / args.speed)  # 原速约 60fps

            # 收集录屏帧
            if args.record:
                frames.append(env.render(mode='rgb_array'))

        rewards.append(total_reward)
        final_x = info.get("x_pos", 0)
        x_pos_list.append(final_x)
        if info.get("flag_get", False):
            flag_get_count += 1

        print(f"  Episode {ep+1}/{args.episodes}: "
              f"reward={total_reward:.1f}, steps={steps}, "
              f"x_pos={final_x}, flag_get={info.get('flag_get', False)}")

        # 保存视频
        if args.record and frames:
            video_dir = config.paths.video_dir
            video_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = video_dir / f"eval_ep{ep+1}_reward{total_reward:.0f}_{timestamp}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=args.fps, quality=8)
            print(f"    视频已保存: {video_path}")

    # 4. 汇总统计
    print(f"\n=== 评估汇总 ===")
    print(f"平均奖励: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"最高奖励: {np.max(rewards):.1f}")
    print(f"平均前进距离 x_pos: {np.mean(x_pos_list):.0f}")
    print(f"通关次数: {flag_get_count}/{args.episodes}")

    env.close()


if __name__ == "__main__":
    main()
