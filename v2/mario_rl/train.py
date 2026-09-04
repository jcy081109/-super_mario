"""
训练入口：流水线 PPO 训练马里奥。

v2 核心：双缓冲流水线收集+更新，CPU/GPU 最大化利用。

用法：
    python -m mario_rl.train --total-timesteps 5000000 --n-envs 16
    python -m mario_rl.train --multi-level --multi-worlds 1 --multi-stages 1 2 3
    python -m mario_rl.train --model-size large --shaping --time-penalty 0.01
"""
import argparse
import time
from typing import Optional

import numpy as np
import torch

from .collector import Collector
from .config import config
from .env_wrappers import make_vec_envs
from .model import ActorCritic
from .pipeline import PipelinePPO
from .ppo import PPO
from .utils import (
    TrainLogger,
    count_parameters,
    evaluate,
    get_device,
    make_lr_schedule,
    set_seed,
)


# ─── 参数解析 ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description="马里奥 PPO 训练（v2 流水线版）")

    # 训练规模
    parser.add_argument("--total-timesteps", type=int, default=config.train.total_timesteps)
    parser.add_argument("--n-envs", type=int, default=config.train.n_envs)
    parser.add_argument("--n-steps", type=int, default=config.ppo.n_steps)
    parser.add_argument("--batch-size", type=int, default=config.ppo.batch_size)
    parser.add_argument("--n-epochs", type=int, default=config.ppo.n_epochs)

    # 学习率
    parser.add_argument("--learning-rate", type=float, default=config.ppo.learning_rate)
    parser.add_argument("--lr-schedule", type=str, default="linear", choices=["constant", "linear"])
    parser.add_argument("--lr-min", type=float, default=0.0, help="学习率下限（线性衰减时不低于此值）")

    # 自适应熵系数
    parser.add_argument("--adaptive-entropy", action="store_true", help="启用自适应熵系数")
    parser.add_argument("--target-entropy", type=float, default=1.0, help="目标熵")
    parser.add_argument("--entropy-band", type=float, default=0.3, help="熵允许波动范围（±band）")
    parser.add_argument("--ent-coef-min", type=float, default=0.01, help="ent_coef下限")
    parser.add_argument("--ent-coef-max", type=float, default=0.1, help="ent_coef上限")
    parser.add_argument("--ent-adapt-interval", type=int, default=10, help="每N轮调整一次ent_coef")

    # PPO 超参
    parser.add_argument("--gamma", type=float, default=config.ppo.gamma)
    parser.add_argument("--gae-lambda", type=float, default=config.ppo.gae_lambda)
    parser.add_argument("--clip-range", type=float, default=config.ppo.clip_range)
    parser.add_argument("--ent-coef", type=float, default=config.ppo.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=config.ppo.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=config.ppo.max_grad_norm)

    # 奖励塑形
    parser.add_argument("--shaping", action="store_true", help="启用奖励塑形（时间惩罚+跳跃惩罚）")
    parser.add_argument("--time-penalty", type=float, default=0.01, help="每步时间惩罚（鼓励快速通关）")
    parser.add_argument("--jump-penalty", type=float, default=0.02, help="每次跳跃惩罚（防止一直跳）")

    # 奖励归一化
    parser.add_argument("--normalize-reward", action="store_true",
                        help="启用奖励归一化（SB3 VecNormalize 风格，稳定 advantage 估计）")

    # 模型
    parser.add_argument("--model-size", type=str, default="nature", choices=["nature", "large"],
                        help="模型大小：nature(168万, SB3兼容) / large(424万, 多关训练)")

    # 流水线
    parser.add_argument("--no-pipeline", action="store_true", help="禁用流水线（串行收集更新）")

    # 环境
    parser.add_argument("--vec-env-type", type=str, default="subproc", choices=["dummy", "subproc"])
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--multi-level", action="store_true")
    parser.add_argument("--multi-worlds", type=int, nargs="+", default=None)
    parser.add_argument("--multi-stages", type=int, nargs="+", default=None)
    parser.add_argument("--eval-stages", type=int, nargs="+", default=None,
                        help="评估用的关卡 stage 列表（默认与训练关相同，多关卡模式下可指定不同的测试关）")
    parser.add_argument("--seed", type=int, default=config.env.seed)

    # 日志与保存
    parser.add_argument("--save-freq", type=int, default=config.train.save_freq)
    parser.add_argument("--eval-freq", type=int, default=config.train.eval_freq)
    parser.add_argument("--log-interval", type=int, default=config.train.log_interval)

    # 加载模型
    parser.add_argument("--load-model", type=str, default=None, help="从 checkpoint 加载模型继续训练")

    return parser.parse_args()


# ─── 训练主函数 ─────────────────────────────────────────────────────

def main() -> None:
    """训练主函数。

    流程：
      1. 解析参数，设置种子和设备
      2. 创建训练环境（和评估环境，多关卡模式下）
      3. 创建模型、PPO、收集器、流水线调度器
      4. 训练循环：学习率调度 → 收集一轮 → 日志 → 保存/评估
      5. 保存最终模型，关闭资源
    """
    args = parse_args()

    # 设置种子和设备
    set_seed(args.seed)
    device = get_device()

    print("=" * 70)
    print("=== 马里奥 PPO 训练（v2 流水线版）===")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"总步数: {args.total_timesteps:,}")
    print(f"并行环境数: {args.n_envs}")
    print(f"n_steps: {args.n_steps}（每环境）, 每轮收集 {args.n_envs * args.n_steps:,} 步")
    print(f"batch_size: {args.batch_size}, n_epochs: {args.n_epochs}")
    print(f"学习率: {args.learning_rate} ({args.lr_schedule})", end="")
    if args.lr_min > 0:
        print(f", 下限: {args.lr_min}")
    else:
        print()
    if args.adaptive_entropy:
        print(f"自适应熵: 启用（目标={args.target_entropy}, 范围±{args.entropy_band}, "
              f"ent_coef范围[{args.ent_coef_min}, {args.ent_coef_max}]）")
    print(f"流水线: {'启用（收集更新并行）' if not args.no_pipeline else '禁用（串行）'}")
    print(f"环境类型: {args.vec_env_type}")
    if args.multi_level:
        worlds = args.multi_worlds if args.multi_worlds else list(range(1, 9))
        stages = args.multi_stages if args.multi_stages else [1, 2, 3]
        print(f"多关卡训练: {len(worlds)}章×{len(stages)}关 = {len(worlds)*len(stages)}关 "
              f"(worlds={worlds}, stages={stages})")
    if args.shaping:
        print(f"奖励塑形: 启用（时间惩罚={args.time_penalty}/步, 跳跃惩罚={args.jump_penalty}/次）")
    if args.normalize_reward:
        print(f"奖励归一化: 启用（SB3 VecNormalize 风格，discounted return running std）")

    # 创建环境
    print(f"\n创建 {args.n_envs} 个环境...")
    envs = make_vec_envs(
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_type=args.vec_env_type,
        world=args.world,
        stage=args.stage,
        multi_level=args.multi_level,
        multi_worlds=args.multi_worlds,
        multi_stages=args.multi_stages,
        frame_skip=config.env.frame_skip,
        stack_size=config.env.stack_size,
        image_size=config.env.image_size,
        shaping=args.shaping,
        time_penalty=args.time_penalty,
        jump_penalty=args.jump_penalty,
    )

    # 获取观测空间形状
    obs_shape = envs.observation_space.shape
    n_actions = envs.action_space.n
    print(f"观测空间: {obs_shape}")
    print(f"动作空间: {n_actions} 个动作")

    # 创建评估环境（多关卡模式下，默认用训练关；可通过 --eval-stages 指定不同测试关）
    eval_envs = None
    if args.multi_level:
        eval_worlds = args.multi_worlds if args.multi_worlds else list(range(1, 9))
        eval_stages = args.eval_stages if args.eval_stages else (args.multi_stages if args.multi_stages else [1, 2, 3])
        print(f"\n创建评估环境（测试关：stages={eval_stages}）...")
        eval_envs = make_vec_envs(
            n_envs=min(args.n_envs, 8),
            seed=args.seed + 1000,
            vec_env_type=args.vec_env_type,
            world=args.world,
            stage=args.stage,
            multi_level=True,
            multi_worlds=eval_worlds,
            multi_stages=eval_stages,
            frame_skip=config.env.frame_skip,
            stack_size=config.env.stack_size,
            image_size=config.env.image_size,
            shaping=args.shaping,
            time_penalty=args.time_penalty,
            jump_penalty=args.jump_penalty,
        )
        print(f"评估环境: {len(eval_worlds)}章×{len(eval_stages)}关(stages={eval_stages}) = {len(eval_worlds)*len(eval_stages)}测试关")

    # 创建模型
    model = ActorCritic(
        input_channels=obs_shape[0],
        n_actions=n_actions,
        model_size=args.model_size,
    ).to(device)
    print(f"模型大小: {args.model_size} | 参数量: {count_parameters(model):,}")

    # 加载模型（如果指定）
    if args.load_model:
        print(f"从 checkpoint 加载模型: {args.load_model}")
        model.load(args.load_model, device)

    # 创建 PPO
    ppo = PPO(
        model=model,
        device=device,
        lr=args.learning_rate,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
    )

    # 创建收集器
    collector = Collector(
        model=model,
        envs=envs,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        obs_shape=obs_shape,
        device=device,
        normalize_reward=args.normalize_reward,
        gamma=args.gamma,
    )

    # 创建流水线调度器
    pipeline = PipelinePPO(
        train_model=model,
        ppo=ppo,
        collector=collector,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        obs_shape=obs_shape,
        device=device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        enabled=not args.no_pipeline,
    )

    # 日志
    logger = TrainLogger(log_dir=str(config.paths.log_dir), use_tensorboard=True)

    # 训练循环
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    start_time = time.time()
    last_save = 0
    last_eval = 0
    best_reward = -float("inf")

    # 学习率调度
    lr_fn = make_lr_schedule(args.learning_rate, args.lr_min, args.lr_schedule)

    # 自适应熵系数状态
    current_ent_coef = args.ent_coef

    try:
        while pipeline.total_timesteps < args.total_timesteps:
            # 更新学习率
            progress = 1.0 - pipeline.total_timesteps / args.total_timesteps
            current_lr = lr_fn(max(progress, 0.0))
            for param_group in ppo.optimizer.param_groups:
                param_group["lr"] = current_lr

            # 收集一轮（流水线模式下自动和更新并行）
            if args.no_pipeline:
                stats = pipeline.collect_serial()
            else:
                stats = pipeline.collect_one_iteration()

            total_steps = stats["total_timesteps"]
            iteration = stats["iteration"]

            # 记录日志
            logger.add_scalar("rollout/ep_rew_mean", stats.get("ep_rew_mean", 0), total_steps)
            logger.add_scalar("rollout/ep_len_mean", stats.get("ep_len_mean", 0), total_steps)
            logger.add_scalar("rollout/ep_x_pos_mean", stats.get("ep_x_pos_mean", 0), total_steps)
            logger.add_scalar("rollout/ep_win_rate", stats.get("ep_win_rate", 0), total_steps)
            logger.add_scalar("time/collect_time", stats.get("collect_time", 0), total_steps)
            logger.add_scalar("train/learning_rate", current_lr, total_steps)

            update_metrics = stats.get("update_metrics")
            if update_metrics:
                # 自适应熵系数调整
                if args.adaptive_entropy and iteration % args.ent_adapt_interval == 0:
                    current_entropy = update_metrics.get("entropy", args.target_entropy)
                    if current_entropy < args.target_entropy - args.entropy_band:
                        # 熵太低，增加ent_coef鼓励探索
                        current_ent_coef = min(current_ent_coef * 1.15, args.ent_coef_max)
                        ppo.ent_coef = current_ent_coef
                    elif current_entropy > args.target_entropy + args.entropy_band:
                        # 熵太高，减少ent_coef促进收敛
                        current_ent_coef = max(current_ent_coef / 1.15, args.ent_coef_min)
                        ppo.ent_coef = current_ent_coef

                logger.add_scalar("train/policy_loss", update_metrics.get("policy_loss", 0), total_steps)
                logger.add_scalar("train/value_loss", update_metrics.get("value_loss", 0), total_steps)
                logger.add_scalar("train/entropy", update_metrics.get("entropy", 0), total_steps)
                logger.add_scalar("train/approx_kl", update_metrics.get("approx_kl", 0), total_steps)
                logger.add_scalar("train/clip_fraction", update_metrics.get("clip_fraction", 0), total_steps)
                logger.add_scalar("train/explained_variance", update_metrics.get("explained_variance", 0), total_steps)
                logger.add_scalar("train/loss", update_metrics.get("loss", 0), total_steps)
                logger.add_scalar("train/ent_coef", current_ent_coef, total_steps)
                logger.add_scalar("time/update_time", update_metrics.get("update_time", 0), total_steps)

            # 打印进度
            if iteration % args.log_interval == 0 or iteration == 1:
                elapsed = time.time() - start_time
                fps = total_steps / max(elapsed, 1)
                remaining = (args.total_timesteps - total_steps) / max(fps, 1)

                print("-" * 70)
                print(f"[进度] {total_steps:>10,}/{args.total_timesteps:,} ({total_steps/args.total_timesteps*100:5.1f}%) | "
                      f"已用: {elapsed/60:5.1f}min | 剩余: {remaining/60:5.1f}min | fps: {fps:6.0f}")
                print(f"[rollout] 均奖励: {stats.get('ep_rew_mean', 0):7.1f} | "
                      f"均长度: {stats.get('ep_len_mean', 0):6.1f} | "
                      f"均x_pos: {stats.get('ep_x_pos_mean', 0):7.1f} | "
                      f"通关率: {stats.get('ep_win_rate', 0)*100:5.1f}%")

                if update_metrics:
                    ent_coef_str = f" | ent_coef: {current_ent_coef:.4f}" if args.adaptive_entropy else ""
                    print(f"[train]   熵: {update_metrics.get('entropy', 0):5.3f} | "
                          f"KL: {update_metrics.get('approx_kl', 0):7.5f} | "
                          f"clip: {update_metrics.get('clip_fraction', 0):5.3f} | "
                          f"V质量: {update_metrics.get('explained_variance', 0):5.3f} | "
                          f"lr: {current_lr:.2e}{ent_coef_str}")

                pipe_stats = pipeline.get_pipeline_stats()
                if pipe_stats.pipeline_enabled and pipe_stats.avg_collect_time > 0:
                    print(f"[pipeline] 收集: {pipe_stats.avg_collect_time:.2f}s | "
                          f"更新: {pipe_stats.avg_update_time:.2f}s | "
                          f"理论加速比: {pipe_stats.speedup:.2f}x")

            # 保存 checkpoint
            if total_steps - last_save >= args.save_freq:
                last_save = total_steps
                checkpoint_path = config.paths.checkpoint_dir / f"mario_ppo_{total_steps}_steps.pt"
                ppo.save(str(checkpoint_path))
                print(f"[保存] 第 {total_steps:,} 步，保存到 {checkpoint_path.name}")

            # 评估
            if total_steps - last_eval >= args.eval_freq:
                last_eval = total_steps
                eval_env = eval_envs if eval_envs is not None else envs
                eval_label = f"测试关stages={eval_stages}" if eval_envs is not None else "训练关"
                print(f"[评估] 第 {total_steps:,} 步，评估 {config.train.eval_episodes} 局（{eval_label}）...")
                eval_results = evaluate(model, eval_env, n_episodes=config.train.eval_episodes, device=device)
                print(f"[评估结果] 均奖励: {eval_results.mean_reward:.1f} ± {eval_results.std_reward:.1f} | "
                      f"均长度: {eval_results.mean_length:.1f} | "
                      f"均x_pos: {eval_results.mean_x_pos:.1f} | "
                      f"通关率: {eval_results.win_rate*100:.1f}%")

                logger.add_scalar("eval/mean_reward", eval_results.mean_reward, total_steps)
                logger.add_scalar("eval/mean_length", eval_results.mean_length, total_steps)
                logger.add_scalar("eval/win_rate", eval_results.win_rate, total_steps)

                if eval_results.mean_reward > best_reward:
                    best_reward = eval_results.mean_reward
                    best_path = config.paths.checkpoint_dir / "best_model.pt"
                    ppo.save(str(best_path))
                    print(f"[评估] 新最佳奖励: {best_reward:.1f}，保存到 best_model.pt")

    except KeyboardInterrupt:
        print("\n\n用户中断，保存最终模型...")

    # 保存最终模型
    final_path = config.paths.checkpoint_dir / "mario_ppo_final.pt"
    ppo.save(str(final_path))
    print(f"最终模型保存到: {final_path}")

    # 关闭
    pipeline.close()
    if eval_envs is not None:
        eval_envs.close()
    logger.close()

    elapsed = time.time() - start_time
    print(f"\n训练完成！总步数: {pipeline.total_timesteps:,} | 总时间: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
