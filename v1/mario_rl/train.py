"""
训练入口：创建环境 → 初始化 PPO → 训练（带 checkpoint）→ 保存最终模型。
支持多环境并行、奖励归一化（VecNormalize）、训练时实时渲染。

用法:
    python -m mario_rl.train                              # 默认：4环境 + 奖励归一化
    python -m mario_rl.train --total-timesteps 500000
    python -m mario_rl.train --n-envs 8                  # 8环境并行
    python -m mario_rl.train --render                    # 训练时弹窗看游戏（实时渲染）
    python -m mario_rl.train --render --render-freq 8    # 每8步渲染一帧
    python -m mario_rl.train --no-norm-reward            # 关闭奖励归一化
    python -m mario_rl.train --custom-cnn                # 轻量CNN
"""
import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import config
from .env_wrappers import make_env
from .model import get_policy_kwargs
from .utils import get_device, set_seed


class EnvFactory:
    """可序列化的环境工厂类，用于 SubprocVecEnv 多进程真并行。

    Windows 上 pickle 无法序列化 lambda/局部函数，因此用模块级类代替。
    每个子进程调用 __call__() 创建独立环境实例。
    """

    def __init__(self, seed, shaping, time_penalty, jump_penalty,
                 multi_level, multi_worlds, multi_stages, levels=None):
        self.seed = seed
        self.shaping = shaping
        self.time_penalty = time_penalty
        self.jump_penalty = jump_penalty
        self.multi_level = multi_level
        self.multi_worlds = multi_worlds
        self.multi_stages = multi_stages
        self.levels = levels

    def __call__(self):
        return make_env(
            seed=self.seed,
            shaping=self.shaping,
            time_penalty=self.time_penalty,
            jump_penalty=self.jump_penalty,
            multi_level=self.multi_level,
            multi_worlds=self.multi_worlds,
            multi_stages=self.multi_stages,
            levels=self.levels,
        )


class RenderCallback(BaseCallback):
    """训练过程中定期渲染游戏画面（弹窗显示当前第一个环境的游戏画面）。

    渲染只显示第一个环境，多环境并行时其他环境在后台跑。
    render_freq 控制渲染频率，每 N 步渲染一帧，避免渲染拖慢训练。
    """

    def __init__(self, render_freq: int = 4, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render(mode='human')
        return True


class AdaptiveEntropyCallback(BaseCallback):
    """自适应熵系数：监控策略熵，熵过低（坍缩迹象）时升高 ent_coef，熵过高时降低。

    原理：ent_coef 控制策略探索程度。熵持续下降说明策略在坍缩（动作越来越确定），
    此时升高 ent_coef 鼓励探索；熵过高说明探索过度，降低 ent_coef 让策略收敛。

    调整逻辑：每次 rollout 结束后，从 logger 读取最近的 entropy_loss（负的熵），
    转为正熵后与目标范围比较，超出范围则按 adjustment_rate 调整 ent_coef。
    """

    def __init__(self, target_entropy: float = 1.0, entropy_band: float = 0.3,
                 min_ent_coef: float = 0.01, max_ent_coef: float = 0.1,
                 adjustment_rate: float = 1.15, verbose: int = 1):
        super().__init__(verbose)
        self.target_entropy = target_entropy      # 目标熵值
        self.entropy_band = entropy_band          # 允许波动范围 [target-band, target+band]
        self.min_ent_coef = min_ent_coef          # ent_coef 下限
        self.max_ent_coef = max_ent_coef          # ent_coef 上限
        self.adjustment_rate = adjustment_rate    # 每次调整幅度（乘/除）
        self.last_adjust_step = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # 从 logger 获取最近的熵（SB3 记录的是 entropy_loss = -entropy）
        entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', None)
        if entropy_loss is None:
            return  # 第一次 rollout 还没更新，logger 中无值

        entropy = -float(entropy_loss)
        current_ent_coef = float(self.model.ent_coef)
        new_ent_coef = current_ent_coef

        if entropy < self.target_entropy - self.entropy_band:
            # 熵过低，有坍缩迹象，升高 ent_coef 鼓励探索
            new_ent_coef = min(current_ent_coef * self.adjustment_rate, self.max_ent_coef)
        elif entropy > self.target_entropy + self.entropy_band:
            # 熵过高，探索过度，降低 ent_coef 让策略收敛
            new_ent_coef = max(current_ent_coef / self.adjustment_rate, self.min_ent_coef)

        if abs(new_ent_coef - current_ent_coef) > 1e-6:
            self.model.ent_coef = new_ent_coef
            if self.verbose > 0:
                direction = "↑ 升高（防坍缩）" if new_ent_coef > current_ent_coef else "↓ 降低（促收敛）"
                print(f"[自适应熵] entropy={entropy:.3f}, ent_coef: {current_ent_coef:.4f} → {new_ent_coef:.4f} {direction}")


class TaggedCheckpointCallback(CheckpointCallback):
    """CheckpointCallback 子类，保存时打印醒目标签。"""
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"\n{'─'*60}")
            print(f"[保存] 第 {self.num_timesteps:,} 步，保存 checkpoint...")
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            print(f"{'─'*60}\n")
        return result


class TaggedEvalCallback(EvalCallback):
    """EvalCallback 子类，评估前后打印醒目标签。"""
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\n{'═'*60}")
            print(f"[评估] 第 {self.num_timesteps:,} 步，评估 {self.n_eval_episodes} 局（测试关泛化能力）...")
            print(f"{'═'*60}")
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print(f"{'═'*60}\n")
        return result


class ProgressCallback(BaseCallback):
    """实时训练进度打印：每 N 步输出当前步数、时间、预计剩余、fps、最近奖励。

    解决 SB3 默认 log_interval 太大导致终端长时间无输出的问题。
    打印内容 flush=True 确保实时显示，不被 Python stdout 缓冲。
    """

    def __init__(self, print_freq: int = 2000, total_timesteps: int = 1_000_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_print_step = 0
        self.last_print_time = None
        self.episode_rewards = []  # 最近完成的 episode 奖励
        self.episode_x_pos = []    # 最近完成的 episode 最终前进距离
        self.episode_flag_get = [] # 最近完成的 episode 是否通关

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_print_time = self.start_time

    def _on_step(self) -> bool:
        # 收集最近完成的 episode 奖励和前进距离（Monitor 会在 info["episode"] 里放这些）
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(ep_info.get("r", 0))
                self.episode_x_pos.append(ep_info.get("x_pos", 0))
                self.episode_flag_get.append(ep_info.get("flag_get", False))
                if len(self.episode_rewards) > 20:
                    self.episode_rewards.pop(0)
                if len(self.episode_x_pos) > 20:
                    self.episode_x_pos.pop(0)
                if len(self.episode_flag_get) > 20:
                    self.episode_flag_get.pop(0)

        # 每 print_freq 步打印一次进度
        if self.n_calls % self.print_freq == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            steps_done = self.num_timesteps

            # 最近一段时间的 fps
            if self.last_print_time and current_time > self.last_print_time:
                steps_since = steps_done - self.last_print_step
                time_since = current_time - self.last_print_time
                recent_fps = steps_since / time_since if time_since > 0 else 0
            else:
                recent_fps = 0

            # 预计剩余时间
            if recent_fps > 1:
                remaining = self.total_timesteps - steps_done
                eta_sec = remaining / recent_fps
                if eta_sec > 3600:
                    eta_str = f"{eta_sec / 3600:.1f}h"
                elif eta_sec > 60:
                    eta_str = f"{eta_sec / 60:.1f}m"
                else:
                    eta_str = f"{eta_sec:.0f}s"
            else:
                eta_str = "计算中"

            # 最近平均奖励、前进距离、通关率
            if self.episode_rewards:
                avg_r = np.mean(self.episode_rewards[-10:])
                avg_x = np.mean(self.episode_x_pos[-10:]) if self.episode_x_pos else 0
                win_rate = np.mean(self.episode_flag_get[-10:]) if self.episode_flag_get else 0
                reward_str = f"均奖励: {avg_r:.0f}, 均x_pos: {avg_x:.0f}, 通关率: {win_rate:.0%}"

                # 手动记录到 TensorBoard（SB3 不会自动记录 info_keywords）
                self.logger.record("rollout/ep_x_pos_mean", avg_x)
                self.logger.record("rollout/ep_win_rate", win_rate)
            else:
                reward_str = "暂无完整episode"

            # 从 logger 读取最近一次 PPO 更新的训练指标（熵、KL、值函数质量）
            train_metrics = ""
            nv = getattr(self.logger, "name_to_value", {})
            if nv:
                entropy = -nv.get("train/entropy_loss", 0)  # entropy_loss 是负数
                approx_kl = nv.get("train/approx_kl", 0)
                exp_var = nv.get("train/explained_variance", 0)
                if entropy > 0:
                    train_metrics = f" | 熵: {entropy:.2f}, KL: {approx_kl:.4f}, V质量: {exp_var:.2f}"

            percent = steps_done / self.total_timesteps * 100

            print(
                f"[进度] {steps_done:>8,}/{self.total_timesteps:,} ({percent:5.1f}%) | "
                f"已用: {elapsed / 60:5.1f}min | 剩余: {eta_str:>6} | "
                f"fps: {recent_fps:4.0f} | {reward_str}{train_metrics}",
                flush=True,
            )

            self.last_print_step = steps_done
            self.last_print_time = current_time

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="训练马里奥 PPO 智能体")
    parser.add_argument("--total-timesteps", type=int, default=config.train.total_timesteps)
    parser.add_argument("--n-steps", type=int, default=config.ppo.n_steps)
    parser.add_argument("--batch-size", type=int, default=config.ppo.batch_size)
    parser.add_argument("--learning-rate", type=float, default=config.ppo.learning_rate)
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["constant", "linear"],
                        help="学习率调度：constant=固定（默认），linear=从初始值线性衰减到0")
    parser.add_argument("--seed", type=int, default=config.env.seed)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="并行环境数（默认4，DummyVecEnv 单进程串行，GPU批量推理）")
    parser.add_argument("--custom-cnn", action="store_true", help="使用自定义轻量 CNN")
    parser.add_argument("--shaping", action="store_true", help="开启奖励塑形（时间惩罚+跳跃惩罚）")
    parser.add_argument("--time-penalty", type=float, default=0.01,
                        help="时间惩罚：每步扣多少奖励（需--shaping，默认0.01）")
    parser.add_argument("--jump-penalty", type=float, default=0.02,
                        help="跳跃惩罚：每次跳扣多少奖励（需--shaping，默认0.02，建议0.01~0.05）")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="熵系数：越大越鼓励探索（默认0.01，策略坍缩时可升到0.05）")
    parser.add_argument("--no-norm-reward", action="store_true",
                        help="关闭奖励归一化（默认开启 VecNormalize）")
    parser.add_argument("--render", action="store_true",
                        help="训练时实时渲染游戏画面（弹窗显示，会拖慢训练）")
    parser.add_argument("--render-freq", type=int, default=4,
                        help="渲染频率：每 N 步渲染一帧（默认4，越小越流畅但越慢）")
    parser.add_argument("--load-model", type=str, default=None,
                        help="从已有 checkpoint 加载模型继续微调（指定 .zip 路径，如 checkpoints/mario_ppo_800000_steps.zip）")
    parser.add_argument("--adaptive-entropy", action="store_true",
                        help="开启自适应熵系数：熵过低（坍缩迹象）时自动升高 ent_coef，熵过高时降低")
    parser.add_argument("--target-entropy", type=float, default=1.0,
                        help="自适应熵的目标熵值（默认1.0，7动作均匀分布熵=1.95）")
    parser.add_argument("--entropy-band", type=float, default=0.3,
                        help="自适应熵的允许波动范围（默认0.3，即目标±0.3）")
    parser.add_argument("--worlds", type=str, default="1,2,3,4,5,6,7,8",
                        help="参与训练的 world 列表，逗号分隔（默认全部8章'1,2,3,4,5,6,7,8'）。默认模式：每章前三关(stage1-3)训练，最后一关(stage4)测试，测试结果取平均")
    parser.add_argument("--multi-level", action="store_true",
                        help="开启多关卡随机训练：每次 reset 随机选关卡，强制模型学通用策略，提升泛化能力")
    parser.add_argument("--multi-worlds", type=str, default="1",
                        help="多关卡训练的 world 范围，逗号分隔（默认'1'，即只在 world 1；'1,2'表示 world 1和2）")
    parser.add_argument("--multi-stages", type=str, default="1,2,3,4",
                        help="多关卡训练的 stage 范围，逗号分隔（默认'1,2,3,4'，即1-1到1-4）")
    parser.add_argument("--vec-env-type", type=str, default="dummy", choices=["dummy", "subproc"],
                        help="并行环境类型：dummy=单进程串行（默认，稳定），subproc=多进程真并行（Windows上启动慢但CPU环境模拟可真正并行）")
    parser.add_argument("--train-levels", type=str, default=None,
                        help="高级：直接指定训练关卡列表，如 '1-1,1-2,2-1'（覆盖--worlds默认模式）")
    parser.add_argument("--eval-level", type=str, default=None,
                        help="高级：直接指定单个测试关卡，如 '2-4'（覆盖--worlds默认模式，默认多测试关取平均）")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="SB3 详细日志打印频率（每N次PPO更新打印一次，默认10。8环境下约8万步一次，进度条已含关键指标）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 覆盖配置
    config.env.seed = args.seed
    config.ppo.n_steps = args.n_steps
    config.ppo.batch_size = args.batch_size
    config.ppo.learning_rate = args.learning_rate
    config.train.total_timesteps = args.total_timesteps
    config.model.use_custom_cnn = args.custom_cnn
    config.ppo.ent_coef = args.ent_coef

    set_seed(args.seed)
    device = get_device()
    print(f"=== 马里奥 PPO 训练 ===")
    print(f"设备: {device}")
    print(f"总步数: {args.total_timesteps:,}")
    print(f"并行环境数: {args.n_envs}")
    print(f"n_steps: {args.n_steps}（每环境）, 每轮收集 {args.n_steps * args.n_envs:,} 步")
    print(f"batch_size: {args.batch_size}, lr: {args.learning_rate} ({args.lr_schedule})")
    print(f"CNN: {'自定义轻量' if args.custom_cnn else 'SB3 NatureCNN'}")
    print(f"奖励塑形: {'开启' if args.shaping else '关闭（默认奖励）'}")
    if args.shaping:
        print(f"  时间惩罚: {args.time_penalty}/步, 跳跃惩罚: {args.jump_penalty}/次")
    print(f"熵系数 ent_coef: {args.ent_coef}")
    print(f"奖励归一化: {'关闭' if args.no_norm_reward else '开启（VecNormalize，仅奖励）'}")
    print(f"实时渲染: {'开启（每%d步一帧）' % args.render_freq if args.render else '关闭'}")

    # 解析关卡配置：默认模式=每章前三关(stage1-3)训练，最后一关(stage4)测试取平均
    worlds_list = tuple(int(x) for x in args.worlds.split(","))

    # 训练关卡：--train-levels 优先，否则用所有 world 的 stage 1-3
    if args.train_levels:
        train_levels = tuple(
            tuple(int(x) for x in lv.strip().split("-"))
            for lv in args.train_levels.split(",")
        )
    else:
        train_levels = tuple((w, s) for w in worlds_list for s in (1, 2, 3))
    args.multi_level = True  # 训练始终用多关卡随机模式
    multi_worlds = (1,)
    multi_stages = (1, 2, 3, 4)
    print(f"训练关卡: {len(train_levels)}关 → {', '.join(f'{w}-{s}' for w, s in train_levels)}")

    # 测试关卡：--eval-level 优先（单关），否则用所有 world 的 stage 4（多关取平均）
    eval_levels = None
    eval_world, eval_stage = None, None
    if args.eval_level:
        eval_world, eval_stage = (int(x) for x in args.eval_level.strip().split("-"))
        print(f"测试关卡: {eval_world}-{eval_stage}（单关评估）")
    else:
        eval_levels = tuple((w, 4) for w in worlds_list)
        print(f"测试关卡: {len(eval_levels)}关 → {', '.join(f'{w}-4' for w in worlds_list)}（每局随机选关，结果取平均）")

    # 1. 创建训练环境（多环境并行，每个环境不同种子增加多样性）
    env_factories = [
        EnvFactory(
            seed=args.seed + i,
            shaping=args.shaping,
            time_penalty=args.time_penalty,
            jump_penalty=args.jump_penalty,
            multi_level=True,
            multi_worlds=multi_worlds,
            multi_stages=multi_stages,
            levels=train_levels,
        )
        for i in range(args.n_envs)
    ]

    if args.vec_env_type == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv
        print(f"并行环境类型: SubprocVecEnv（多进程真并行，{args.n_envs}个进程，启动较慢）")
        env = SubprocVecEnv(env_factories)
    else:
        print(f"并行环境类型: DummyVecEnv（单进程串行，{args.n_envs}个环境）")
        env = DummyVecEnv(env_factories)

    print(f"观测空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space.n} 个动作")

    # 2. 奖励归一化（VecNormalize）
    #    norm_obs=False：观测已在 Wrapper 中归一化到 [0,1]，不需重复归一化
    #    norm_reward=True：用 running mean/std 归一化奖励，解决马里奥奖励尺度过大问题
    #    clip_reward=10.0：归一化后裁剪到 [-10, 10]，防止异常奖励
    #    微调时：自动查找与模型配对的 vecnormalize pkl（如 mario_ppo_50000_steps.zip → mario_ppo_vecnormalize_50000_steps.pkl）
    if not args.no_norm_reward:
        if args.load_model:
            model_path = Path(args.load_model)
            vecnorm_path = None
            # 尝试匹配 CheckpointCallback 保存的配对 pkl：{prefix}_{steps}_steps.zip → {prefix}_vecnormalize_{steps}_steps.pkl
            import re
            m = re.match(r"^(.+)_(\d+)_steps\.zip$", model_path.name)
            if m:
                paired_pkl = model_path.parent / f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl"
                if paired_pkl.exists():
                    vecnorm_path = paired_pkl
            # 兼容旧格式：同目录下的 vecnormalize.pkl
            if vecnorm_path is None:
                legacy_pkl = model_path.parent / "vecnormalize.pkl"
                if legacy_pkl.exists():
                    vecnorm_path = legacy_pkl

            if vecnorm_path is not None:
                env = VecNormalize.load(str(vecnorm_path), env)
                env.training = True
                env.norm_reward = True
                print(f"VecNormalize 已加载已有统计信息: {vecnorm_path.name}")
            else:
                env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
                print("VecNormalize 已包装（未找到配对的 vecnormalize pkl，统计信息将重新积累）")
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            print("VecNormalize 已包装（仅奖励归一化，观测保持 [0,1]）")

    # 3. 评估环境（用于 EvalCallback）
    #    评估环境也必须用 VecNormalize 包装，否则 EvalCallback 的 sync_envs_normalization
    #    会因训练/评估环境包装结构不一致而报错（AssertionError）
    #    评估时设置 norm_reward=False（看原始奖励）和 training=False（不更新统计信息）
    #    多测试关模式：每次 reset 随机选测试关，EvalCallback 评估N局取平均=所有测试关平均
    if eval_levels is not None:
        eval_env = DummyVecEnv([lambda: make_env(
            seed=args.seed + 100,
            shaping=args.shaping,
            time_penalty=args.time_penalty,
            jump_penalty=args.jump_penalty,
            multi_level=True,
            levels=eval_levels,
        )])
    else:
        eval_env = DummyVecEnv([lambda: make_env(
            seed=args.seed + 100,
            shaping=args.shaping,
            time_penalty=args.time_penalty,
            jump_penalty=args.jump_penalty,
            world=eval_world,
            stage=eval_stage,
        )])
    if not args.no_norm_reward:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
        print("评估环境已用 VecNormalize 包装（仅结构对齐，不归一化奖励，不更新统计）")

    # 4. 初始化 PPO
    # 学习率调度：constant=固定值，linear=从初始值线性衰减到0
    # SB3 的 callable 签名：lr(progress_remaining: float) -> float
    # progress_remaining 从 1.0（训练开始）线性降到 0.0（训练结束）
    if args.lr_schedule == "linear":
        initial_lr = config.ppo.learning_rate
        learning_rate = lambda progress_remaining: initial_lr * progress_remaining
        print(f"学习率调度: 线性衰减，从 {initial_lr} 降到 0")
    else:
        learning_rate = config.ppo.learning_rate

    ppo_kwargs = dict(
        learning_rate=learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        verbose=1,
        device=device,
        tensorboard_log=str(config.paths.log_dir),
        policy_kwargs=get_policy_kwargs(),
    )

    if args.load_model:
        # 从已有 checkpoint 加载模型继续微调
        # 注意：env 已经在上面用 VecNormalize 包装好（含加载已有统计信息），直接传入即可
        print(f"从 checkpoint 加载模型: {args.load_model}")
        model = PPO.load(args.load_model, env=env, device=device)
        # 微调时覆盖关键超参（学习率、熵系数等）
        model.learning_rate = learning_rate
        model.ent_coef = config.ppo.ent_coef
        model.n_steps = config.ppo.n_steps
        model.batch_size = config.ppo.batch_size
        model.n_epochs = config.ppo.n_epochs
        # 重要：PPO.load() 不会重建 lr_schedule，必须手动调用，否则学习率调度不生效
        model._setup_lr_schedule()
        print(f"微调模式：额外训练 {args.total_timesteps:,} 步")
    else:
        model = PPO("CnnPolicy", env, **ppo_kwargs)

    print(f"模型参数量: {sum(p.numel() for p in model.policy.parameters()):,}")

    # 5. 回调：定期保存 checkpoint + 评估 + 可选实时渲染
    callbacks = []

    # 注意：SB3 的 Callback 用 n_calls 计数，多环境下 n_calls = 总步数 / n_envs
    # 所以 save_freq/eval_freq 需要除以 n_envs，才能按总步数保存/评估
    ckpt_save_freq = config.train.save_freq // args.n_envs
    eval_freq = config.train.eval_freq // args.n_envs

    checkpoint_cb = TaggedCheckpointCallback(
        save_freq=ckpt_save_freq,
        save_path=str(config.paths.checkpoint_dir),
        name_prefix="mario_ppo",
        save_vecnormalize=True,  # 同时保存 VecNormalize 统计信息，方便后续微调
    )
    callbacks.append(checkpoint_cb)
    print(f"Checkpoint 保存频率: 每 {config.train.save_freq:,} 总步（实际 callback freq={ckpt_save_freq}）")

    eval_cb = TaggedEvalCallback(
        eval_env,
        best_model_save_path=str(config.paths.checkpoint_dir / "best"),
        log_path=str(config.paths.log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=config.train.eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_cb)

    if args.render:
        render_cb = RenderCallback(render_freq=args.render_freq)
        callbacks.append(render_cb)
        print(f"RenderCallback 已启用（每 {args.render_freq} 步渲染一帧，仅显示环境0）")

    # 实时进度回调（每 2000 步打印步数/时间/ETA/fps/最近奖励）
    progress_cb = ProgressCallback(
        print_freq=2000,
        total_timesteps=config.train.total_timesteps,
    )
    callbacks.append(progress_cb)

    # 自适应熵系数回调（熵过低时升高 ent_coef 防坍缩，熵过高时降低促收敛）
    if args.adaptive_entropy:
        adaptive_ent_cb = AdaptiveEntropyCallback(
            target_entropy=args.target_entropy,
            entropy_band=args.entropy_band,
            verbose=1,
        )
        callbacks.append(adaptive_ent_cb)
        print(f"自适应熵系数已启用（目标熵={args.target_entropy}，范围±{args.entropy_band}，ent_coef 范围 0.01~0.1）")

    # 6. 训练（log_interval：每N次PPO更新打印详细指标，默认10；进度条已含关键指标）
    model.learn(
        total_timesteps=config.train.total_timesteps,
        callback=callbacks,
        log_interval=args.log_interval,
    )

    # 7. 保存最终模型 + VecNormalize 统计信息
    final_path = config.paths.checkpoint_dir / "mario_ppo_final"
    model.save(str(final_path))
    print(f"\n训练完成！最终模型已保存: {final_path}.zip")

    if not args.no_norm_reward:
        vec_norm_path = config.paths.checkpoint_dir / "vecnormalize.pkl"
        env.save(str(vec_norm_path))
        print(f"VecNormalize 统计信息已保存: {vec_norm_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
