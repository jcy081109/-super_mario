# 马里奥强化学习项目 · 设计文档

## 1. 项目结构

```
mario-rl/
├── mario_rl/                  # 核心包
│   ├── __init__.py
│   ├── config.py              # 全局配置（环境/PPO/模型/训练/路径）
│   ├── env_wrappers.py        # 环境 Wrapper 管线 + RewardShaping
│   ├── model.py               # CNN 策略网络（自定义轻量版 + SB3 NatureCNN）
│   ├── train.py               # 训练入口（4环境+奖励归一化+渲染+进度回调）
│   ├── evaluate.py            # 评估/推理演示（渲染+录屏）
│   ├── watch.py               # 训练时实时观看（自动刷新最新checkpoint）
│   ├── utils.py               # 工具函数（种子/设备/日志）
│   └── tests/
│       ├── __init__.py
│       └── test_wrappers.py   # 单元测试（12 项）
├── docs/
│   └── phase1-requirements-and-selection.md  # 阶段一：需求与技术选型
├── checkpoints/                # 模型保存 + vecnormalize.pkl
├── logs/                       # TensorBoard 日志
├── videos/                     # 录屏
├── DESIGN.md                   # 本文档
├── README.md                   # 项目总入口
├── requirements.txt            # 锁定依赖版本
├── verify_env.py               # 阶段二：环境验证
└── verify_pipeline.py          # 阶段三：全管线验证
```

## 2. 环境 Wrapper 管线

```
原始帧 (240, 256, 3) uint8
  → SkipFrame(4)               每 4 帧执行一次动作，累计奖励
  → GrayScaleObservation        RGB → 灰度 (240, 256) uint8
  → ResizeObservation(84)       opencv 缩放 (84, 84) uint8
  → [可选] RewardShaping         时间惩罚 + 跳跃惩罚（--shaping 开启）
  → FrameStack(4)               堆叠 4 帧 (4, 84, 84) uint8
  → NormalizeObservation         像素 /255 → (4, 84, 84) float32 [0,1]
  → Monitor                      记录 episode 奖励/长度/x_pos/通关率
  → [训练时] VecNormalize        奖励归一化（norm_obs=False, norm_reward=True）
```

最终观测空间 `(4, 84, 84) float32`，符合 SB3 `CnnPolicy` 的 CHW 格式。

### 关键设计决策

1. **归一化必须在 FrameStack 之后**：`FrameStack` 用 `observation_space.dtype` 强制转换数组。若先归一化为 float32 但 `observation_space` 仍是 uint8，堆叠时会把 [0,1] 强制转回 uint8（全变 0）。自定义 `NormalizeObservation` 正确更新 `observation_space.dtype=np.float32`。

2. **不用 gym 内置 `TransformObservation` 做归一化**：它不更新 `observation_space`，导致上述 dtype 不一致问题。

3. **GrayScaleObservation(keep_dim=False)**：输出 (H, W) 无通道维，使 FrameStack 后直接得到 (4, 84, 84)，符合 SB3 `CnnPolicy` 的 CHW 格式，无需额外转置。

4. **动作空间 SIMPLE_MOVEMENT (7 动作)**：`[NOOP, right, right+A, right+B, right+A+B, A, left]`。比 RIGHT_ONLY(5) 表达力强，比完整 12 动作简单。

5. **Monitor 包装**：SB3 的 `rollout/ep_rew_mean` 依赖 Monitor 记录的 episode 信息。不加 Monitor 则 TensorBoard 中没有 rollout 数据。`info_keywords=("x_pos", "flag_get")` 额外记录前进距离和通关状态。

6. **VecNormalize 只归一化奖励**：`norm_obs=False`（观测已在 Wrapper 中归一化到 [0,1]，避免重复归一化），`norm_reward=True`（解决马里奥奖励尺度过大导致值函数学不会、策略不更新的问题）。VecNormalize 只在训练时包装，评估/推理时不需要。

## 3. 训练架构

### 多环境并行

- 默认 4 环境并行（`DummyVecEnv`），每个环境不同种子增加多样性
- 4 个观测拼成 batch 一次性喂给 GPU，GPU 利用率从 ~15% 提升到 ~50%+
- fps 从单环境 ~81 提升到 4 环境 ~150+
- n_steps=1024 是每环境步数，每轮实际收集 1024×4=4096 步

### 奖励归一化（VecNormalize）

**问题**：马里奥原生奖励前进距离每步 +1~+5，一局几千。值函数初始化时输出接近 0，预测误差巨大（value_loss=9.35），导致优势函数算不准，策略梯度极小（2.6e-6），策略不更新。

**解决**：`VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0)`，用 running mean/std 自动缩放奖励。

**效果**：approx_kl 从 1e-6 → 0.01，entropy 从 0.01 → 1.9，explained_variance 从 -0.016 → 0.75。

### 回调（Callbacks）

| 回调 | 功能 | 触发频率 |
|---|---|---|
| `CheckpointCallback` | 保存模型 checkpoint | 每 50000 步 |
| `EvalCallback` | 评估模型性能，保存最佳模型 | 每 100000 步 |
| `RenderCallback` | 实时渲染游戏画面（弹窗） | 每 N 步（默认4），`--render` 开启 |
| `ProgressCallback` | 终端打印实时进度 + 记录 x_pos/通关率到 TensorBoard | 每 2000 步 |

### ProgressCallback 设计

每 2000 步打印：
```
[进度] 50,000/500,000 (10.0%) | 已用: 5.2min | 剩余: 47min | fps: 160 | 均奖励: 850, 均x_pos: 1200, 通关率: 5%
```

同时用 `self.logger.record()` 手动将 `ep_x_pos_mean` 和 `ep_win_rate` 写入 TensorBoard（SB3 1.8.0 不会自动记录 Monitor 的 info_keywords）。

## 4. CNN 模型设计

### 默认：SB3 内置 NatureCNN

- 3 层卷积：32→64→64（kernel 8/4/3, stride 4/2/1）
- Flatten → FC 512
- 总参数量：约 168 万
- 符合"2-3 层卷积"要求，稳定可靠

### 可选：自定义轻量 CNN

- 3 层卷积：16→32→32（同 kernel/stride）
- Flatten → FC 256
- 算力紧张时切换：`--custom-cnn`

### 关键设计决策

**`normalize_images=False`**：SB3 的 NatureCNN 默认期望 uint8 [0,255] 并内部归一化。我们已在 Wrapper 中归一化到 [0,1]，必须传 `normalize_images=False`，否则 SB3 会把 [0,1] 再除以 255 变成接近 0。

## 5. PPO 超参

| 超参 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 3e-4 | SB3 默认 |
| n_steps | 1024 | 每环境每次更新收集步数 |
| batch_size | 64 | minibatch |
| n_epochs | 4 | 每次更新 epoch |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE |
| clip_range | 0.2 | PPO 策略剪辑 |
| ent_coef | 0.01 | 熵系数，鼓励探索；策略坍缩时可升到 0.05（`--ent-coef`） |
| vf_coef | 0.5 | 值函数系数 |
| max_grad_norm | 0.5 | 梯度裁剪 |
| n_envs | 4 | 并行环境数（`--n-envs`） |
| total_timesteps | 1,000,000 | 先 100 万步看效果 |

### 学习率调度（`--lr-schedule`）

SB3 的 `learning_rate` 参数支持固定值或 callable 函数。函数签名为 `lr(progress_remaining: float) -> float`，其中 `progress_remaining` 从 1.0（训练开始）线性降到 0.0（训练结束）。

| 模式 | 说明 | 公式 |
|---|---|---|
| `constant`（默认） | 固定学习率 | `lr = initial_lr` |
| `linear` | 线性衰减到 0 | `lr = initial_lr × progress_remaining` |

**为什么用线性衰减**：训练后期 approx_kl 和 clip_fraction 会持续上升（学习率偏高导致策略更新剧烈），线性衰减能在后期自动降低学习率，精细调优，提高稳定性。

**微调注意**：`PPO.load()` 加载模型后，内部 `lr_schedule` 仍是旧的调度，必须调用 `model._setup_lr_schedule()` 重建，否则新的学习率调度不生效。

### 微调（`--load-model`）

从已有 checkpoint 加载模型继续训练，适用于：
- 训练中断后恢复
- 在已有模型基础上精细调优
- 换不同超参继续训练

**关键流程**：
1. 创建环境并用 VecNormalize 包装（如找到配对 pkl 则加载已有统计信息）
2. `PPO.load(path, env=env, device=device)` 加载模型
3. 覆盖关键超参（learning_rate, ent_coef, n_steps 等）
4. 调用 `model._setup_lr_schedule()` 重建学习率调度
5. `model.learn(total_timesteps=额外步数)` 继续训练

**VecNormalize 统计信息配对**：
- CheckpointCallback 保存 `mario_ppo_{N}_steps.zip` 时，同时保存 `mario_ppo_vecnormalize_{N}_steps.pkl`
- 微调时用正则从模型文件名提取步数，自动查找配对 pkl
- 找不到配对 pkl 时回退查找 `vecnormalize.pkl`（旧格式）
- 都找不到则从零开始积累统计信息（前几千步略有波动）

**微调推荐超参**：
- 学习率：1e-4 ~ 5e-5（从头训练用 3e-4）
- 学习率调度：linear
- 额外步数：10万 ~ 50万

### 自适应熵系数（`--adaptive-entropy`）

类似 SAC 算法的自动温度调节，通过自定义 Callback 在训练过程中动态调整 ent_coef。

**原理**：ent_coef 控制策略探索程度。熵持续下降说明策略在坍缩（动作越来越确定），此时升高 ent_coef 鼓励探索；熵过高说明探索过度，降低 ent_coef 让策略收敛。

**调整逻辑**（每次 rollout 结束后，即每 4096 步）：
1. 从 logger 读取 `train/entropy_loss`（负的熵），转为正熵
2. 与目标范围 `[target_entropy - band, target_entropy + band]` 比较
3. 熵 < 下限 → ent_coef × 1.15（升高，防坍缩）
4. 熵 > 上限 → ent_coef ÷ 1.15（降低，促收敛）
5. ent_coef 限制在 `[0.01, 0.1]`

**默认参数**：
| 参数 | 默认值 | 含义 |
|---|---|---|
| target_entropy | 1.0 | 目标熵（7动作均匀分布=1.95，收敛后通常0.5~1.0） |
| entropy_band | 0.3 | 允许波动范围，即目标±0.3 |
| min_ent_coef | 0.01 | 下限（保证基本探索，训练初期熵高时不会降得太低） |
| max_ent_coef | 0.1 | 上限（防止过度探索） |
| adjustment_rate | 1.15 | 每次调整幅度 |

**为什么 min_ent_coef=0.01**：训练初期策略随机，熵≈1.95，远高于目标上限1.3，系统会持续降低 ent_coef。如果下限是 0.005，约 2 万步就降到下限，探索不足。设为 0.01 保证初期基本探索，等于"只防坍缩，不干预初期探索"。

**500万步训练验证**：
- 熵从 1.95 稳定降到 0.75，刚好在目标范围下限（0.7）附近
- 说明自适应熵起到了作用：当熵降到 0.7 以下时升高 ent_coef，防止继续下降
- 模型完全收敛，评估通关率 100%

## 6. 奖励设计

### 默认：原生奖励

使用 gym-super-mario-bros 原生奖励：
- 前进距离（x_pos 变化）——主要奖励信号
- 击杀敌人 +1
- 收集金币 +1
- 通关（flag_get）+15
- 死亡/超时 -15

### 可选塑形（RewardShaping，`--shaping` 开启）

| 惩罚 | 默认值 | 作用 |
|---|---|---|
| 时间惩罚 `time_penalty` | 0.01/步 | 鼓励快速前进，不原地磨蹭 |
| 跳跃惩罚 `jump_penalty` | 0.02/次 |  discouraging 过度跳跃，解决"一直跳掉坑里" |

**跳跃惩罚原理**：SIMPLE_MOVEMENT 中包含跳跃的动作是 2(right+A)、4(right+A+B)、5(A)。每次执行这些动作时扣除 `jump_penalty`，让模型学会只有在需要时才跳，而不是一直按右+跳。

**注意**：跳跃惩罚不宜过大（建议 0.01~0.05），否则模型会完全不跳，无法越过障碍。

### 训练时奖励归一化

VecNormalize 在塑形奖励基础上再做 running mean/std 归一化，确保奖励尺度稳定。

## 7. 可视化设计

### 评估时渲染 + 录屏（evaluate.py）

- `--render`：实时弹窗渲染游戏画面
- `--record`：录制 mp4 视频保存到 `videos/`
- `--speed`：慢放/快放（0.5=慢放，2.0=快放）

### 训练时实时观看（watch.py）

- 自动加载 `checkpoints/` 下最新模型
- 实时渲染玩游戏
- 每局结束后自动检测并加载更新的 checkpoint
- 训练在后台跑的同时，前台开 watch 可实时看 AI 进步过程

### 训练时内嵌渲染（train.py --render）

- `RenderCallback` 在训练过程中定期渲染第一个环境的画面
- `--render-freq` 控制渲染频率（默认每 4 步一帧）
- 会拖慢训练 30-50%，建议短训调试用

## 8. 关键踩坑记录

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | gym 0.21 安装失败 | pip 26 拒绝旧元数据 (`opencv-python>=3.`) | 降级 pip 23.3.2 + setuptools 65.7.0 |
| 2 | numpy 被升到 2.x | SB3 extra 依赖拉取 | 强制锁 numpy==1.26.4 |
| 3 | opencv 5.0 要求 numpy>=2 | 版本冲突 | 降级 opencv 4.10.0.84 |
| 4 | 观测全黑 (uint8 [0,1]) | 归一化在 FrameStack 之前，dtype 不一致 | 归一化移到 FrameStack 之后，自定义 NormalizeObservation |
| 5 | PPO 初始化断言 | SB3 期望 uint8 图像，内部归一化 | policy_kwargs 传 `normalize_images=False` |
| 6 | 推理 action 类型错误 | model.predict 返回 ndarray，JoypadSpace 需要 int | step 前 `int(action)` |
| 7 | RTX 5060 不识别 | Blackwell (sm_120) 需 PyTorch cu128+ | 安装 torch 2.11.0+cu128 |
| 8 | 策略不更新（approx_kl=1e-6, entropy=0.01） | 马里奥奖励尺度过大，值函数学不会 | VecNormalize 奖励归一化（norm_reward=True） |
| 9 | TensorBoard 没有 rollout/ep_rew_mean | 训练环境缺少 Monitor wrapper | make_env 最后加 Monitor(env, info_keywords=...) |
| 10 | 多环境训练的模型加载到单环境报错 | `set_env` 要求 n_envs 一致 | 改用 `PPO.load(path, env=env)` 方式加载 |
| 11 | Monitor info_keywords 不写入 TensorBoard | SB3 1.8.0 logger 不自动记录 info_keywords | ProgressCallback 中用 `self.logger.record()` 手动记录 |
| 12 | 训练时终端长时间无输出 | log_interval=10，4环境下每 40960 步才打印 | log_interval=1 + ProgressCallback 每 2000 步打印 |
| 13 | 模型一直跳导致掉坑里 | 策略坍缩到"右+跳"，跳跃无成本 | RewardShaping 加跳跃惩罚 + ent_coef 升到 0.05 |
| 14 | EvalCallback 评估时崩溃（AssertionError） | 训练环境用 VecNormalize 包装，评估环境没有，sync_envs_normalization 发现包装不一致 | 评估环境也用 VecNormalize 包装（norm_reward=False, training=False） |
| 15 | Checkpoint 保存频率远低于预期（4环境下每20万步才保存） | SB3 Callback 用 n_calls 计数，多环境下 n_calls=总步数/n_envs，save_freq=50000 实际对应20万总步 | save_freq 和 eval_freq 都除以 n_envs |
| 16 | watch 每局完全一样（reward/steps/x_pos 全相同） | deterministic 默认 True，贪心策略+固定种子=每局动作序列完全相同 | watch 默认改成 deterministic=False（随机策略），想看最优表现加 --deterministic |
| 17 | 微调时学习率不衰减（train/learning_rate 一直是初始值） | PPO.load() 加载模型后，内部 lr_schedule 仍是旧的固定值调度，只设置 model.learning_rate 不够 | 加载模型后必须调用 model._setup_lr_schedule() 重建调度函数 |
| 18 | 微调时报 AttributeError: 'PPO' object has no attribute 'set_vec_normalize_env' | SB3 PPO 没有 set_vec_normalize_env 方法，只有 get_vec_normalize_env | 改为在 PPO.load 之前就用 VecNormalize.load() 加载统计信息并包装好环境，再把包装好的环境传给 PPO.load(env=env) |
| 19 | 微调时只查找 vecnormalize.pkl，不识别带步数的配对 pkl | CheckpointCallback 保存的 pkl 文件名是 {prefix}_vecnormalize_{steps}_steps.pkl，不是固定的 vecnormalize.pkl | 用正则从模型文件名提取步数，自动构造配对 pkl 文件名查找，找不到再回退到 vecnormalize.pkl |
| 20 | 微调时报 NameError: name 'Path' is not defined | train.py 使用了 Path 但未导入 | 加 from pathlib import Path |
| 21 | 自适应熵 Callback 无法实例化（TypeError: Can't instantiate abstract class） | SB3 BaseCallback 是抽象基类，要求子类必须实现 _on_step 方法，AdaptiveEntropyCallback 只实现了 _on_rollout_end | 加 def _on_step(self) -> bool: return True |

## 9. 验证结果

- **环境验证**（阶段二）：所有包导入正常，随机策略 50 步闭环无异常
- **管线验证**（阶段三）：环境→PPO初始化→128步训练→保存加载→推理20步，全流程通过
  - 模型参数量：1,688,232
- **单元测试**：12/12 通过
- **4环境+奖励归一化验证**（阶段四）：
  - 4环境创建 + VecNormalize 包装正常
  - PPO 训练 16384 步，指标正常：approx_kl=0.01, entropy=1.9, explained_variance=0.75
  - 模型保存 + VecNormalize 统计信息保存正常
  - 多环境模型加载到单环境推理正常
- **可视化验证**：rgb_array 渲染、human 窗口渲染、100步录屏，全部正常
- **x_pos指标验证**：16384步训练中，ep_x_pos_mean 从 773 → 1163，通关率从 0% → 10%
- **跳跃惩罚验证**：--shaping --jump-penalty 0.02 --ent-coef 0.05 训练正常，无报错
- **80 万步训练成果**（阶段四）：
  - 通关率：10%
  - 平均 x_pos：1779 / 3161（56%）
  - 平均奖励：1684
  - 平均每局步数：211
  - explained_variance：0.86
  - 训练速度：~150 fps（4环境，RTX 5060）
- **微调功能验证**：--load-model 加载 80 万步 checkpoint + --lr-schedule linear + --learning-rate 1e-4，训练正常，VecNormalize 统计信息自动加载，学习率衰减生效
- **watch 随机策略验证**：默认 deterministic=False，每局表现不同，x_pos 从 314 到 3161（通关）均有，通关率约 4%（250局中10局通关）
- **自适应熵验证**：--adaptive-entropy 训练正常，熵从 1.95 稳定降到 0.75（目标范围下限），ent_coef 动态调整生效，无报错
- **500 万步训练成果**（阶段四最终）：
  - 通关率（训练）：90%
  - 通关率（评估5局）：100%（奖励±0.00，每局路径完全一致）
  - 平均 x_pos：3043 / 3161（96%）
  - 平均奖励：3040（训练）/ 3069（评估）
  - 平均每局步数：265
  - entropy：0.75（稳定在自适应熵目标范围下限）
  - explained_variance：0.999
  - approx_kl：4.2e-6（几乎为0，完全收敛）
  - clip_fraction：0
  - 训练速度：~170 fps（8环境，RTX 5060）
  - 训练时长：约 8 小时（480分钟）
  - 训练配置：--adaptive-entropy --lr-schedule linear --n-envs 8 --total-timesteps 5000000
- **8环境并行性能发现**：8环境相比4环境速度提升仅10-30%（不会翻倍），原因：①DummyVecEnv是串行执行环境step，不是真正并行；②GPU推理可能已饱和；③更大batch增加CPU-GPU数据传输开销
