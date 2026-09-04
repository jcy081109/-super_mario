# Mario RL v2 - 流水线 PPO 从零实现

> 基于强化学习的 AI 玩转马里奥项目，v2 版本从零实现 PPO 算法，核心特性是**双缓冲流水线收集+更新**，解决 v1 中 GPU 利用率低的瓶颈。

## 与 v1 的区别

| 特性 | v1 (SB3) | v2 (从零实现) |
|---|---|---|
| PPO 算法 | 依赖 stable_baselines3 黑盒 | 从零实现，完全可控 |
| 收集更新 | 串行（收集→更新→收集） | **流水线并行**（收集第N轮和更新第N-1轮同时进行） |
| 模型快照 | 无 | collect_model 快照，保证 on-policy 约束 |
| 双缓冲 | 无 | buffer_a / buffer_b 交替使用 |
| 自适应熵 | Callback 实现 | 内置，每N轮自动调整 ent_coef |
| 奖励塑形 | 支持 | 支持（时间惩罚+跳跃惩罚） |
| 学习率下限 | 无 | 支持 `--lr-min`，防止衰减到 0 |
| 多关卡训练集/测试集 | 分离（24训练+8测试） | 分离（单独评估环境，stage4 作测试关） |
| 可定制性 | 受 SB3 限制 | 完全可控 |

## 核心架构

### 流水线设计原理

```
时间 →
收集线程: [第1轮 收集(θ₀)] [第2轮 收集(θ₁)] [第3轮 收集(θ₂)] ...
更新线程:                  [第1轮 更新(θ₀→θ₁)] [第2轮 更新(θ₁→θ₂)] ...

关键：
- collect_model（收集用）是 train_model（训练用）的快照
- 每轮收集开始前，collect_model 同步为 train_model 最新参数
- 收集第N轮用 θ_{N-1}，更新第N-1轮用 θ_{N-2}→θ_{N-1}，两者并行不冲突
- 每批数据来自同一个固定策略，满足 on-policy 约束
```

### 模块结构

```
mario_rl/
├── __init__.py          # 版本信息 v2.0.0
├── config.py            # 全局配置（环境/PPO/流水线/训练/路径）
├── utils.py             # 工具函数（种子/设备/日志/统计/滑动窗口）
├── model.py             # NatureCNN + ActorCritic 策略网络（1,688,232参数）
├── ppo.py               # PPO 核心算法（RolloutBuffer + GAE + Clip损失 + 更新）
├── collector.py         # 多环境并行数据收集器 + Episode 统计 + make_vec_envs
├── pipeline.py          # 流水线调度器（双缓冲 + 模型快照 + 收集更新并行）
├── env_wrappers.py      # 环境 Wrapper（SkipFrame/灰度/缩放/帧堆叠/奖励塑形/RandomLevelEnv）
├── train.py             # 训练入口（含自适应熵、奖励塑形、多关卡评估）
└── watch.py             # 观看 AI 玩游戏（支持 --all-levels 32关遍历）
```

## 安装

```powershell
# 复用 v1 的 conda 环境即可（依赖完全一致）
conda activate mario-rl

# 或新建环境
conda create -n mario-rl-v2 python=3.10 -y
conda activate mario-rl-v2
pip install pip==23.3.2 setuptools==65.7.0 wheel==0.41.2
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 快速开始

### 单关卡训练（验证流水线效果）

```powershell
python -m mario_rl.train `
  --total-timesteps 2000000 `
  --n-envs 16 --n-steps 512 `
  --batch-size 256 --n-epochs 4 `
  --learning-rate 3e-4 --lr-schedule linear `
  --ent-coef 0.02 `
  --world 1 --stage 1
```

### 全关卡训练（推荐配置，防坍缩）

```powershell
python -m mario_rl.train `
  --total-timesteps 20000000 `
  --n-envs 32 --n-steps 512 `
  --batch-size 512 --n-epochs 4 `
  --learning-rate 2e-4 `
  --lr-schedule linear --lr-min 1e-5 `
  --adaptive-entropy `
  --target-entropy 1.2 `
  --entropy-band 0.3 `
  --ent-coef 0.03 `
  --ent-coef-min 0.01 `
  --ent-coef-max 0.15 `
  --ent-adapt-interval 5 `
  --shaping `
  --time-penalty 0.01 `
  --jump-penalty 0.02 `
  --multi-level `
  --vec-env-type subproc `
  --save-freq 500000 --eval-freq 500000 `
  --log-interval 5
```

**多关卡划分**：训练集 24 关（world 1-8, stage 1-3），测试集 8 关（每章 stage 4，单独评估环境）。

### 其他常用命令

```powershell
# 禁用流水线（串行模式，用于对比加速比）
python -m mario_rl.train --no-pipeline

# 从 checkpoint 继续训练
python -m mario_rl.train --load-model checkpoints/mario_ppo_500000_steps.pt

# 观看 AI 玩游戏
python -m mario_rl.watch                          # 最新模型玩 1-1
python -m mario_rl.watch --world 1 --stage 1     # 指定关卡
python -m mario_rl.watch --all-levels             # 遍历全部 32 关
python -m mario_rl.watch --speed 2.0 --no-render  # 2倍速不渲染
```

## 命令行参数

### 训练规模

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--total-timesteps` | 10,000,000 | 总训练步数（环境交互总次数） |
| `--n-envs` | 32 | 并行环境数 |
| `--n-steps` | 512 | 每环境每轮收集步数（n_envs × n_steps = 每轮总步数） |
| `--batch-size` | 512 | 更新时小批量大小 |
| `--n-epochs` | 8 | 同一批数据反复训练轮数（多关卡建议 ≤4，防过拟合） |

### 学习率

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--learning-rate` | 3e-4 | 初始学习率（多关卡建议 2e-4，更稳定） |
| `--lr-schedule` | linear | 调度方式（constant/linear） |
| `--lr-min` | 0.0 | 学习率下限（线性衰减到此值停止，建议 1e-5） |

### PPO 算法

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gamma` | 0.99 | 未来奖励折扣因子 |
| `--gae-lambda` | 0.95 | GAE 优势函数平滑参数 |
| `--clip-range` | 0.2 | PPO 裁剪范围（策略更新幅度限制 ±20%） |
| `--ent-coef` | 0.01 | 熵系数初始值（越大探索越强，多关卡建议 0.03） |
| `--vf-coef` | 0.5 | 价值损失权重 |
| `--max-grad-norm` | 0.5 | 梯度裁剪上限 |

### 自适应熵（防坍缩核心）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--adaptive-entropy` | False | 启用自适应熵调节 |
| `--target-entropy` | 1.0 | 目标熵值（太低=策略太确定，太高=太随机，多关卡建议 1.2） |
| `--entropy-band` | 0.3 | 熵允许波动范围（±band，范围内不调整） |
| `--ent-coef-min` | 0.01 | ent_coef 下限 |
| `--ent-coef-max` | 0.1 | ent_coef 上限（坍缩时可强行加探索，建议 0.15） |
| `--ent-adapt-interval` | 10 | 每几轮检查一次熵并调整（越小反应越快，建议 5） |

### 奖励塑形

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--shaping` | False | 启用奖励塑形（在原始奖励基础上加惩罚） |
| `--time-penalty` | 0.01 | 每步时间惩罚（鼓励快速通关，防止原地磨蹭） |
| `--jump-penalty` | 0.02 | 每次跳跃惩罚（防止一直乱跳） |

### 环境

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--vec-env-type` | subproc | 环境并行方式（subproc=多进程真并行 / dummy=单进程） |
| `--world` | 1 | 单关卡模式世界（1-8） |
| `--stage` | 1 | 单关卡模式关卡（1-4） |
| `--multi-level` | False | 多关卡随机训练（24训练关：world1-8, stage1-3） |
| `--multi-worlds` | None | 自定义训练世界列表（如 --multi-worlds 1 2 3） |
| `--multi-stages` | None | 自定义训练关卡列表（如 --multi-stages 1 2 3） |
| `--seed` | 42 | 随机种子 |

### 流水线、日志与保存

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--no-pipeline` | False | 禁用流水线（串行收集更新，用于对比） |
| `--save-freq` | 200,000 | 每多少步保存 checkpoint |
| `--eval-freq` | 200,000 | 每多少步评估一次（多关模式用测试关 stage4） |
| `--log-interval` | 10 | 每几轮打印一次进度 |
| `--load-model` | None | 从 checkpoint 加载继续训练 |

## PPO 算法实现细节

### 损失函数

```
总损失 = 策略损失 + vf_coef × 价值损失 + ent_coef × 熵损失

策略损失（PPO Clip）：
  L_clip = -E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]
  其中 r_t = π_new(a|s) / π_old(a|s)

价值损失：
  L_vf = MSE(V(s), R_t)

熵损失（鼓励探索）：
  L_ent = -E[H(π)]
```

### GAE 优势函数

```
A_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

### 模型结构（NatureCNN）

```
输入: (batch, 4, 84, 84)
  ↓
Conv2d(4→32, 8×8, stride=4) + ReLU  → (32, 20, 20)
Conv2d(32→64, 4×4, stride=2) + ReLU → (64, 9, 9)
Conv2d(64→64, 3×3, stride=1) + ReLU → (64, 7, 7)
Flatten → 3136
Linear(3136→512) + ReLU
  ↓
┌─────────────┴─────────────┐
│ 策略头 Linear(512→7)       │ 价值头 Linear(512→1)
│ 输出动作概率分布            │ 输出状态价值 V(s)
└─────────────────────────────┴──────────┘
```

**参数量**：1,688,232（与 v1 SB3 NatureCNN 完全一致）

### 与 SB3 PPO 的算法一致性

| 组件 | SB3 PPO | v2 自研 PPO | 一致性 |
|---|---|---|---|
| GAE 优势函数 | ✓ | ✓ | 完全一致 |
| PPO Clip 策略损失 | ✓ | ✓ | 完全一致 |
| 价值损失（MSE） | ✓ | ✓ | 完全一致 |
| 熵损失 | ✓ | ✓ | 完全一致 |
| 优势归一化 | ✓ | ✓ | 完全一致 |
| 梯度裁剪 | ✓ | ✓ | 完全一致 |
| 正交初始化 | ✓ | ✓ | 完全一致 |
| Adam 优化器（eps=1e-5） | ✓ | ✓ | 完全一致 |
| 小批量随机打乱 | ✓ | ✓ | 完全一致 |
| 多轮更新（n_epochs） | ✓ | ✓ | 完全一致 |

**结论**：v2 自研 PPO 的核心算法与 SB3 完全一致，200 万步单关训练曲线与 v1 同期表现高度吻合，无训练效果缺失。

## 训练结果与发现

### v2 首训：200 万步单关（16环境）

| 指标 | 初始 | 最终 | 趋势 |
|---|---|---|---|
| fps | 1040 | ~950 | 稳定（v1 同期 ~300fps，**3.2倍**） |
| 训练奖励 | 425 | 1637 | ↑ 持续上升 |
| 训练 x_pos | 509 | 1732 | ↑ 持续上升 |
| 训练通关率 | 0% | 6% | ↑ 开始通关 |
| 熵 | 1.46 | 0.40 | ↓ 策略收敛中 |
| V 质量 | 0.65 | 0.89 | ↑ 价值网络拟合好 |

**总耗时**：35 分钟（v1 同期约 2 小时）

### v2 全关卡训练：32 环境性能分析

| 指标 | 值 | 说明 |
|---|---|---|
| fps | ~460-600 | 32 环境多关卡，比单关慢但仍比 v1 快 |
| 收集时间/轮 | 33-36s | **绝对瓶颈**（32个NES模拟器+多进程通信） |
| 更新时间/轮 | 2.1-2.5s | 模型小+ n_epochs=4，GPU 很快 |
| 收集:更新 | ~15:1 | 严重不平衡 |
| 流水线加速比 | 1.06-1.2x | 更新被完全重叠，加速比低 |

### 策略坍缩问题与解决方案

**现象**（首次全关卡训练，n_epochs=8, lr=3e-4, ent_coef=0.01）：
- 98 万步时熵从 0.58 暴跌到 **0.097**（严重坍缩）
- KL 飙到 **0.869**（更新太激进）
- 均奖励跌到 116，均长度飙到 606（模型原地磨蹭）

**原因**：
1. n_epochs=8 + lr=3e-4 对多关卡训练太激进
2. ent_coef=0.01 初始探索不足
3. 自适应熵每 10 轮才调整，反应太慢

**解决方案（已验证有效）**：

| 调整 | 原值 | 新值 | 效果 |
|---|---|---|---|
| `--n-epochs` | 8 | 4 | 降低 KL，防过拟合 |
| `--learning-rate` | 3e-4 | 2e-4 | 更稳定的更新 |
| `--ent-coef` | 0.01 | 0.03 | 初始探索更强 |
| `--target-entropy` | 1.0 | 1.2 | 保持更高探索水平 |
| `--ent-coef-max` | 0.1 | 0.15 | 给自适应熵更大空间 |
| `--ent-adapt-interval` | 10 | 5 | 反应更快 |
| `--shaping` | 关 | 开 | 时间惩罚防原地磨蹭 |
| `--lr-min` | 0 | 1e-5 | 防止后期完全不学习 |

### 收集阶段为什么是瓶颈

```
每一步严格串行：
GPU 推理(2ms) → CPU 环境 step(30ms) → GPU 推理(2ms) → CPU step(30ms) ...
```

- 收集阶段**需要 GPU**，但只做前向推理（不做反向传播）
- 每次推理 batch=32（环境数），模型 168 万参数，GPU 几毫秒完成
- GPU 93% 时间在等 CPU 环境模拟（NES 模拟器纯 CPU）
- 无法并行：下一个 obs 依赖当前 action，当前 action 依赖当前 obs 的推理
- 这是 PPO 算法层面的硬伤，流水线只能重叠收集和更新，无法解决收集内部的串行

## 训练指标说明

| 指标 | 含义 | 健康范围 |
|---|---|---|
| `rollout/ep_rew_mean` | 平均每局奖励 | 持续上升 |
| `rollout/ep_len_mean` | 平均每局步数 | 稳定（过高=原地磨蹭） |
| `rollout/ep_x_pos_mean` | 平均到达 x 坐标 | 持续上升（1-1终点~3161） |
| `rollout/ep_win_rate` | 通关率 | 持续上升，目标>90% |
| `train/entropy` | 策略熵（探索程度） | 0.5-1.5，<0.3=坍缩风险 |
| `train/approx_kl` | 新旧策略 KL 散度 | 0.001-0.05，>0.1=更新太激进 |
| `train/clip_fraction` | 被裁剪的样本比例 | 0.1-0.3 |
| `train/explained_variance` | 价值网络解释方差 | 0.5-1.0，越高越好 |
| `train/ent_coef` | 当前熵系数（自适应时变化） | 稳定上升=在救场 |
| `time/collect_time` | 一轮收集时间 | 稳定 |
| `time/update_time` | 一轮更新时间 | 稳定 |

## 已知限制

1. **收集阶段 GPU 利用率低**：PPO 收集阶段每步必须 GPU 推理→CPU step 严格串行，GPU 大部分时间等 CPU，利用率仅 5-10%。流水线只能重叠收集和更新，无法解决收集内部串行。
2. **流水线加速比受收集:更新比例限制**：当收集远慢于更新时（如 32 环境多关卡），更新被完全重叠，加速比仅 1.1x 左右。只有收集≈更新时才能达到理论 2x。
3. **仅用 SB3 的 VecEnv**：环境封装仍依赖 stable_baselines3 的 SubprocVecEnv/DummyVecEnv，但 PPO 算法完全从零实现。
4. **Windows 多进程限制**：SubprocVecEnv 在 Windows 上启动较慢，32 进程启动需 30 秒以上。
5. **无 VecNormalize 奖励归一化**：v2 未实现 SB3 的 VecNormalize，但从实际训练看 PPO 更新正常，不影响效果。

## 项目阶段

- [x] 阶段一：需求分析与技术选型
- [x] 阶段二：环境搭建
- [x] 阶段三：结构化设计
- [x] 阶段四：v1 单关训练完成（通关率 90%/100%）
- [x] 阶段五：v2 流水线 PPO 从零实现（含自适应熵、奖励塑形、多关卡评估）
- [x] 阶段六：v2 单关验证（200 万步，fps 提升 3.2 倍）
- [ ] 阶段七：v2 全关卡 2000 万步训练（进行中）
- [ ] 阶段八：性能调优与最终交付

## 许可证

MIT
