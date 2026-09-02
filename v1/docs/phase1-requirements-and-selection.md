# 阶段一：需求分析与技术选型

## 1. 项目目标

使用强化学习训练 AI 自主玩《超级马里奥》1-1 关，在笔记本/云平台算力约束下，跑通"状态—动作—奖励"训练闭环，看到学习效果。

**核心定位**：以低算力为约束、跑通为准的强化学习落地项目，不追求快速通关或极限表现。

## 2. 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| FR-1 | 环境建模与交互 | 用 gym-super-mario-bros 将游戏抽象为标准 RL 接口（reset/step/close），实现"状态-动作-奖励"循环 |
| FR-2 | 环境简化与适配 | 聚焦角色视野裁剪、固定初始状态减少随机性、降低学习难度 |
| FR-3 | 状态预处理 | 240×256 RGB → 低维特征（灰度/缩放/归一化/帧堆叠），过滤天空等无关像素，解决计算量问题 |
| FR-4 | 强化学习算法 | PPO（Proximal Policy Optimization），通过策略剪辑保证训练稳定性 |
| FR-5 | CNN 视觉特征 | 2-3 层卷积神经网络，平衡性能与效率，适配低算力设备 |
| FR-6 | 训练与评估 | checkpoint 保存/续训、reward 曲线可视化、加载模型自动玩演示 |

## 3. 非功能需求与约束

| 约束 | 内容 |
|------|------|
| 算力 | 笔记本 CPU 可跑；有 NVIDIA GPU 则加速（本项目 RTX 5060 8GB） |
| 可复现 | 依赖版本 pin 死、随机种子固定、requirements.txt 提交 |
| 可维护 | 代码规范（lint/类型/docstring）、单元测试、git 分支协作 |
| 时间 | 训练以"流程跑通 + reward 上升"为准，不追求通关 |
| 团队 | 3-5 人协作，按项目开发流程推进 |

## 4. 范围界定

### 范围内
- 1-1 单关卡
- PPO + CNN（2-3 层卷积）
- 简化动作空间（SIMPLE_MOVEMENT 7 动作）
- 低分辨率输入（84×84 灰度）
- 笔记本/RTX 5060 训练

### 范围外
- 多关卡通关与泛化
- 极限速通或炫技级表现
- 分布式训练
- 生产环境部署

## 5. 验收标准

| 编号 | 标准 | 验证方式 |
|------|------|----------|
| A1 | 环境闭环可跑通 | 随机策略连续 step 无异常 |
| A2 | 学习效果可见 | 训练 reward 曲线呈上升趋势 |
| A3 | 模型可交付 | 模型可保存/加载，自动游玩 demo 可运行 |
| A4 | 工程质量 | 单元测试通过、README + 设计文档齐全 |

## 6. 技术选型

### 6.1 环境库版本路线

| 方案 | 组成 | 结论 |
|------|------|------|
| **路线 A（采用）** | `gym==0.21.0` + `gym-super-mario-bros==7.4.0` + `stable-baselines3==1.8.0` | 社区可复现教程最多、最稳定 |
| 路线 B | `gymnasium==0.29.x` + `stable-baselines3==2.x` + `Shimmy` | 新版生态，但兼容坑更多，留作后续迁移 |

**选择路线 A 的理由**：
- gym-super-mario-bros 7.4.0 与 gym 0.21.0 原生兼容，无需适配层
- SB3 1.8.0 是 1.x 最后一版，PPO 实现成熟稳定
- 网上可复现的锁定版本教程最多，降低踩坑成本
- 团队首次跑通优先，后续可迁移 gymnasium

### 6.2 动作空间

**选择 SIMPLE_MOVEMENT（7 动作）**：
```
[NOOP, right, right+A, right+B, right+A+B, A, left]
```

| 动作集 | 动作数 | 评价 |
|--------|--------|------|
| RIGHT_ONLY | 5 | 表达力不足，无法跳+下等组合 |
| **SIMPLE_MOVEMENT** | **7** | **平衡点，能学会通关，学习难度适中** |
| COMPLEX_MOVEMENT | 12 | 表达力强但学习难度大，低算力下收敛慢 |
| 完整 256 动作 | 256 | 红白机全按键组合，完全不必要 |

### 6.3 状态预处理管线

```
原始帧 (240,256,3) uint8
  → SkipFrame(4)               每 4 帧执行一次动作，减少交互次数
  → GrayScaleObservation        RGB → 灰度，去除颜色冗余
  → ResizeObservation(84)       缩放至 84×84（Atari 风格标准尺寸）
  → FrameStack(4)               堆叠最近 4 帧，让 CNN 感知运动
  → NormalizeObservation         像素 /255 → [0,1] float32
```

最终观测空间：`(4, 84, 84) float32`，符合 SB3 `CnnPolicy` 的 CHW 格式。

**关键设计决策**：
- **84×84**：Atari 风格 RL 标准，平衡信息保留与计算量
- **堆叠 4 帧**：单帧无法感知运动方向（起跳/下落/敌人移动），4 帧堆叠提供时序信息
- **FrameSkip 4**：马里奥 60fps，每 4 帧执行一次动作相当于 15fps 决策，足够且加速训练
- **归一化在 FrameStack 之后**：避免 dtype 不一致导致 FrameStack 强制转回 uint8（详见 DESIGN.md 踩坑记录）

### 6.4 强化学习算法

**选择 PPO（Proximal Policy Optimization）**

| 算法 | 类型 | 评价 |
|------|------|------|
| **PPO** | **on-policy 策略梯度** | **策略剪辑保证稳定，SB3 成熟实现，社区马里奥项目主流选择** |
| DQN | off-policy 值函数 | 需要大容量经验回放缓冲，内存与调参成本高，离散动作尚可但稳定性不如 PPO |
| A2C/A3C | on-policy 策略梯度 | 多进程加速，但 Windows 上 SubprocVecEnv 支持差，单环境优势不明显 |
| SAC | off-policy 最大熵 | 主要用于连续动作空间，离散动作非最优 |

**PPO 优势**：
- 策略剪辑（clip_range=0.2）限制每次更新幅度，训练稳定
- on-policy 无需大容量 replay buffer，内存占用低
- SB3 提供 `CnnPolicy` 开箱即用，支持自定义特征提取器
- 超参鲁棒性好，默认值即可工作

### 6.5 CNN 模型结构

**默认：SB3 内置 NatureCNN（3 层卷积）**

| 层 | 输入 | 输出 | kernel | stride |
|----|------|------|--------|--------|
| Conv1 | (4, 84, 84) | (32, 20, 20) | 8 | 4 |
| Conv2 | (32, 20, 20) | (64, 9, 9) | 4 | 2 |
| Conv3 | (64, 9, 9) | (64, 7, 7) | 3 | 1 |
| Flatten | (64, 7, 7) | 3136 | - | - |
| FC | 3136 | 512 | - | - |
| Action head | 512 | 7 | - | - |
| Value head | 512 | 1 | - | - |

总参数量：约 **168 万**。

**可选：自定义轻量 CNN（算力紧张时切换）**
- 3 层卷积：16→32→32（同 kernel/stride）
- FC：256
- 参数量约为 NatureCNN 的 1/4

**关键配置**：`normalize_images=False`——因为 Wrapper 已将像素归一化到 [0,1]，SB3 不应再次归一化（否则会把 [0,1] 除以 255 变成接近 0）。

### 6.6 奖励设计

**默认：使用 gym-super-mario-bros 原生奖励**
- 前进距离（v_pos 变化）——主要奖励信号
- 击杀敌人 +1
- 收集金币 +1
- 通关（flag_get）+15
- 死亡/超时 -15

**可选塑形**：`RewardShaping` Wrapper
- 时间惩罚：每步 -0.01，鼓励快速前进
- 卡住惩罚：x_pos 长时间不变时给负奖励

**策略**：先跑通默认奖励，收敛慢再开启塑形。避免过早引入复杂奖励导致 reward hacking。

## 7. PPO 超参初值

| 超参 | 值 | 说明 |
|------|-----|------|
| learning_rate | 3e-4 | SB3 默认 |
| n_steps | 1024 | 每次更新收集步数；笔记本可降 512 |
| batch_size | 64 | minibatch 大小 |
| n_epochs | 4 | 每次更新的 epoch 数 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE lambda |
| clip_range | 0.2 | PPO 策略剪辑范围 |
| ent_coef | 0.01 | 熵系数，鼓励探索 |
| vf_coef | 0.5 | 值函数系数 |
| max_grad_norm | 0.5 | 梯度裁剪 |
| total_timesteps | 1,000,000 | 先 100 万步看效果 |
| seed | 42 | 随机种子 |

## 8. 版本矩阵

```
# 核心
python==3.10
torch==2.11.0+cu128          # RTX 5060 (sm_120) 必需 cu128+
torchvision==0.26.0+cu128
gym==0.21.0
gym-super-mario-bros==7.4.0
nes-py==8.2.1
pyglet==1.5.21
stable-baselines3==1.8.0
numpy==1.26.4                 # 必须 <2.0
opencv-python==4.10.0.84     # 5.0 要求 numpy>=2
opencv-python-headless==4.10.0.84

# 工具链（gym 0.21 必需）
pip==23.3.2
setuptools==65.7.0
wheel==0.38.4
```

完整列表见 `requirements.txt`。

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 版本不兼容（gym/gymnasium/numpy） | 环境无法安装或运行 | 走路线 A、锁死版本、numpy<2.0、pip<24.1 |
| RTX 5060 (Blackwell) 不被 PyTorch 识别 | 只能用 CPU，训练极慢 | 安装 torch cu128+（2.8.0+），已验证 2.11.0+cu128 可用 |
| 训练不收敛 / reward 不上涨 | 看不到学习效果 | 奖励塑形、缩小动作空间、先小规模验证再放长训 |
| 笔记本算力不足 | 训练时间过长 | 降分辨率（84×84）、2-3 层 CNN、FrameSkip、调低 n_steps |
| 训练中断丢进度 | 长时间训练白费 | 后台训练 + CheckpointCallback 定期保存，可续训 |
| 无头环境无法渲染 | 无法看游戏画面 | opencv-python-headless + 关闭 render，用录屏替代 |

## 10. 团队分工建议

| 角色 | 职责 | 对应阶段 |
|------|------|----------|
| 项目负责人/架构 | 需求、方案、进度把控、版本矩阵决策 | 1、3 |
| 环境与预处理工程师 | 环境搭建、Wrapper 管线、状态预处理、奖励塑形 | 2、3 |
| 算法与训练工程师 | PPO 实现与调参、CNN 设计、长时间训练 | 4、5 |
| 测试与评估 | 单元测试、训练曲线分析、通关进度评估 | 4、5 |
| 文档与部署 | README、训练报告、推理 demo、打包 | 6 |

> 3-5 人团队可合并角色，如环境与算法由同一人负责。

## 11. 参考资料

- gym-super-mario-bros: https://github.com/Kautenja/gym-super-mario-bros
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- PyTorch 马里奥教程: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
- PyTorch Blackwell (sm_120) 支持: https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099
