# Super Mario Keyboard Play

这是一个基于 `gym_super_mario_bros` 的可操作版超级马里奥游戏。

## 1. 创建 Conda 环境

进入项目目录：

```powershell
cd C:\Users\16937\Desktop\super_mario
```

创建 Conda 环境：

```powershell
conda create -y -n mario-play python=3.10 pip
```

激活环境：

```powershell
conda activate mario-play
```

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

## 3. 安装当前项目

```powershell
pip install -e . --no-build-isolation
```

## 4. 启动游戏

```powershell
python -m mario_rl.play
```

启动后会弹出马里奥游戏窗口。

## 5. 操作方式

```text
方向键左 / A       左移
方向键右 / D       右移
Space / W / ↑      跳跃
S / ↓              加速，和右移、跳跃组合使用
Esc                退出窗口
```

## 常见问题

如果看到 Gym 过期警告，可以忽略。`gym_super_mario_bros` 依赖旧版 Gym，只要游戏窗口能弹出就不影响使用。

如果没有弹窗，确认你使用的是 Python 3.10 环境，并且已经执行过：

```powershell
conda activate mario-play
pip install -e . --no-build-isolation
```
