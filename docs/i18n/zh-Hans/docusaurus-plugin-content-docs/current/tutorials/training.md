---
sidebar_position: 5
---

# 训练

可以训练全身追踪策略，也可以训练独立的纯 RL 梯子策略。追踪策略可导出为 ONNX 格式用于推理部署。

:::info
数据准备请参阅 [数据集参考](../reference/dataset)。常见训练问题请参阅 [训练故障排查](../reference/training-troubleshooting)。
:::

## 环境安装

```bash
conda create -n teleopit python=3.10
conda activate teleopit
pip install -e '.[train]'
```

验证安装：
```bash
python -c "import train_mimic.tasks; print('training OK')"
```

动作数据集仅为 `General-Tracking-G1` 所必需。下载分发的最小数据集，并生成合并后的预计算训练数据集：

```bash
python scripts/setup/download_assets.py --only robots data
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets --outdir data/datasets_precomputed --jobs 8
```

纯 RL 梯子任务只需要 `robots` 资产组，不需要数据集。

## 训练

### 冒烟测试

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets_precomputed
```

### 完整训练

```bash
python train_mimic/scripts/train.py \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

### 多卡训练

```bash
python train_mimic/scripts/train.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

### 多机多卡训练

跨多台机器训练时，直接使用 `torchrun`：

```bash
torchrun \
    --nnodes=$PET_NNODES \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$PET_MASTER_ADDR \
    --master_port=$PET_MASTER_PORT \
    train_mimic/scripts/train.py \
    --num_envs 1024 \
    --max_iterations 1000 \
    --motion_file data/datasets_precomputed
```

**注意事项：**
- 多卡模式下 `--num_envs` 为每张 GPU 的环境数量
- 多机模式下 `--num_envs` 也按每个进程计算，因此总环境数会随 `world_size` 线性增长
- 默认日志工具为 TensorBoard。使用 `--logger wandb` 或 `--logger swanlab` 可选择 W&B 或 SwanLab；项目名默认使用 `experiment_name`
- `--motion_file` 接受预计算训练数据集根目录或单个预计算 `.h5` shard；shard 会递归发现
- 如果只有最小分发 shard，先运行 `python train_mimic/scripts/data/precompute_dataset.py <minimal_dataset> --outdir <precomputed_dataset>`，再把预计算输出传给训练。
- 训练会在启动时把所有发现的预计算 motion window 全量加载到内存中。
- `--max_iterations` 表示追加迭代次数；例如从 `model_12000.pt` 恢复训练并设置 `--max_iterations 18000`，最终将训练到 `model_30000.pt`

## 纯 RL 梯子训练

如果目标是直接通过强化学习训练 G1 攀爬 A 字梯，请使用专用的梯子训练入口：

```bash
python train_mimic/scripts/train_ladder.py \
    --num_envs 64 \
    --max_iterations 100
```

完整训练示例：

```bash
python train_mimic/scripts/train_ladder.py \
    --num_envs 4096 \
    --max_iterations 60000
```

梯子训练入口刻意不提供 `--motion_file`、采样模式或 rewind 选项。它使用 TemporalCNN 策略，并为
状态历史和梯子几何分别配置 Conv1d 编码器。24D 梯子指令生成 117D 当前 Actor 组和 120D 特权
Critic 组；两者还分别接收 10 帧历史，以及一个包含有限横档端点和有效位的 `9 x 7` 躯干坐标系张量。
手脚目标塑形使用有符号的接近进度：靠近目标为正，悬停为零，
远离目标为负。这样可以防止策略仅将脚保持在横档附近而不接触，却持续获得接近奖励。橡胶手的模拟
焊接约束属于环境机制；脚部使用普通物理接触，策略动作仍然是 G1 的 29 个关节目标。

指令使用显式的五状态顺序，并在每一层横档上重复：

1. 稳定双脚和两只已固定的手；
2. 移动第一只手；
3. 移动第二只手；
4. 在双手固定时移动第一只脚；
5. 移动第二只脚，然后返回稳定阶段。

策略会在现有 24D 指令中以五维 one-hot 看到该状态。手部目标需要连续 3 帧保持接近且低速；脚部目标
需要连续 5 帧同时满足距离、低速和真实接触传感器命中。自适应课程会记录每个终止回合：只有到达当前
已解锁阶段之后的边界才算成功；跌倒、根部过低终止以及普通回合超时都算失败。只有同时满足以下两个
条件，才会打开下一阶段：

- 滚动窗口已包含 100 个回合，且成功率严格高于 80%（至少 81 次成功）；
- 当前阶段已经累计至少 120,000 个环境步。

使用 `num_steps_per_env=24` 时，每次阶段切换的最小预算为 5,000 次 PPO 迭代。如果每个阶段在首次具备
资格时都立即通过，则最早时间表如下：

| 可用阶段 | 最早环境步数 | 最早 PPO 迭代 |
|---|---:|---:|
| 稳定 | 0 | 0 |
| 第一只手 | 120,000 | 5,000 |
| 第二只手 | 240,000 | 10,000 |
| 第一只脚 | 360,000 | 15,000 |
| 第二只脚/完整循环 | 480,000 | 20,000 |

如果成功率不高于 80%，无论已经经过多少步，该阶段都会继续保持锁定。到达锁定边界时会截断当前短
前缀回合，使 PPO 能立即开始下一次尝试，而不必等待普通超时。奖励权重按相应的长期尺度切换：

| 环境步数 | 手/脚进度 | 躯干上升/稳定 | 手部推进 | 脚部推进 | 稳定/循环/完成 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 / 8 | 12 / 10 | 25 | 50 | 30 / 80 / 100 |
| 240,000 | 6 / 12 | 16 / 8 | 35 | 70 | 25 / 120 / 180 |
| 480,000 | 8 / 16 | 20 / 6 | 45 | 90 | 20 / 160 / 260 |

躯干上升奖励只在双手固定的脚部阶段启用。密集的躯干速度奖励、更强的直立奖励、更强的动作变化率和
全局关节速度惩罚、关节加速度惩罚，以及额外的支撑关节速度惩罚，会稳定躯干和必须保持固定的肢体。
脚部接触与脚部切换的权重刻意高于手部对应项。切换和完成奖励仍是与时间步长无关的单步脉冲。梯子
runner 会合并所有 GPU 的结果窗口，并在检查点中保存已解锁阶段、阶段起始步和最近结果。恢复缺少
自适应课程状态的旧固定时间表检查点时会明确失败。回放和评测会立即解锁全部阶段，并使用最终阶段权重。

长期梯子 PPO 预设使用 60,000 次迭代，每 1,000 次迭代保存一次；高斯探索的初始标准差为 0.7，学习率为
`5e-4`，熵系数为 `0.005`。梯子训练不会随机化初始回合长度，因为人为制造的提前超时会污染最近 100
回合的课程成功窗口。

梯子几何形状和抓握点会在标准 `assets/robots/unitree_g1/g1_29dof.xml` 之上动态生成。
每一侧梯面都使用固定且可碰撞的侧轨，以及相互独立、顶部平坦的箱形横档，并采用刚性接触设置。机器人不能穿过这些杆件，平坦的横档顶部可以支撑双脚。每一侧梯面后方都会偏置放置一个薄的不可见阻挡器；它使用独立的碰撞掩码，只阻止骨盆、躯干和头部进入 A 型梯内部，同时允许手和脚接触外露杆件，因此横档间隙在视觉上仍保持为空。G1 的默认碰撞编辑器运行后，梯子碰撞配置会显式重新启用所有动态生成的侧轨和横档。
五阶段 FSM 每次只指令一个肢体移动。两个脚部阶段中双手都会保持固定，未移动的脚继续充当物理支撑。

每个回合都会直接从梯子上的攀爬姿势开始。双脚与第 2 根横档形成物理接触，双手初始固定在第 5 根
横档；稳定阶段要求双脚接触，同时躯干和关节速度足够低，之后才会释放第一只移动手。每次完成一轮
手脚循环后都会重复相同的稳定检查。这样学习问题中不再包含从地面走向梯子的阶段。

梯子构建器会移除 XML 中内置的 `floor`，只保留一个由场景管理的地面。该地面使用纯色、无反射材质，
而不是默认的重复棋盘纹理。梯子播放还会使用一个受控光源，并关闭阴影和反射，以避免地面摩尔纹和
阴影贴图伪影。检查点保存在 `logs/rsl_rl/g1_ladder_rl/` 下，不使用 `save_onnx.py` 导出。

多组 TemporalCNN 协议与早期的 105D/108D 及扁平 117D/120D MLP 梯子检查点不兼容。当前帧维度
仍为 117D/120D，但模型现在还需要历史和梯子几何输入。阶段 one-hot 语义、有序 FSM、攀爬关键帧、
按计划调整的奖励、已启用的杆件接触以及躯干阻挡碰撞几何，也要求开始新的训练。

多 GPU 启动使用相同的每 GPU 环境数约定：

```bash
python train_mimic/scripts/train_ladder.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 60000
```

:::warning 多 GPU 使用的 Warp 版本
当前支持的 `mjlab==1.4.0` / MuJoCo Warp 3.8 训练栈要求
`warp-lang==1.15.0`。Warp 1.16.0 会在传感器 kernel 编译阶段报错
`Referencing undefined symbol: xmat`；四张 GPU 同时冷启动时，这看起来像分布式
启动错误，但实际上发生在 PPO 或 NCCL 初始化之前。可用以下命令修复已有环境：

```bash
python -m pip install --force-reinstall "warp-lang==1.15.0"
python -c "import warp as wp; print(wp.__version__)"
```

训练入口现在会在创建 CUDA 环境之前检查该版本，并在版本不兼容时显示相同的修复命令。
:::

## 导出 ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output track.onnx \
    --history_length 10
```

导出的模型为双输入 ONNX（`obs` + `obs_history`）。推理端需要与当前 `velcmd_history` 观测匹配的 167D 双输入 ONNX 策略。

## 评估

### 播放验证

```bash
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed
```

### 定量评估

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed \
    --num_envs 1
```

### 带视频的定量评估

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed \
    --num_envs 1 \
    --video \
    --video_length 600
```

### 梯子定量评估

使用专用的纯 RL 评估脚本检查梯子策略。该过程不会继续执行 PPO 更新，也不需要动作数据集：

```bash
python train_mimic/scripts/benchmark_ladder.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --num_envs 64 \
    --num_eval_steps 5000
```

脚本会在 `benchmark_results/` 下写入文本和 JSON 报告，其中包含成功、失败、超时、横档进度、
抓握、到目标的距离、奖励和回合长度统计。超时会计为已完成但未成功的回合，因此默认的 20 秒
回合限制可以防止停滞策略从成功率统计中消失。

视频评估使用单个环境，视频保存在 `benchmark_results/videos/` 下：

```bash
python train_mimic/scripts/benchmark_ladder.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --num_envs 1 \
    --video \
    --video_length 1000
```

### 仅录制梯子策略视频

只需要 MP4 时，请使用专用录像脚本。它不会计算评估指标，也不会创建文本或 JSON 报告。录像固定使用
一个环境，并在首次回合终止或达到帧数上限时停止，因此自动重置不会出现在视频中：

```bash
python train_mimic/scripts/record_ladder_video.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --output ladder.mp4 \
    --frames 1000
```

录像脚本默认使用位于机器人一侧梯面外部的固定世界坐标概览相机（`azimuth=30`、`elevation=-5`、
`distance=4.5`），并对准攀爬面的中部。与跟踪躯干的相机相比，这种构图会让完整机器人和梯子始终
保持在画面内，并防止另一侧梯面遮挡策略动作。可以使用 `--camera_azimuth`、
`--camera_elevation`、`--camera_distance`、`--camera_lookat X Y Z` 和 `--camera_fovy` 覆盖默认构图。

## 训练架构

```text
train_mimic/scripts
    -> train_mimic/app.py
    -> tracking and RL-only ladder task configs
    -> mjlab + rsl_rl
```

关键文件：
- `train_mimic/app.py` - 训练/播放/评估的统一入口
- `train_mimic/tasks/tracking/config/env.py` - General-Tracking-G1 和动态生成梯子的环境构建器
- `train_mimic/tasks/tracking/config/rl.py` - TemporalCNN 追踪和梯子 PPO 配置
- `train_mimic/tasks/tracking/mdp/ladder.py` - 梯子指令、抓握状态、奖励和成功终止条件
- `train_mimic/tasks/tracking/mdp/commands.py` - 支持 `uniform`、`start` 和 `rewind` 采样模式。训练默认使用 `rewind`；播放/评估使用 `start`。
