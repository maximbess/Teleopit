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
状态历史和梯子几何分别配置 Conv1d 编码器。24D 梯子指令生成 117D 当前 Actor 组和 120D 无噪声
Critic 组；两者还分别接收 10 帧历史，以及一个 `9 x 15` 躯干坐标系横档 token 张量。只有 Critic
会接收一个独立的当前帧 15D 特权向量，包括归一化剩余回合时间、全身高度、回合高度记录及其差值、循环/阶段上升量、
躯干速度、躯干方向与支撑偏移误差、关节速度 RMS、两个物理脚部支撑、阶段所需支撑是否有效、释放
ramp 进度以及归一化阶段停留进度。该向量不会进入 `critic_history`，从而避免不必要地扩大时序编码器。
每个 token
包含有限横档端点、有效位，以及左右手脚各自的目标和支撑标记。24D 指令中的手脚目标向量也使用
相同的躯干坐标系。原先表示重置后初始化状态的标量现在表示活动手的连续抓握强度，对应的已握持
横档标记也会按同一数值逐渐减弱。任务塑形只使用一个按阶段调节的势函数。它将归一化的全身高度
（骨盆与躯干质心高度的平均值）和活动肢体到目标的归一化距离组合起来。只有当前阶段要求的支撑和
躯干姿态有效时才会支付正向进度；负向进度始终保留。橡胶手的模拟
焊接约束属于环境机制；脚部使用普通物理接触，策略动作仍然是 G1 的 29 个关节目标。

梯子专用 MuJoCo 关节配置会把全部肩、肘和腕部执行器的力矩上限缩放到标准 G1 数值的 70%：
25 Nm 的手臂组限制为 17.5 Nm，5 Nm 的腕部 pitch/yaw 组限制为 3.5 Nm。该力矩饱和只由梯子
机器人构建器应用；跟踪与推理仍使用标准 G1 上限。

指令使用显式的五状态顺序，并在每一层横档上重复：

1. 稳定双脚和两只已固定的手；
2. 移动第一只手；
3. 移动第二只手；
4. 在双手固定时移动第一只脚；
5. 移动第二只脚，然后返回稳定阶段。

每个手部移动阶段都会先进入内部 `PRE_RELEASE` 子阶段；它不会增加第六个 one-hot 状态。两个 weld
会在连续 8 个稳定的预载转移步中保持启用。随后，所选环境的 weld 会用 20 步 smoothstep 从编译时
的 equality 参数逐渐软化到 `solref.timeconst=0.18` 和 `solimp dmin=dmax=0.05`。只有当双脚和另一只
手继续支撑机器人，并且躯干速度、躯干姿态误差和相对支撑的质心误差分别保持在
`0.12 m/s`、`0.25 rad` 和 `0.12 m` 以内时，ramp 才会前进。任一条件失效都会使 ramp
回退 2 步。关节运动仍会受到惩罚，但不能阻止释放。只有在最小抓握强度下再连续稳定 5 帧，才会执行
二值 detach。若在 `PRE_RELEASE` 中停留 300 个策略步，回合会以失败终止。它与其他所有不成功的终止一样，
都会收到 `-50` 的终止奖励脉冲，因此故意跌倒无法规避更大的停滞专用代价。MJLab 会按环境展开
`eq_solref` 和 `eq_solimp`，因此并行环境可以独立完成释放。

回合诊断会分别记录手部连接、双脚支撑、躯干速度、躯干姿态和支撑偏移 gate，同时记录它们的组合
有效性，以及归一化的预载、软化 ramp 和最终停留进度。因此，释放停滞可以归因到具体条件，而不再
只有终止时的 `pre_release_stalled` 结果。

阶段有效性 gate 所使用的支撑中心也是连续的：手部位置按抓握强度和 attached 状态加权，脚部位置按
物理支撑状态加权。因此，在 ramp 过程中，平衡目标会从四点支撑几何连续移动到未来的三点支撑几何，
并且不会跟随已经自由伸出的手。

策略会在现有 24D 指令中以五维 one-hot 看到该状态。手部目标需要连续 3 帧保持接近且低速；脚部目标
需要连续 5 帧同时满足距离、低速、真实接触传感器命中和有效的全身高度。第一只脚相对阶段开始最多
允许降低 0.03 m；第二只脚必须相对循环开始将全身提升至少 0.12 m。稳定阶段会为每个环境随机要求连续 50--100
个有效帧（50 Hz 下为 1--2 秒）；一旦手部固定、脚部支撑、躯干速度或关节速度条件失效，保持计数就会
重新开始。每次成功的阶段切换都会存入容量受限的分阶段 GPU 状态库。阶段解锁后，50% 的训练重置会
从可用阶段边界中均匀采样，其余重置仍使用确定性的攀爬关键帧。恢复状态包括根部和关节的位置与速度、
手部锚点姿态、焊接状态，以及配置目标阶段之前的手脚横档分配。从边界开始的回合仍用于 PPO 训练，
但不会进入课程晋级窗口。自适应课程只记录符合条件的完整前缀终止回合：只有到达当前
已解锁阶段之后的边界才算成功；跌倒、根部过低终止以及普通回合超时都算失败。只有同时满足以下两个
条件，才会打开下一阶段：

- 滚动窗口已包含 100 个回合，且成功率严格高于 80%（至少 81 次成功）；
- 当前阶段已经累计达到其配置的最小环境步数。

使用 `num_steps_per_env=24` 时，初始稳定阶段使用 36,000 个环境步（1,500 次 PPO 迭代），之后每次
阶段切换使用 120,000 个环境步（5,000 次迭代）。如果每个阶段在首次具备资格时都立即通过，则最早
时间表如下：

| 可用阶段 | 最早环境步数 | 最早 PPO 迭代 |
|---|---:|---:|
| 稳定 | 0 | 0 |
| 第一只手 | 36,000 | 1,500 |
| 第二只手 | 156,000 | 6,500 |
| 第一只脚 | 276,000 | 11,500 |
| 第二只脚/完整循环 | 396,000 | 16,500 |

如果成功率不高于 80%，无论已经经过多少步，该阶段都会继续保持锁定。到达锁定边界时会截断当前短
前缀回合，使 PPO 能立即开始下一次尝试，而不必等待普通超时。整个训练过程使用固定的奖励集合：

| 奖励项 | 权重 |
|---|---:|
| 有支撑时的全身高度新历史最大值 | 20 |
| 感知阶段的有符号脚部放置 | 8 |
| 按阶段调节的目标进度 | 8 |
| 任一有序阶段完成 | 25 |
| 稳定阶段躯干朝向平方误差 | -1 |
| 不成功的回合终止 | -50 |
| 最终成功 | 100 |
| 稳定阶段以外的存活奖励 | 3 |
| 缺失必要足部支撑（每只脚） | -2 |
| 不依赖接触的支撑脚恢复进度 | 4 |
| 稳定阈值违反量 | -1 |
| 稳定阶段归一化初始姿态偏差 | -2 |
| 稳定阶段关节速度平方均值 | -0.2 |
| 所有阶段超过 1 N 的非手足梯子接触身体数 | -2 |
| 动作变化率 | -0.1 |
| 关节限位 | -10 |
| 超过 1 N 的自接触槽位 | -0.1 |
| 踝关节加速度平方和 | -2.5e-6 |

进度与脚部放置分别为五个阶段注册 `ladder_<phase>_progress` 和
`ladder_<phase>_foot_placement`。每项权重为 8，在其他阶段输出零；它们的总和等于原有未拆分的塑形奖励。
所有历史状态均在掩码之前更新。高度与事件奖励共享，朝向项仅用于稳定阶段。
共享正则项复用跟踪任务的自接触传感器和踝关节选择，不再额外惩罚全部关节的加速度。

每个控制步的奖励为
`dt * (20 H + sum_p I_p (8 P_p + 8 F_p) + 4 R - 2 M - I_stabilize (O + V + 2 E_q + 0.2 E_v) - 2 N_bad + 3 (1 - I_stabilize) - 0.1 A - 10 L - 0.1 C - 2.5e-6 Q) + 25 B - 50 D + 100 S`。

稳定阶段新增姿态代价 `E_q = sum_j w_j ((q_j-q_start_j)/0.25)^2 / sum_j w_j`：
参考始终为固定的初始梯上关键帧，未指定关节为零，不随状态库重置而改变。
髋、膝、腰关节权重为 2，踝关节为 1，手臂为 0.5。
稳定阶段还增加无死区速度代价 `E_v = mean_j (joint_vel_j / 1 rad/s)^2`。
所有阶段的 `N_bad` 统计当前梯子接触力超过 1 N 的身体数量，排除腕 yaw 手部和踝 roll 足部。
独立传感器覆盖两侧梯子的立柱与横档，每个身体仅保留一个最大力槽位，不使用历史接触。
凸分解不会增加身体计数；模型没有隐形面阻挡器。高度奖励和阶段切换条件保持不变。
奖励管理器仅对这三项乘一次 `dt`。
H 为有支撑时的新高度变化率，P 和 F 为原有截断后的进度与脚部放置变化率，O 为躯干朝向平方误差，
A 为动作变化平方和，L 为软关节限位超出量，C 为超过 1 N 的自接触槽位数，Q 为踝关节加速度平方和。
B、D、S 分别表示阶段完成、失败终止和最终成功事件。事件项内部已经除以 dt，因此其奖励为脉冲。


M 为每个有效步骤缺失的必要物理足部支撑数。R 为必要支撑脚到其保持横档的距离总和的有符号减少量，
除以 `2 * 0.35 m * dt`，不乘以接触标志。移脚阶段的两个项均排除当前移动脚。
恢复进度的历史在重置、阶段或保持横档变化时重新建立；静止为零，远离为负。
V 为躯干线速度、关节速度 RMS、躯干/骨盆最大角速度、腰部关节速度及相对支撑的躯干偏移这五项
超过阈值的正值经过归一化后的平方均值。各项以现有稳定阈值归一化（阈值为零时使用单位尺度），
满足阈值时惩罚为零。该稠密惩罚在所有条件通过之前提供信号，不改变保持时长和阶段转换条件。
稳定阶段的存活奖励为零。

`record_ladder_video.py --ladder_phase first_hand` 允许稳定阶段进入第一只手阶段，
仅在该手阶段成功后冻结转换。奖励修改影响后续训练，不会改变已有检查点在回放时输出的动作。

终止失败项适用于普通超时、`PRE_RELEASE` 停滞、跌倒和根部过低终止。最终成功以及到达已完成但仍锁定的
课程边界时的有意截断不计入该项。

新高度奖励会记录当前回合中骨盆与躯干 COM 平均高度的历史最大值。该记录始终会更新，但只有当前
阶段要求的物理支撑存在时才会支付正向增量。因此，无支撑的向上弹跳会消耗新的高度记录却得不到
奖励，之后回到相同高度也不能再次收集奖励。该项返回 `delta_height / step_dt`；经过奖励管理器的
时间步缩放后，整回合的最大贡献为 `20 * (max_height - initial_height)`，且不依赖控制频率。

脚部放置项同样是有符号势函数差，而不是保持原地不动即可持续获得的稠密奖励。在稳定阶段和手部阶段，
两只脚都会相对于各自当前分配的横档进行评估。在脚部阶段，支撑脚仍对应其原有横档，而选中的脚则
相对于新的目标横档进行评估。放置质量由实际横档接触和标准差为 0.06 m 的指数距离分数组成。建立
目标接触时支付正向势能变化，失去接触时应用大小相等的负向变化，而接触状态不变时奖励为零。因此，
一次完整的脱离/重新接触循环净奖励为零，无法通过接触抖动刷取奖励；无支撑的身体弹跳也会损失脚部
放置势能，并且得不到高度奖励。

按阶段调节的势函数包含到活动手或脚目标的归一化距离；在手部阶段还包含系数为 `1.0` 的连续释放进度
`1 - grip_strength`。全身高度仅由新高度奖励处理。抓握 ramp 的反向变化会产生等量的负向进度，因此
一次完整的软化/重新抓紧循环净奖励为零。
`STABILIZE` 仍在同一项中使用归一化停留进度。通常，当阶段支撑或姿态约束无效时，正向目标进度保留 25%
强度；约束有效时使用完整强度。一旦所选手已经脱离，除非这些约束有效，否则正向手部目标进度降为零，
从而避免不稳定地俯冲向横档也能获得进度。负向目标
进度不会衰减。移动阶段只有在所需支撑和相对支撑的质心偏移条件继续成立，并且躯干速度不超过
0.20 m/s、关节速度 RMS 不超过 1.0 rad/s 时才能完成。躯干姿态仍用于奖励塑形和诊断，但不再阻止
阶段切换。阶段完成和最终成功奖励仍是与时间步长无关的
单步脉冲。梯子 runner 会合并所有
GPU 的结果窗口，并在检查点中保存已解锁阶段、阶段起始步、最近结果以及阶段边界状态库。恢复缺少
自适应课程状态的旧固定时间表检查点时会明确失败。回放和评测会立即解锁全部阶段，并使用相同的固定奖励定义。

长期梯子 PPO 预设使用 60,000 次迭代，每 1,000 次迭代保存一次；高斯探索的初始标准差为 0.7，并将
有效值限制在 `[0.25, 1.0]`，学习率为 `5e-4`，熵系数为 `0.005`。梯子 runner 还会在每次更新和加载
检查点后把原始 scalar/log std 参数投影回边界，防止它停留在 clamp 下方而梯度为零。打开新的课程阶段时，
std 会恢复到 `0.7`，其优化器动量会被清除，自适应学习率状态和优化器参数组也会恢复到 `5e-4`；
`Policy/raw_mean_std` 会把投影后的原始参数与策略的有效 std 分开记录。梯子训练不会随机化初始回合长度，
因为人为制造的提前超时会污染最近 100
回合的课程成功窗口。

梯子几何形状和抓握点会在标准 `assets/robots/unitree_g1/g1_29dof.xml` 之上动态生成。
每一侧梯面都使用固定且可碰撞的侧轨，以及相互独立、顶部平坦的箱形横档，并采用刚性接触设置。机器人与这些杆件发生碰撞，平坦的横档顶部支撑双脚。横档间隙在物理上保持开放，没有不可见的梯面阻挡器。下文介绍的梯子专用 G1 碰撞覆盖层描述实际机体表面。G1 的默认碰撞编辑器运行后，梯子碰撞配置会显式重新启用所有动态生成的侧轨和横档。
五阶段 FSM 每次只指令一个肢体移动。两个脚部阶段中双手都会保持固定，未移动的脚继续充当物理支撑。

每个回合都会直接从梯子上的攀爬姿势开始。双脚与第 2 根横档形成物理接触，双手初始固定在第 5 根
横档；稳定阶段要求双脚接触，同时躯干和关节速度足够低，之后才会释放第一只移动手。每次完成一轮
手脚循环后都会重复相同的稳定检查。这样学习问题中不再包含从地面走向梯子的阶段。

梯子构建器会移除 XML 中内置的 `floor`，只保留一个由场景管理的地面。该地面使用纯色、无反射材质，
而不是默认的重复棋盘纹理。梯子播放还会使用一个受控光源，并关闭阴影和反射，以避免地面摩尔纹和
阴影贴图伪影。检查点保存在 `logs/rsl_rl/g1_ladder_rl/` 下，不使用 `save_onnx.py` 导出。

多组 TemporalCNN 协议与早期的 105D/108D 及扁平 117D/120D MLP 梯子检查点不兼容。基础当前帧维度
仍为 117D/120D，但模型现在还需要历史、`9 x 15` 目标感知梯子几何输入和 Critic 独有的当前 15D
reward/FSM 状态。缺少该 Critic 组的旧检查点不支持标准完整恢复；若要复用旧策略，需要显式地只加载
Actor，并重新初始化 Critic。使用早期 `9 x 7`
纯端点几何的 TemporalCNN 检查点也不兼容。连续抓握强度标量和反馈控制的软释放机制还改变了
24D 指令的语义与阶段切换分布。阶段 one-hot 语义、有序 FSM、攀爬关键帧、阶段势函数奖励和全身提升 gate、
已启用的杆件接触以及精细表面碰撞几何，因此都要求开始新的训练。
课程检查点版本也已提升，因此尝试恢复使用旧版按阶段调权多项奖励训练的策略时会立即失败。

### 梯子碰撞资产

训练或播放前，安装构建依赖并准备资产：

```bash
pip install -e '.[train,collision-build]'
python scripts/setup/download_assets.py --only robots g1_collision
```

`g1_collision` 显式下载
[OmniRetarget/Holosoma](https://github.com/amazon-far/holosoma/tree/bccd4d7451640a2800ddc77e469d911a84f91994/src/holosoma_retargeting/holosoma_retargeting/models/g1)
中的 G1 碰撞表面，固定到提交 `bccd4d7451640a2800ddc77e469d911a84f91994`。
它独立于 ModelScope/HuggingFace 的 `--source` 设置，也不包含在默认托管资产下载中。
CPU 构建使用 CoACD 1.0.14，每个来源链接最多生成 12 个凸部件，每个部件最多 64 个顶点。
这是近似表面模型，并非精确的三角形间碰撞。电机外壳和踝部铰链以实心凸包作为分解输入；
内部 CAD 孔洞不纳入接触模型。首次构建每个链接可能需要数分钟；完成的部件会被缓存。
来源版本、校验和、分解设置和上游许可声明保存在
`assets/robots/unitree_g1/omniretarget_collision/` 中。部件数量上限可能覆盖 CoACD 请求的凹度阈值；
该阈值不是误差保证。Git 忽略这些资产。
缺失或损坏的部件会阻止梯子环境启动，并提示重新构建。

覆盖层替换 25 个来源链接上的基本碰撞体，包括骨盆外壳、躯干、头、髋部、大腿、小腿、踝部、足部、
肩部 yaw 链接、肘部和腕部。固定附属链接的变换会合并到标准父链接中。
G1 rev. 1.0 使用不同的腰部组件：分解其自身的躯干表面，而非较旧的来源躯干。
四个连续的水平凸截段描述其外形，不再复制内部 CAD 空腔；
头部覆盖层在躯干坐标系中向上移动 10 mm，以匹配标准可视网格。躯干来源校验和会被记录和验证。
标准关节原点差异（waist roll +9 mm、waist pitch -19 mm、shoulder pitch +10 mm）保持不变。
保留标准手部胶囊和附着点，因为此任务使用自身的橡胶手抓握机制。
标准 G1 XML 仍定义全部 29 个关节、质量、惯量、执行器设置和可视网格。
动作跟踪与推理继续使用原碰撞模型。

侧轨和横档启用 MULTICCD 并使用零碰撞 margin：MuJoCo Warp 不支持 MULTICCD
箱体/网格碰撞对的非零 margin。它们保留指定的刚性接触参数，并使用高于机器人足部的接触优先级
（`condim=4`，侧轨滑动摩擦系数 1.4，横档 1.8）。因此，仅随机化机器人材质不会改变梯子接触材质。
更多碰撞部件会增加碰撞计算量；选择环境数量前，应在目标 GPU 上测量训练吞吐量。
此接触模型应重新训练策略；加载旧权重不会恢复旧碰撞物理，也不会使缓存的阶段边界状态自动有效。
课程检查点版本 4 拒绝从更早的奖励或接触模型完整恢复训练。旧 Actor 权重采用了不同的阶段归一化语义，不能将其在新模型中的播放结果视为等价复现。

梯子阶段指示量在 Actor、Critic、历史输入及导出策略中均不经过经验归一化，连续特征仍使用运行统计。
自碰撞传感器仅选择 G1 身体链接，排除同一实体中的梯子身体。
20 秒任务期限是终止失败，产生 -50 脉冲，PPO 不再对其进行超时 bootstrap。
Critic 的当前帧特权组包含归一化剩余时间；成功到达锁定课程边界仍属于截断，不产生失败惩罚。

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

同一入口也可以播放不需要动作数据集的纯 RL 梯子检查点：

```bash
python train_mimic/scripts/play.py \
    --task G1-Ladder-Climb-RL \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --ladder_phase second_hand
```

`--ladder_phase` 可选 `stabilize`、`first_hand`、`second_hand`、
`first_foot` 或 `second_foot`（默认值）。该参数选择有序 curriculum 前缀中
允许执行的最深阶段，而不是孤立的初始状态。例如，`second_hand` 会依次执行
`stabilize -> first_hand -> second_hand`，并在进入 `first_foot` 前重置；这样可
保持手部附着和脚部支撑在物理上连续一致。省略该参数或选择 `second_foot` 时，
会播放完整攀爬过程。

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
    --frames 1000 \
    --ladder_phase second_hand
```

录像脚本接受与交互式播放相同的 `--ladder_phase` 取值和有序前缀语义。显式
提供该参数时，FSM 会在所选阶段成功后冻结，不再切换阶段，也不会在 curriculum
边界终止。策略和物理仿真会继续运行，直到达到帧数上限或出现真实失败，因此
`stabilize` 可用于持续检查平衡能力。省略该参数时，会录制完整攀爬过程。

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
