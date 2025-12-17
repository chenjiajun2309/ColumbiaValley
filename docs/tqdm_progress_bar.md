# 进度条功能说明

## 功能

已添加tqdm进度条，显示simulation和RL训练的进度信息。

## 显示内容

### 主进度条
- **当前步骤**：Step X/Y
- **模拟时间**：当前游戏内时间
- **每步耗时**：step_time
- **Agent处理时间**：agent_time
- **RL训练时间**：rl_train（仅在训练时显示）
- **预计剩余时间**：ETA（分钟）

### RL训练信息
当RL训练开始时，会显示：
- 🔄 训练开始提示
- 收集的transition数量
- ✅ 训练完成时间
- 💾 模型保存信息（定期保存时）

### 完成总结
Simulation完成后会显示：
- 总步数
- 总耗时（分钟和秒）
- 平均每步耗时
- RL训练状态

## 安装

```bash
pip install tqdm
```

## 示例输出

```
Simulation: 100%|████████████████| 20/20 [02:30<00:00, 7.50s/step, step_time=7.45s, agent_time=6.20s, rl_train=1.25s, ETA=0.0m]

🔄 RL Training at step 5: 150 transitions
✅ Training completed in 1.25s

============================================================
Simulation completed!
Total steps: 20
Total time: 2.50 minutes (150.00 seconds)
Average time per step: 7.50 seconds
RL training: Enabled (interval: 5 steps)
============================================================
```

## 兼容性

如果没有安装tqdm，代码会自动使用fallback模式，不会显示进度条但功能正常。

## 自定义

可以通过修改`start.py`中的tqdm参数来自定义显示格式：
- `bar_format`: 进度条格式
- `unit`: 单位（默认"step"）
- `desc`: 描述文字

