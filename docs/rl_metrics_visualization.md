# RL Metrics 记录和可视化

## 功能概述

已实现完整的RL metrics记录和可视化系统，用于跟踪和证明强化学习策略对agents行为的影响。

## 功能特性

### 1. Metrics记录

- **Reward历史**：记录每个agent的每次reward
- **Reward组件**：记录各个reward组件的值（persona_alignment, interaction_quality等）
- **训练统计**：记录每个训练步骤的统计信息
- **Action分布**：记录每个agent采取的各种action的分布
- **Episode奖励**：记录每个episode的总奖励

### 2. 可视化图表

在simulation结束时自动生成：

1. **reward_trends.png** - Reward趋势图
   - 每个agent的reward随时间变化
   - 包含移动平均线

2. **reward_components.png** - Reward组件图
   - 显示各个reward组件的贡献
   - 帮助理解reward的来源

3. **reward_distribution.png** - Reward分布图
   - 显示reward的分布情况
   - 包含均值线

4. **action_distribution.png** - Action分布图
   - 显示每个agent采取的各种action的频率
   - 帮助理解行为模式

5. **rl_summary.txt** - 文本摘要
   - 包含所有统计信息
   - 便于快速查看

## 使用方法

### 运行Simulation

```bash
python start.py \
    --name test-rl \
    --use_rl \
    --step 20 \
    --rl_train_interval 5 \
    --verbose info \
    --stride 15
```

### 查看结果

Simulation结束后，在checkpoints文件夹中：

```
results/checkpoints/test-rl/
├── rl_metrics.json          # 原始metrics数据
└── rl_visualizations/       # 可视化图表
    ├── reward_trends.png
    ├── reward_components.png
    ├── reward_distribution.png
    ├── action_distribution.png
    └── rl_summary.txt
```

## Metrics数据结构

### Reward History

```json
{
  "reward_history": {
    "Ava_Lee": [
      {
        "step": 5,
        "reward": 0.25,
        "timestamp": 1234567890.123,
        "components": {
          "persona_alignment": 0.0,
          "interaction_quality": 0.3,
          "relationship_growth": 0.1,
          "diversity": 0.0,
          "schedule_completion": 0.2,
          "total": 0.25
        }
      }
    ]
  }
}
```

### Training Stats

```json
{
  "training_stats": {
    "5": {
      "Ava_Lee": {
        "current_reward": 0.25,
        "mean_reward": 0.23,
        "std_reward": 0.05,
        "min_reward": 0.15,
        "max_reward": 0.35,
        "total_rewards": 10,
        "components": {...}
      }
    }
  }
}
```

## Reward组件说明

### 1. Persona Alignment (人设一致性)
- 评估action是否符合agent的人设
- 权重：0.3

### 2. Interaction Quality (交互质量)
- 评估交互的质量（如聊天长度、相关性）
- 权重：0.3

### 3. Relationship Growth (关系发展)
- 评估关系分数的增长
- 权重：0.2

### 4. Diversity (多样性)
- 鼓励行为多样性，避免重复
- 权重：0.1

### 5. Schedule Completion (计划完成)
- 评估是否按计划执行
- 权重：0.1

## 证明RL有效性的指标

### 1. Reward趋势
- **上升趋势**：说明RL策略在改进
- **稳定性**：说明策略收敛

### 2. Reward组件分析
- **Persona Alignment提升**：说明行为更符合人设
- **Interaction Quality提升**：说明交互质量改善
- **Relationship Growth提升**：说明关系发展更好

### 3. Action分布
- **多样性增加**：说明行为更丰富
- **特定action增加**：说明策略学习到偏好

### 4. 统计对比
- **Mean Reward提升**：总体表现改善
- **Std Reward降低**：行为更稳定

## 示例分析

### 场景1：RL策略有效

```
Agent: Ava_Lee
  Mean Reward: 0.15 → 0.25 (提升67%)
  Persona Alignment: 0.0 → 0.2 (提升)
  Interaction Quality: 0.1 → 0.3 (提升200%)
```

**结论**：RL策略成功改善了agent的行为

### 场景2：需要调整

```
Agent: Benjamin_Carter
  Mean Reward: 0.20 → 0.18 (下降)
  Diversity: -0.3 (重复行为)
```

**结论**：需要调整reward权重或策略

## 自定义配置

在`config.json`中可以配置：

```json
{
  "rl": {
    "metrics": {
      "checkpoints_folder": "results/checkpoints/test-rl"
    },
    "reward_function": {
      "persona_alignment_weight": 0.3,
      "interaction_quality_weight": 0.3,
      "relationship_growth_weight": 0.2,
      "diversity_weight": 0.1,
      "schedule_completion_weight": 0.1
    }
  }
}
```

## 注意事项

1. **数据量**：每个step都会记录，数据量可能较大
2. **存储空间**：确保有足够的存储空间
3. **可视化时间**：生成图表可能需要几秒钟
4. **依赖**：需要matplotlib库（已包含在requirements.txt中）

## 故障排除

### 问题：可视化生成失败

**原因**：可能缺少matplotlib或后端问题

**解决**：
```bash
pip install matplotlib
```

### 问题：没有metrics数据

**原因**：RL未启用或collector未初始化

**解决**：确保`--use_rl`参数已设置

### 问题：图表为空

**原因**：没有足够的训练数据

**解决**：运行更多steps或降低`rl_train_interval`

## 总结

这个系统提供了完整的RL metrics跟踪和可视化，可以帮助：

1. **证明RL有效性**：通过reward趋势和组件分析
2. **理解行为变化**：通过action分布和统计
3. **优化策略**：通过识别问题和调整权重
4. **展示结果**：通过图表和摘要报告

