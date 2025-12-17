# MAPPO实现指南

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy
# 可选：使用Stable-Baselines3
pip install stable-baselines3[extra]
```

### 2. 集成到现有Agent类

在`Agent`类中添加RL支持：

```python
# 在 agent.py 的 __init__ 中添加
if config.get("use_rl", False):
    from modules.rl.state_extractor import StateExtractor
    from modules.rl.action_space import ActionSpace
    self.state_extractor = StateExtractor()
    self.action_space = ActionSpace()
    self.use_rl = True
    self.rl_action = None
    self.rl_action_history = []
else:
    self.use_rl = False
```

### 3. 修改决策逻辑

在`Agent.think()`或`Agent._reaction()`中集成RL决策：

```python
def _rl_decide_action(self, agents):
    """使用RL策略决定动作"""
    if not self.use_rl or not hasattr(self, 'rl_policy'):
        return None
    
    # 提取状态
    state = self.state_extractor.extract(self, get_game())
    state_vec = self.state_extractor.state_to_vector(state)
    
    # 获取动作
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        action_dist = self.rl_policy(state_tensor)
        action_id = action_dist.sample().item()
    
    # 解码动作
    action_dict = self.action_space.decode_action(action_id, self, get_game())
    return action_dict
```

### 4. 训练流程示例

```python
from modules.rl.mappo import MAPPOTrainer

# 初始化训练器
trainer = MAPPOTrainer(
    agents=game.agents,
    game=game,
    config={
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "rollout_length": 2048,
        "num_epochs": 10,
    }
)

# 训练循环
for episode in range(num_episodes):
    # 收集经验
    rollouts = trainer.collect_rollout(num_steps=2048)
    
    # 训练
    trainer.train_step(rollouts)
    
    # 定期保存
    if episode % 100 == 0:
        trainer.save_models(f"checkpoints/episode_{episode}")
```

## 关键集成点

### 1. 状态提取时机
- 在`Agent.think()`开始时提取状态
- 在动作执行后提取下一状态

### 2. 动作执行
- 在`Agent._reaction()`中检查RL动作
- 在`Agent._chat_with()`中使用RL指导对话策略
- 在`Agent._determine_action()`中使用RL选择活动

### 3. 奖励计算
- 在每个决策步骤后计算奖励
- 在对话结束后计算交互奖励
- 在关系变化时计算关系奖励

## 注意事项

1. **状态一致性**：确保状态提取在训练和推理时一致
2. **动作有效性**：在执行RL动作前检查有效性
3. **奖励设计**：仔细设计奖励函数，避免奖励hacking
4. **训练稳定性**：使用梯度裁剪、学习率调度等技巧
5. **计算资源**：多智能体训练需要大量计算，考虑使用GPU

## 进阶优化

1. **使用预训练模型**：从LLM决策中学习初始策略
2. **课程学习**：从简单场景开始，逐步增加复杂度
3. **对手建模**：让智能体学习预测其他智能体的行为
4. **分层强化学习**：将决策分为高层策略和底层执行

