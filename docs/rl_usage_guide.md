# RL在线训练使用指南

## 快速开始

### 1. 启用RL训练

运行simulation时添加`--use-rl`参数：

```bash
python start.py --name columbia1 --use-rl --step 100 --rl-train-interval 10
```

### 2. 参数说明

- `--use-rl`: 启用RL训练
- `--rl-train-interval`: RL训练间隔（每N步训练一次），默认10

### 3. 完整流程

```
python start.py
    ↓
SimulateServer初始化
    ↓
Game初始化 → 创建OnlineDataCollector (如果use_rl=True)
    ↓
SimulateServer初始化 → 创建MAPPOTrainer (如果use_rl=True)
    ↓
开始simulation循环
    ↓
每个step:
    1. 每个agent执行think()
        ↓
        Agent.think()开始
            → collector.start_collection() 提取状态
            → Agent做出决策（LLM）
            → collector.record_llm_action() 记录动作
            → Agent执行动作
            → collector.end_collection() 计算奖励，存储transition
        ↓
    2. 如果step % rl_train_interval == 0:
        → collector.flush_all_buffers() 获取所有数据
        → trainer.train_step(rollouts) 训练
        → 定期保存模型
    ↓
继续下一个step
```

## 代码集成说明

### 1. Agent.think()集成

在`Agent.think()`中添加了数据采集hooks：

```python
def think(self, status, agents):
    # 开始采集
    collector.start_collection(self.name, self, game)
    
    # ... 原有逻辑 ...
    
    # 记录动作
    collector.record_llm_action(...)
    
    # 结束采集
    collector.end_collection(self.name, self, game)
```

### 2. Agent._chat_with()集成

在`Agent._chat_with()`中添加了聊天交互记录：

```python
def _chat_with(self, other, focus):
    # ... 聊天逻辑 ...
    
    # 记录聊天交互
    collector.record_chat_interaction(...)
```

### 3. Game.__init__()集成

在`Game.__init__()`中初始化数据采集器：

```python
if config.get("use_rl", False):
    from modules.rl.data_collector import OnlineDataCollector
    self.rl_collector = OnlineDataCollector(...)
```

### 4. SimulateServer集成

在`SimulateServer`中添加了：
- RL Trainer初始化
- 训练循环逻辑
- 模型保存逻辑

## 配置选项

### 在config.json中配置

```json
{
    "use_rl": true,
    "rl_train_interval": 10,
    "rl": {
        "trainer": {
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "rollout_length": 2048,
            "num_epochs": 10
        },
        "collector": {
            "max_buffer_size": 1000
        }
    }
}
```

### 命令行参数

```bash
# 基本使用
python start.py --name test --use-rl

# 自定义训练间隔
python start.py --name test --use-rl --rl-train-interval 5

# 完整参数
python start.py \
    --name columbia1 \
    --use-rl \
    --step 100 \
    --stride 15 \
    --rl-train-interval 10 \
    --verbose info
```

## 数据流程

### 1. 数据采集（每个step）

```
Agent.think() 调用
    ↓
collector.start_collection()
    → 提取当前状态（state）
    ↓
Agent决策
    ↓
collector.record_llm_action()
    → 记录动作（action）
    ↓
Agent执行
    ↓
collector.end_collection()
    → 提取下一状态（next_state）
    → 计算奖励（reward）
    → 存储transition: (state, action, reward, next_state)
```

### 2. 训练（每N步）

```
step % rl_train_interval == 0
    ↓
collector.flush_all_buffers()
    → 获取所有agent的rollouts
    ↓
trainer.train_step(rollouts)
    → 计算advantages
    → 更新策略网络
    → 更新价值网络
    ↓
定期保存模型（每10次训练）
```

## 输出文件

### 1. 模型文件

保存在：`results/checkpoints/{name}/rl_models/step_{step}/`

- `policy_net.pt` - 策略网络（如果使用共享网络）
- `value_net.pt` - 价值网络（如果使用共享网络）
- 或每个agent的单独模型文件

### 2. 日志

RL训练信息会输出到日志中：
- 数据采集状态
- 训练进度
- 模型保存信息

## 注意事项

### 1. 依赖

确保安装了必要的依赖：
```bash
pip install torch numpy
```

### 2. 性能

- RL训练会增加计算开销
- 建议在GPU上运行（如果可用）
- 可以通过`rl_train_interval`调整训练频率

### 3. 内存

- 数据采集器会缓存transitions
- `max_buffer_size`控制最大缓存大小
- 定期训练会清空缓存

### 4. 错误处理

- 如果RL初始化失败，会记录警告但继续运行
- 如果训练失败，会记录警告但继续simulation
- 不会因为RL错误中断整个simulation

## 调试

### 1. 检查RL是否启用

查看日志中是否有：
```
RL Data Collector initialized
RL Trainer initialized
```

### 2. 检查数据采集

查看日志中是否有：
```
RL Training at step X: Y transitions collected
```

### 3. 检查模型保存

查看是否有模型文件生成：
```
results/checkpoints/{name}/rl_models/step_{step}/
```

## 示例

### 示例1：基本使用

```bash
# 运行100步，每10步训练一次
python start.py --name test-rl --use-rl --step 100 --rl-train-interval 10
```

### 示例2：快速训练（更频繁）

```bash
# 每5步训练一次
python start.py --name test-rl-fast --use-rl --step 100 --rl-train-interval 5
```

### 示例3：仅采集数据，不训练

```bash
# 设置很大的训练间隔，实际上不训练
python start.py --name test-collect --use-rl --step 100 --rl-train-interval 1000
```

## 下一步

1. **调优超参数**：根据实际情况调整学习率、折扣因子等
2. **改进奖励函数**：根据具体需求调整奖励函数
3. **优化网络架构**：根据状态空间调整网络结构
4. **评估效果**：使用训练好的模型评估智能体行为

