# Concepts 说明

## 什么是 Concepts？

**Concepts** 是agent在`percept()`方法中感知到的周围环境中的**事件和交互**。

### Concepts的来源

1. **周围tile上的events**：其他agents的活动、物体状态等
2. **聊天交互**：其他agents的对话
3. **环境事件**：物体被使用、位置变化等

### Concepts的作用

- 用于agent的**决策**（是否反应、是否聊天等）
- 存储在**记忆系统**中（associate memory）
- 影响**poignancy**（重要性分数），触发反思

## 日志格式说明

```
[INFO]: Sophia_Rossi percept 4/3 concepts
```

### 数字含义

- **第一个数字（4）**：`valid_num` - **有效的、非idle的concepts数量**
  - 这些concepts会被添加到记忆系统
  - 会增加poignancy分数
  
- **第二个数字（3）**：`len(self.concepts)` - **最终保留的concepts总数**
  - 包括所有concepts（包括idle）
  - 但**排除了agent自己的events**

### 为什么会出现不一致？

#### 情况1：4/3（第一个数字 > 第二个数字）

```
valid_num = 4  (有效的concepts)
total concepts = 3  (最终保留的)
```

**原因**：
1. 有些concepts的subject是agent自己，被过滤掉了（第337行）
2. 或者有些idle concepts被添加但最终被过滤

#### 情况2：2/3（第一个数字 < 第二个数字）

```
valid_num = 2  (有效的concepts)
total concepts = 3  (最终保留的)
```

**原因**：
1. 有些concepts是"idle"类型，不计入valid_num，但被添加到concepts列表
2. 最终保留的concepts包括idle concepts

## 代码逻辑

```python
def percept(self):
    # 1. 收集周围的事件
    events = ...  # 从周围tile收集
    
    # 2. 处理每个事件
    self.concepts, valid_num = [], 0
    for event in events:
        if event.object == "idle":
            # Idle events不计入valid_num，但添加到concepts
            node = Concept.from_event("idle_...", "event", event, poignancy=1)
            self.concepts.append(node)
        else:
            # 非idle events计入valid_num
            valid_num += 1
            node = self._add_concept(node_type, event)
            self.status["poignancy"] += node.poignancy
            self.concepts.append(node)
    
    # 3. 过滤掉自己的events
    self.concepts = [c for c in self.concepts if c.event.subject != self.name]
    
    # 4. 记录日志
    logger.info("{} percept {}/{} concepts".format(self.name, valid_num, len(self.concepts)))
```

## 这是问题吗？

**不是问题！** 这是正常的行为：

1. **valid_num** 统计的是**有意义的、非idle的concepts**
2. **len(self.concepts)** 是**最终保留的concepts**（可能包括idle，但排除了自己的）

### 正常情况示例

- `percept 3/3 concepts` - 3个有效concepts，全部保留
- `percept 4/3 concepts` - 4个有效concepts，但1个被过滤（可能是自己的）
- `percept 2/3 concepts` - 2个有效concepts，1个idle concept被添加

## Concepts的类型

### 1. Event Concepts
- 其他agents的活动
- 物体状态变化
- 环境事件

### 2. Chat Concepts
- 其他agents之间的对话
- 可以触发自己的聊天反应

### 3. Idle Concepts
- "idle"状态的事件
- 重要性低（poignancy=1）
- 不计入valid_num

## 配置参数

在`config.json`中：

```json
{
    "agent": {
        "percept": {
            "att_bandwidth": 8  // 最多处理8个events
        }
    }
}
```

- `att_bandwidth`：注意力带宽，限制每次percept处理的events数量

## 总结

- **Concepts** = agent感知到的周围环境中的事件
- **4/3** = 4个有效concepts，3个最终保留
- **这是正常行为**，不是bug
- Concepts用于agent的决策和记忆系统

