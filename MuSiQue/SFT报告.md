# Cognifold SFT 评测报告

**生成时间**: 2026-07-07 11:30:47
**样本数量**: 308
**基座模型**: Qwen3-4B
**LoRA 适配器**: saves/qwen3-4b/lora/sft-cognifold
**概念匹配阈值**: 0.6
**相似度模式**: sentence-transformers

---

## 综合评分

| 模型            | 综合得分    |
| --------------- | ----------- |
| 基座模型 (Base) | **0.5867**  |
| LoRA SFT 模型   | **0.7418**  |
| 提升幅度 (Δ)    | **+0.1550** |

---

## 四维度 F1 总览

| 维度 F1        | 基座模型 | LoRA SFT | 提升 (Δ)  |
| -------------- | -------- | -------- | --------- |
| **Event F1**   | 0.9888   | 0.9978   | ↑ +0.0090 |
| **Concept F1** | 0.1487   | 0.3588   | ↑ +0.2100 |
| **Edge F1**    | 0.4878   | 0.8072   | ↑ +0.3194 |
| **Update F1**  | 0.7543   | 0.9048   | ↑ +0.1505 |

> **说明**: Event F1 = 2 × Precision × Recall / (Precision + Recall)，其中 Precision 为事件内容质量（ID/标题/描述/类型加权），Recall 为事件创建率。
> Concept F1 = 创建决策 F1 × 平均内容相似度（综合衡量创建决策和内容质量）。
> Edge F1 = 基于 ID 映射的边连接 F1。
> Update F1 = 基于 node_id + 字段匹配的 Precision/Recall 调和平均。

---

## 详细指标对比

| 指标                       | 基座模型   | LoRA SFT   | 提升 (Δ)      |
| -------------------------- | ---------- | ---------- | ------------- |
| **JSON 格式**              |            |            |               |
| JSON 解析成功率            | 99.4%      | 100.0%     | ↑ +0.0065     |
| 顶层结构完整率             | 99.4%      | 100.0%     | ↑ +0.0065     |
| operations 是数组          | 99.4%      | 100.0%     | ↑ +0.0065     |
|                            |            |            |               |
| **EVENT 节点**             |            |            |               |
| Event Precision (内容质量) | 0.9841     | 0.9957     | ↑ +0.0115     |
| Event Recall (创建率)      | 0.9935     | 1.0000     | ↑ +0.0065     |
| **Event F1**               | **0.9888** | **0.9978** | **↑ +0.0090** |
| Event ID 精确匹配          | 99.0%      | 99.7%      | ↑ +0.0065     |
| Event Title 精确匹配       | 96.4%      | 99.7%      | ↑ +0.0325     |
| Event Title ROUGE-L        | 0.9870     | 0.9992     | ↑ +0.0122     |
| Event Description ROUGE-L  | 0.9822     | 0.9923     | ↑ +0.0101     |
|                            |            |            |               |
| **CONCEPT 节点**           |            |            |               |
| Concept Precision          | 17.5%      | 68.6%      | ↑ +0.5103     |
| Concept Recall             | 27.8%      | 33.3%      | ↑ +0.0556     |
| **Concept F1**             | **21.5%**  | **44.9%**  | **↑ +0.2335** |
| Concept 假阳性率           | 39.8%      | 4.7%       | ↓ -0.3517     |
| Concept 内容相似度         | 0.6916     | 0.7998     | ↑ +0.1081     |
| **Concept 综合 F1**        | **0.1487** | **0.3588** | **↑ +0.2100** |
|                            |            |            |               |
| **EDGE**                   |            |            |               |
| Edge 数量 MAE              | 0.9318     | 0.6234     | ↓ -0.3084     |
| Edge Type F1               | 80.5%      | 74.0%      | ↓ -0.0644     |
| **Edge F1 (ID映射)**       | **48.8%**  | **80.7%**  | **↑ +0.3194** |
|                            |            |            |               |
| **UPDATE**                 |            |            |               |
| Update 数量 MAE            | 0.3312     | 0.1753     | ↓ -0.1558     |
| Update Precision           | 70.2%      | 92.0%      | ↑ +0.2176     |
| Update Recall              | 81.5%      | 89.0%      | ↑ +0.0756     |
| **Update F1**              | **75.4%**  | **90.5%**  | **↑ +0.1505** |
| Update node_id 匹配率      | 50.0%      | 76.2%      | ↑ +0.2619     |

> **说明**:
> - 带 ↑ 的 Δ 表示提升（越高越好），↓ 表示降低（越低越好）
> - Concept 假阳性率、Edge/Update 数量 MAE 是越低越好的指标，Δ 前加 ↓ 表示降低（改善）

---

## Concept 创建混淆矩阵

### 基座模型 (Base)

|              | GT: 有 Concept | GT: 无 Concept |
| ------------ | -------------- | -------------- |
| **Pred: 有** | (见 metrics)   | -              |
| **Pred: 无** | -              | -              |

### LoRA SFT 模型

|              | GT: 有 Concept | GT: 无 Concept |
| ------------ | -------------- | -------------- |
| **Pred: 有** | (见 metrics)   | -              |
| **Pred: 无** | -              | -              |

> 详细的 TP/FP/FN/TN 数值请查看上方 Concept 指标中的 concept_tp/fp/fn/tn 字段。

---

## 典型错误分析

### JSON 解析失败
| 模型     | 解析失败数 | 失败率 |
| -------- | ---------- | ------ |
| 基座模型 | 2          | 0.6%   |
| LoRA SFT | 0          | 0.0%   |

### 缺少 EVENT 节点
| 模型     | 缺少数 | 比例 |
| -------- | ------ | ---- |
| 基座模型 | 0      | 0.0% |
| LoRA SFT | 0      | 0.0% |

---

## 典型样本对比

### 样本 47 (LoRA 修复了解析失败)

**Ground Truth (摘要)**:
```json
{
  "reasoning": "New event about the 2015 Rugby World Cup Final. It relates to an existing event about the Rugby World Cup (ecaee487-...), so we can create a concept for 'Rugby World Cup' to group these. No football-related concepts are applicable.",
  "operations": [
    {
      "op": "ADD_NODE",
      "node_type": "event",
      "data": {
        "event_id": "90ba706e-8e6f-4eea-9a70-37698b873003",
        "title": "2015 Rugby World Cup Final",
        "event_type": "information_paragraph",
  ...
```

**Base 预测 (摘要)**:
```
```json
{
  "reasoning": "The new event describes the 2015 Rugby World Cup Final, a significant international rugby match between New Zealand and Australia. This event introduces a new domain (rugby) not previously represented in the graph, which is distinct from the existing football-focused content. While there is no direct overlap with existing concepts, the event provides factual information a...
```

**LoRA 预测 (摘要)**:
```
{
  "reasoning": "The new event is a rugby World Cup final, unrelated to the existing football-focused concepts. No patterns emerge to warrant a new concept or intent. Only the event node is added.",
  "operations": [
    {
      "op": "ADD_NODE",
      "node_type": "event",
      "data": {
        "event_id": "90ba706e-8e6f-4eea-9a70-37698b873003",
        "title": "2015 Rugby World Cup Final",
 ...
```

---

### 样本 268 (LoRA 修复了解析失败)

**Ground Truth (摘要)**:
```json
{
  "reasoning": "The new event is a paragraph about a 2006 genetic study showing Ashkenazi Jews are a homogeneous subgroup regardless of origin, reinforcing the existing concept of Ashkenazi Genetic Origins. It also mentions Haredi endogamy, but this does not yet warrant a new concept. No intent is generated because no goal or actionable pattern is implied.",
  "operations": [
    {
      "op": "ADD_NODE",
      "node_type": "event",
      "data": {
        "event_id": "99fbfe23-bde2-415b-831b-...
```

**Base 预测 (摘要)**:
```
```json
{
  "reasoning": "The new event reinforces the genetic homogeneity of Ashkenazi Jews across diverse European origins and highlights the role of endogamy and Haredi intermarriage in preserving their genetic cohort. This aligns with and strengthens existing concepts like c-ashkenazi-genetic-origins and c-ashkenazi-jews. However, the event introduces a new nuance: the prevalence of Ashkenazi ...
```

**LoRA 预测 (摘要)**:
```
{
  "reasoning": "The event describes a 2006 study on Ashkenazi Jews' genetic homogeneity, which directly reinforces the existing 'Ashkenazi Genetic Origins' concept. No new concept is needed as the pattern is already captured. The concept strength is increased from 0.7 to 0.8 to reflect additional evidence.",
  "operations": [
    {
      "op": "ADD_NODE",
      "node_type": "event",
      "data"...
```

---

### 样本 0 (两者均解析成功)

**Ground Truth (摘要)**:

```json
{
  "reasoning": "The new event describes a specific song written by Lennon-McCartney, directly supporting the existing concept of their partnership. It reinforces the pattern of their collaborative songwriting output, so we connect it with a grounds edge and strengthen the concept.",
  "operations": [
    {
      "op": "ADD_NODE",
      "node_type": "event",
      "data": {
        "event_id": "f2e0c970-fc0d-4027-b0c7-aaea21f9589c",
        "title": "It's for You",
        "event_type": "inform...
```

**Base 预测 (摘要)**:

```
```json
{
  "reasoning": "The new event introduces 'It's for You' as a song by Lennon and McCartney, confirming their collaborative work and specifying it as a McCartney composition. This reinforces the existing concept of the Lennon–McCartney partnership and adds specific detail about a song attributed to them. No new concept is needed as the pattern is already captured in the existing concept. T...
```

**LoRA 预测 (摘要)**:
```
{
  "reasoning": "The event describes a song written by Lennon and McCartney, directly reinforcing the existing concept of their partnership. No new concept is needed as only two events are present. The concept strength is increased from 0.5 to 0.7.",
  "operations": [
    {
      "op": "ADD_NODE",
      "node_type": "event",
      "data": {
        "event_id": "f2e0c970-fc0d-4027-b0c7-aaea21f9589...
```

---