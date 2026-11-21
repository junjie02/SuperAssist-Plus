# 题目文件与 Manifest 格式

## 题目 Markdown

每道原试卷题使用一个 ASCII 规范文件，如 `question-01.md` 或 `question-02.md`。同一道题后续续传时先读再合并写回，不另建 `续1`、`续2` 文件。标题不可省略；没有信息时明确写“未提供”，不要删除字段。

```markdown
# 第1题

## 基本信息

- 原试卷题号：第2题
- 题目信息状态：未完 / 完整
- 缺失位置：材料第4段在“……”处中断 / 无
- 题型：归纳概括（summary）
- 满分：20
- 字数限制：不超过200字

## 题干

逐字题干。

## 相关材料

只放本题所需材料，并保留段落序号。若整段材料同时服务多题，可以完整重复，避免评分实例缺上下文。

## 官方答案或评分参考

- 状态：已提供 / 未提供
- 来源：试卷附带官方答案 / 用户提供的官方评分参考 / 未提供

官方原文逐字转录。状态为“未提供”时，本节只写“未提供”，不得由主 Agent 生成、概括或模拟采分点。

## 考生答案逐字转写

- 状态：待上传 / 未完 / 已提供

有多少可靠内容写多少。后续答案先读取本文件，再把新内容合并到本节；确认该题答案完整后才改为“已提供”。逐字保留原答案，不纠错、不润色、不重排。

## OCR疑点

- 第2行“治理/管理”字迹不清，暂记为“治理”。
- 末尾约3字无法辨认，标记为 `[无法辨认]`。
```

不得把主 Agent 推导的采分点、另一题答案、评分意见或其他评分员分数写入题目文件。采分依据由 A/B/C 专家实例分别独立推导并保存在各自结果中。

## manifest.json

`reserve` 会先创建 `status=reserved` 的占位 manifest。每次创建或续写题目文件后都同步 manifest：题目信息仍有未完时使用 `collecting_questions`，题目均完整但仍缺答案时使用 `awaiting_answers`，题目和答案均完整时才使用 `ready`。以下是原试卷第 2、4 题中第 4 题仍未上传完的示例：

```json
{
  "schema_version": 1,
  "batch_id": "使用 reserve 返回的 batch_id",
  "status": "collecting_questions",
  "user_scope": "使用 reserve 返回目录对应的当前用户范围",
  "thread_id": "当前对话 thread_id",
  "question_count": 2,
  "created_at": "保留 reserve 创建时间；无法读取时可使用当前 ISO 8601 时间",
  "questions": [
    {"number": 1, "type": "analysis", "max_score": 20, "file": "questions/question-01.md"},
    {"number": 2, "type": "essay", "max_score": 40, "file": "questions/question-02.md"}
  ]
}
```

必须使用 `reserve` 返回的真实 `batch_id` 和当前 `thread_id`。`question_count` 必须等于本批次实际评分题数，范围为 1-20。满分按用户试卷填写，示例分值不是固定值。`file` 必须位于本批次 `questions/` 下；manifest 的 `number` 是批次内顺序，必须从 1 连续排列。只评部分题时，在题目 Markdown 的“原试卷题号”中保留原编号。

必须保留 `reserve` 生成的身份字段。manifest 的 `number` 是评分批次内顺序，必须从 1 连续排列；`file` 使用真实的题号文件名。后续图片先调用 `current`，再 `list_files(questions_dir)` 并读取相应文件。新增题目时同步增加 `question_count` 和 `questions`；续传同一道题时只更新原文件。

`start` 只接受 `status=ready`、题目数量与 `question_count` 一致的 manifest，并硬性检查每个题目文件标记 `题目信息状态：完整`，且“题干”和“相关材料”均非空；“考生答案逐字转写”必须标记 `- 状态：已提供` 并含非空答案。任何“未完”、待上传或空内容都会被拒绝，评分 subagent 不会启动。

若用户明确指定只评分前 N 道题，可在启动前调用 `scope` 将 manifest 的评分范围收缩为前 N 道；该操作必须基于用户明确指令，不能由 Agent 自行猜测。
