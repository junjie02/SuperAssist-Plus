---
name: hubei-shenlun-grading
description: Persist Hubei Shenlun uploads incrementally into one readable file per original question, continue incomplete files across later image batches, append candidate answers, and grade only after each question and answer is complete. Use for 申论题目梳理或建档、分批上传题目或答案、湖北省考申论批改、申论估分、逐题评分或作文点评。
---

# 湖北申论答卷评分

主 Agent 把每批图片中能够确认的内容立即按原题号写入独立文件。题目跨图片或跨消息时，文件可保持“未完”；后续先读取旧文件，再把新内容归入同一道题并更新完整性。评分 subagent 只读取已经完整的题目文件，主 Agent 不制定采分点、不参与打分。

## 必读引用

- 第一次处理本套题目前读取 [question-file-format.md](references/question-file-format.md)，采用其中的文件状态和 manifest 契约。
- 只有用户明确要求评分，且所有题目和答案均完整时才读取 [scoring-rules.md](references/scoring-rules.md)。仅保存或续写内容时不要读取评分规则。
- 汇总前读取 [final-report-format.md](references/final-report-format.md)，按规定组织用户可见报告。

## 保存规则

- 同一套题第一次出现时调用 `shenlun_grading(action="reserve", question_count=当前已知题数)`；后续批次调用 `current` 找回它。使用返回的 `questions_dir`，不要另建位置。若后来识别出更多题，可增加文件并同步增大 manifest 的 `question_count` 和 `questions`。
- 文件名统一使用 ASCII 规范 `questions/question-01.md`、`questions/question-02.md`。原试卷题号和小题号写在文件内容的“原试卷题号”字段中；不得使用中文或整批图片名，也不得把多道题写进一个文件。
- 每收到一批图片，都先调用 `list_files(questions_dir)`。同一道题已有文件时先 `read_file`，再把新识别的题干、材料、官方参考或答案归入对应段落，并用 `write_file` 写回完整合并结果；不得覆盖丢失旧内容，也不得把新内容机械堆到错误段落末尾。
- 有多少可靠内容就写多少。题目尚跨页、材料尚未结束或题号归属仍不确定时，文件写 `题目信息状态：未完`，同时记录缺少哪一页或停在什么文字。补充图片到达后继续该文件；确认题干和相关材料完整时改为 `题目信息状态：完整`。
- 一批图片包含多道题时，按题号拆开写入不同文件。共享材料可原文写入每个相关题目文件，确保评分 subagent 单独读取一个文件就有完整上下文。归属无法判断的片段不要猜写，保留在 OCR 疑点并向用户说明。
- 每次落盘后同步 manifest：`questions` 只列已经创建的文件并使用真实相对路径；仍有题目信息未完时设 `status=collecting_questions`，全部题目信息完整但仍缺答案时设 `status=awaiting_answers`，所有题目与答案均完整时才设 `status=ready`。
- 写完 manifest 后先调用 `shenlun_grading(action="check", manifest_path=...)`。返回 `ready=false` 时继续收集，不得调用 `start`；只有返回 `ready=true` 且用户明确要求评分时才启动。
- 若当前会话此前已经明确表达“批改/评分/估分”意图，后续只上传答案图片也视为继续该评分请求；保存答案后直接执行 `check`，通过后启动评分。若会话没有既有评分意图，单独上传图片仍先询问是否评分。
- 若用户明确说“先批前 N 题”或“只批这 N 道”，调用 `shenlun_grading(action="scope", question_count=N, manifest_path=...)` 收缩为批次前 N 道，再执行 `check`；不得自行猜测用户想跳过哪些题。
- 批改工作区的文件只能通过 `write_file` 或评分工具写入，禁止在 Shell 中使用补丁、重定向或脚本直接修改 manifest/题目文件。

## 答案与评分

- 用户分批上传答案时采用同样的读后合并方式。答案不完整可以先写并标记 `状态：未完`；确认该题答案上传完整后改为 `状态：已提供`。保留原措辞、错别字和 `[无法辨认]`，不纠错、不润色、不重排。
- 仅梳理、保存或续写题目与答案时，**禁止调用 `start`、禁止直接调用 `task`、禁止启动任何评分 subagent**。
- 只有用户明确要求批改、评分或估分，而且 manifest 中每道题的题目信息与答案都完整时，才调用 `shenlun_grading(action="start", manifest_path=...)`。答案完整本身不等于用户授权评分。
- 文件含有 `OCR疑点` 时，必须先向用户确认题目数量、题型和答案转写；确认后在 manifest 或对应题目中设置 `ocr_confirmed=true`，再执行 `check`/`start`。
- 启动后保存 `job_id` 并重复调用 `next` 收齐结果；超时继续等待，恢复检查使用 `status`。同题双评、三评和仲裁只由评分程序控制。完成后以 `summary.json` 和各题 `final.json` 汇总。

## 约束

- 一个批次只处理同一套题目及其后来上传的续页和答案。明确开始另一套题时才重新 `reserve`。
- 不得因题目数量不是 5 而拒绝评分。部分题目经过用户明确指定或确认后，按实际题数创建独立批次。
- 主 Agent 不得创建、改写或补全采分点。官方评分参考缺失时只记录“未提供”，由每个题型专家实例独立推导本次评分依据。
- 官方评分参考存在时，主 Agent 只逐字转录并记录来源，不得解释其含义或预先判断考生是否命中。
- 不向任一普通评分实例透露其他评分员分数。只有评分程序启动的仲裁模式可读取 A/B/C 结果。
- 飞书进度由评分事件自动更新。不要用普通进度文字猜测尚未落盘的分数。
- 最终回复以 `summary.json` 和各题 `final.json` 为准，不以模型记忆中的中间文本为准。必须逐题按“题目→解析”的顺序输出：每道题先原样展示该题 `question.stem`，紧接该题评分解析，再进入下一道题；禁止只给解析，也禁止先集中列完全部题目再集中给解析。
- 汇总评分时逐项展示每位评分员的全部 `score_components` 和 `deduction_total`，并核对分项之和与 `score`、最终取平均或仲裁公式一致；不得把“对策采分点分”误写成完整对策分，也不得在文字结论中否认同一评分员 JSON 已明确给出的命中或遗漏。
