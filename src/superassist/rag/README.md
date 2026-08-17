# SuperAssist Hybrid Agentic RAG

本模块负责用户上传资料的解析、切片、索引、检索和来源审计，与 CogniFold 长期记忆图完全独立。
第一版采用本地优先架构，不依赖外部搜索数据库：

- SQLite 保存原始 Chunk，并通过 FTS5 提供 BM25；
- FAISS 保存同一批 Chunk 的 Dense embedding；
- 应用层用 Reciprocal Rank Fusion（RRF）融合两路排名；
- Agent 可以改写查询并重复检索，Session 跨调用去重和控制累计证据预算。

实体、关系、摘要和文件级来源都不作为证据。**唯一证据单元是实际进入模型上下文的原文 Chunk。**

## 1. 数据流

```text
上传文件
  -> 扩展名/大小/SHA-256 校验
  -> 格式解析
  -> Unicode 与空白规范化
  -> 标题/段落/句子边界切片
  -> 批量 Embedding
  -> SQLite chunks + FTS5
  -> FAISS Dense index

用户问题
  -> Dense Top-N
  -> BM25 Top-N
  -> RRF 融合
  -> 内容 Hash 去重
  -> Top-K + Token 预算
  -> 原文 Chunk 上下文
  -> Agent 回答或改写后继续 rag_search
```

## 2. 存储

每个用户使用 `sha256(user_id)[:24]` 隔离目录：

```text
.superassist/rag/<user-hash>/
  documents.json          文档清单和索引状态
  files/                  原始上传文件
  hybrid.sqlite3          Chunk 表和 FTS5 索引
  chunks.faiss            Dense 向量索引
  chunks.mapping.json     FAISS 行号到 Chunk ID 的映射
```

`chunks` 表保存文档 ID、文件名、顺序、section ID、标题路径、原文、Token 数、内容 Hash 和
float32 embedding。FTS5 只保存为中文 bigram 与英文词元预处理后的检索文本；回答始终读取 `chunks.text`，
不会把分词文本交给模型。

文档状态为 `queued -> parsing -> indexing -> ready|failed`。删除状态为 `deleting`，删除操作同时清理
SQLite、FTS5、FAISS、原文件和 manifest。相同文件内容通过 SHA-256 拒绝重复上传。

## 3. 解析与切片

支持格式由 `documents.py` 声明：TXT、Markdown、JSON、CSV、HTML、PDF、DOCX、PPTX 和 XLSX。
第一版复用对应 Python 解析器，不含版面 OCR；扫描 PDF、复杂表格和图片文字属于后续能力。

`chunking.py` 执行确定性结构切片：

1. NFKC、换行、控制字符和连续空白规范化；
2. 识别 Markdown 标题、章节编号、Slide 和 Sheet 标记；
3. 在同一 section 内按段落和句子边界组装；
4. 默认目标 384 Token、最大 480 Token、重叠 64 Token；
5. Chunk ID 由文档 ID、顺序和内容 Hash 确定。

Token 计数使用完全离线的本地估算器，避免 `tiktoken` 首次运行隐式联网。后续更换 embedding 模型时，
应将它替换为对应模型的本地 tokenizer 并全量重建索引。

## 4. 混合检索

`HybridRAGService.retrieve(user_id, query, mode)` 支持：

| mode | 路径 |
| --- | --- |
| `hybrid` | Dense + BM25 + RRF，默认 |
| `dense` | 只使用 FAISS 语义召回 |
| `bm25` | 只使用 FTS5 关键词召回 |

Dense 使用归一化向量内积，等价于余弦相似度。BM25 对文件名、标题路径和正文使用不同权重。
Hybrid 不直接比较两种异构分数，而按排名计算：

```text
score = 0.55 / (rrf_k + dense_rank) + 0.45 / (rrf_k + bm25_rank)
```

默认每路召回 50 个候选，`rrf_k=60`，融合后最多返回 10 个 Chunk，并按 5000 Token 预算构建上下文。
内容 Hash 相同的 Chunk 只返回一次。每条上下文头包含文件名、Chunk ID 和可选 section：

```text
[上传资料:manual.pdf | chunk:chunk-abc | section:第三章 > 检索]
原始证据文本
```

## 5. Agentic 检索

RAG 模式下，`RagRetrievalMiddleware` 在模型首次调用前自动执行原问题的 `hybrid` 检索。模型认为证据
不足时可以调用：

```text
rag_search(query, mode="hybrid|dense|bm25")
```

系统不设置 RAG 专用的固定三次限制。`RagTurnSession` 维护：

- 已使用查询和来源；
- 已见 Chunk ID，后续检索不重复注入；
- 累计原文证据 Token，默认上限 8000；
- 连续无新增检索次数，默认 2 次后停止；
- 停止原因和完整检索轨迹。

通用 Agent 工具/递归保护仍然有效。动态提示要求 Agent 在没有新 Chunk或证据预算耗尽后停止，不得重复
等价查询。检索异常转换为失败结果，不会中断整轮聊天。

## 6. 配置

| 环境变量 | 默认值 |
| --- | ---: |
| `SUPERASSIST_RAG_MAX_FILE_SIZE_MB` | 25 |
| `SUPERASSIST_RAG_MAX_FILES_PER_BATCH` | 20 |
| `SUPERASSIST_RAG_CHUNK_TARGET_TOKENS` | 384 |
| `SUPERASSIST_RAG_CHUNK_MAX_TOKENS` | 480 |
| `SUPERASSIST_RAG_CHUNK_OVERLAP_TOKENS` | 64 |
| `SUPERASSIST_RAG_CANDIDATE_TOP_K` | 50 |
| `SUPERASSIST_RAG_CHUNK_TOP_K` | 10 |
| `SUPERASSIST_RAG_RRF_K` | 60 |
| `SUPERASSIST_RAG_CONTEXT_MAX_TOKENS` | 5000 |
| `SUPERASSIST_RAG_ACCUMULATED_EVIDENCE_MAX_TOKENS` | 8000 |
| `SUPERASSIST_RAG_STAGNANT_SEARCH_LIMIT` | 2 |

## 7. 评测合同

Retriever 评测只能使用 `hits[].chunk_id`，不能使用文档 ID、来源 ID、摘要或其它派生数据冒充证据。
Dense、BM25、Hybrid 和后续 Reranker 的对照必须使用相同原文 Chunk、最终 K 和可见上下文 Token 预算。

第一阶段消融组：

1. Dense only；
2. BM25 only；
3. Dense + BM25 + RRF；
4. 后续增加 Hybrid + Reranker。

报告 Recall@K、Precision@K、All-evidence@K、MRR、gold quote 覆盖率、回答准确率、P50/P95 延迟；
Token 必须拆成检索上下文、在线检索模型调用和回答输入输出，不能统称“检索 Token”。

## 8. 当前限制

- SQLite/FAISS 适合单机第一版，不支持多进程共同写同一用户索引；
- 中文 BM25 使用内置 bigram 分词，尚未接入专业中文 analyzer；
- 没有 OCR、父子 Chunk 扩展、ACL、版本切换和事务 outbox；
- 没有 Cross-Encoder reranker；
- FAISS 在文档新增或删除时按用户全量重建。

这些边界由 `Retriever` 和 `HybridRAGService` 接口隔离。升级企业版时可把存储替换为 PostgreSQL、对象存储
和 OpenSearch，而不改变 Agent 工具与原文证据合同。
