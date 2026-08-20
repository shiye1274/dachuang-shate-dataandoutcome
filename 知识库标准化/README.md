# 沙特市场知识库标准化基线

本目录保存 `ksa-20260820-v1` 的第一版标准化知识库基线，服务于沙特市场的合规、文献、案例、问答和营销表达场景。它不是实时法规数据库；正式使用前，涉及海关、税务、SABER、PDPL 或其他准入要求的内容仍应回到官方来源复核。

## JSONL 记录约定

每行是一个独立 JSON 对象。网页导入器使用 `id`、`dimension`、`sub_category`、`title`、`text_primary`、`text_zh`、`text_en`、`tags` 和 `source` 进行检索；以下字段用于版本和新鲜度管理：

| 字段 | 含义 |
|---|---|
| `published_at` | 原始来源发布日期；未知时为 `null` |
| `effective_from` / `effective_to` | 规则生效和失效日期；未知时为 `null` |
| `retrieved_at` | 本次抓取/整理时间，不等于法规生效时间 |
| `knowledge_version` | 本次知识库发布版本 |
| `content_hash` | 正文 SHA-256，用于判断内容是否真正变化 |
| `review_status` | `pending`、`approved` 或 `rejected` |
| `source.revision` | 来源仓库或导入批次版本 |
| `source.url` | 原始来源 URL；没有 URL 时保留来源名称和文件路径 |
| `supersedes_id` | 新记录替代旧记录时使用的旧记录 ID |

## 当前版本

版本和记录数量以 `knowledge_manifest.json` 为准。`retrieved_at` 用于回答“什么时候收集的”，`published_at` 用于回答“来源什么时候发布的”，`effective_from` 用于回答“规则什么时候开始生效”，三者不能混用。

首次导入属于基线整理，所有由 PDF、CSV、Markdown 或 JSON 转换得到的记录默认进入 `pending` 状态。后续自动更新只生成候选记录和 Pull Request，合规内容经人工复核后再合并到正式分支。
