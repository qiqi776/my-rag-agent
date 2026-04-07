# Minimal Modular RAG Project

当前仓库处于 M9：在 M8 的增强型 ingestion 基础上，已经补上 agent-ready tool registry、deterministic workflow state 和最小 workflow CLI，把项目推进到“可被 Agent 安全复用的 RAG 工程”阶段。

- 中文规格文档：`specs/minimal-modular-rag-project.md`
- Python 工程配置：`pyproject.toml`
- 配置样例：`config/settings.yaml.example`
- 源码目录：`src/`
- 测试目录：`tests/`

## 当前状态

当前已经完成的能力：

- `load -> split -> embed -> store` 的本地文本摄取链路
- `load -> transform -> split -> embed -> store` 的增强型摄取链路
- `query -> dense retrieve -> SearchOutput` 的查询链路
- `query -> dense retrieve -> sparse retrieve -> RRF fuse -> SearchOutput` 的 hybrid 查询链路
- agent-ready tool registry 与 workflow runner
- `research_and_answer` 最小 workflow
- 基于 JSONL 的 ingestion / query traces
- trace exploration：`TraceReader`、`mrag-traces`
- 文档生命周期操作：`list` / `delete`
- CLI 入口：`mrag-ingest`、`mrag-query`、`mrag-docs`、`mrag-answer`、`mrag-traces`、`mrag-eval`、`mrag-agent`
- MCP 入口：`mrag-mcp`
- adapter base contracts：`BaseLoader`、`BaseEmbedding`、`BaseVectorStore`
- factory 装配：CLI 通过配置选择具体 provider
- sparse retrieval 组件：`SparseRetriever`
- fusion 组件：`rrf_fuse`
- response 组件：`ResponseBuilder`、`SearchOutput`、`Citation`
- answer 组件：`AnswerBuilder`、`AnswerOutput`、`AnswerService`
- evaluation 组件：`RetrievalEvalRunner`、`AnswerEvalRunner`
- MCP 组件：本地可测的 tool server、共享 mapper、共享依赖装配层
- ingestion 组件：`IngestionPipeline`、页级 `IngestionUnit`、transform plugins
- agent 组件：`ToolRegistry`、`WorkflowState`、`WorkflowRunner`、agent-ready tools

当前内置 provider：

- loader: `text`
- loader: `pdf`
- embedding: `fake`
- llm: `fake`
- reranker: `fake`
- vector_store: `memory`（纯内存，仅适合单进程测试）
- vector_store: `local_json`（本地 JSON 快照，当前 CLI 默认使用）

当前内置 retrieval mode：

- `dense`
- `hybrid`

当前的 vector store 边界：

- `InMemoryVectorStore` 现在只负责内存数据结构和检索逻辑，不再写文件
- `LocalJsonVectorStore` 负责把内存状态持久化到 `storage_path`
- 这样 application 层只依赖 `BaseVectorStore`，后续切换真实持久化后端时不需要改 service 层

当前未做：

- HTTP 接口
- 真实外部 LLM / reranker provider
- 完整多模态检索 / OCR / 外部图像大模型链路
- 复杂自主 Agent loop / planner / 长时记忆
- 重量级 Dashboard / 在线 Evaluation 平台

## 目录结构

```text
config/
specs/
src/
tests/
```

## 开发与运行

```bash
uv venv
uv sync --extra dev
```

默认配置中的 vector store provider 是 `local_json`，因此 CLI 多次运行之间会复用本地快照。
默认 retrieval mode 是 `dense`，也可以通过配置或 CLI 参数切换成 `hybrid`。
查询服务现在输出稳定的 `SearchOutput` 结构，包含 `results` 和 `citations`，后续可直接复用于 CLI、MCP 或 HTTP 接口。
答案服务会在 `SearchOutput` 之上继续执行 `rerank -> answer synthesis`，输出独立的 `AnswerOutput`。
默认 loader provider 仍然是 `text`；切换到 `pdf` 后，ingestion pipeline 会在 chunking 之前先执行 page-aware transforms。

当前 MCP 层复用的是同一套 application services：

- `SearchService`
- `DocumentService`
- `SearchOutput`

当前 M9 的 Agent-ready 层同样复用 application services：

- `SearchService`
- `AnswerService`
- `DocumentService`
- `IngestService`

示例命令：

```bash
uv run mrag-ingest ./docs --collection knowledge
uv run mrag-query "semantic embeddings" --collection knowledge
uv run mrag-query "semantic embeddings" --collection knowledge --mode hybrid
uv run mrag-docs list --collection knowledge
uv run mrag-docs delete <doc_id> --collection knowledge
uv run mrag-answer "semantic embeddings" --collection knowledge
uv run mrag-traces stats
uv run mrag-eval all --retrieval-fixtures ./tests/fixtures/evaluation/retrieval_cases.json --answer-fixtures ./tests/fixtures/evaluation/answer_cases.json
uv run mrag-mcp list-tools
uv run mrag-mcp call-tool query_knowledge --arguments-json '{"query":"semantic embeddings","collection":"knowledge"}'
uv run mrag-agent list-tools
uv run mrag-agent list-workflows
uv run mrag-agent run-workflow research_and_answer "semantic embeddings" --collection knowledge --mode hybrid
```

## MCP 接口

当前 MCP 接口层位于 `src/interfaces/mcp/`，分为四部分：

- `dependencies.py`
  - 统一完成 settings、provider 和 application service 装配
- `mappers.py`
  - 把 `SearchOutput`、`DocumentSummary`、`DeleteDocumentResult` 映射成 MCP payload
- `tools/`
  - 只做参数解析和调用 service
- `server.py`
  - 提供本地可测的 tool registry、错误映射和调用入口

当前内置 MCP tools：

- `query_knowledge`
- `list_documents`
- `delete_document`

这层不会复制 retrieval 或 document lifecycle 逻辑，只做协议映射。

## Answer Flow

当前 M6 的生成链路是：

```text
query -> SearchService -> SearchOutput -> Reranker -> FakeLLM -> AnswerOutput
```

关键约束：

- `SearchOutput` 仍然是 retrieval-level contract
- `AnswerOutput` 是 answer-level contract
- 不把生成逻辑塞回 `SearchService`
- 当前 LLM 和 reranker 都是本地 fake 实现，用于打通链路和稳定测试

## Observability And Evaluation

当前 M7/M8 组合后，项目具备三类支撑能力：

- `TraceReader`
  - 读取 JSONL traces
  - 按 `ingestion / query / answer` 过滤
  - 汇总阶段次数与耗时
- ingestion pipeline
  - `PdfLoader` 提供 PDF 文本抽取与页级 metadata
  - `MetadataEnrichmentTransform`
  - `ChunkRefinementTransform`
  - `ImageCaptioningTransform`（stub / fake，可关闭）
- regression runner
  - `RetrievalEvalRunner`
  - `AnswerEvalRunner`

对应命令：

```bash
uv run mrag-ingest ./tests/fixtures/evaluation/corpus --collection knowledge
uv run mrag-traces list --limit 10
uv run mrag-traces stats --trace-type answer
uv run mrag-eval retrieval --fixtures ./tests/fixtures/evaluation/retrieval_cases.json
uv run mrag-eval answer --fixtures ./tests/fixtures/evaluation/answer_cases.json
```

这些能力建立在已有的 `SearchOutput`、`AnswerOutput` 和 JSONL trace contract 之上，不需要把评估逻辑塞回 `SearchService` 或 `AnswerService`。默认 regression fixtures 假定你已经先 ingest 了 `tests/fixtures/evaluation/corpus/` 里的语料。

## M8 Ingestion

当前 M8 的 ingestion 重点是扩展边界，而不是一步做到完整多模态系统：

- `PdfLoader`
  - 输出统一 `Document`
  - 在 metadata 中保留 `source_path`、`doc_type`、`page_count` 和 `pages`
- `IngestionPipeline`
  - 把 loader 输出扩成页级 `IngestionUnit`
  - 在 `load` 和 `split` 之间执行 transforms
  - 关闭 transforms 时退化为原有文本 ingest 行为
- transforms
  - `metadata_enrichment`：补 `section_title`
  - `chunk_refinement`：规则清洗空白与页噪声
  - `image_captioning`：当前是 stub / fake 扩展点，不依赖外部视觉模型

当前页码 metadata 会进入 chunk record，并在 query / answer citations 中保留 `page` 字段，方便后续回溯来源。

## M9 Agent-Ready Layer

当前 M9 的重点不是做复杂自主 Agent，而是把现有 RAG 工程整理成稳定可复用的上层能力：

- `ToolRegistry`
  - 注册、列出、查找和调用 agent-ready tools
  - 错误统一映射为稳定 `ToolResult`
- agent-ready tools
  - `search_knowledge`
  - `answer_question`
  - `list_documents`
  - `delete_document`
  - 这些 tool 全部复用现有 application services，不复制业务逻辑
- `WorkflowState`
  - 记录 `workflow_id`
  - 记录调用过的 tools、输入、输出摘要、中间结果和最终输出
- `WorkflowRunner`
  - 当前内置 `research_and_answer`
  - 按固定顺序执行 `search_knowledge -> answer_question`
  - 输出稳定 JSON，便于 CLI、脚本和后续 runtime 复用

当前边界：

- 不是完整自主 Agent 平台
- 不包含 planner、memory、multi-agent orchestration
- workflow 仍然是 deterministic stub，而不是开放式 Agent loop

## 测试

```bash
uv run pytest
uv run ruff check .
```

## 下一步扩展点

- 新增真实 embedding provider，并通过 `src/adapters/embedding/factory.py` 接入
- 新增真实 vector store 后端，并实现 `BaseVectorStore`
- 新增真实 LLM / reranker provider，并通过 factory 接入
- 在当前 transform contract 上继续扩展 OCR、图像 captioning 和更复杂的 PDF 版面处理
- 在当前 agent-ready contract 上继续扩展 planner、memory 和更复杂的 workflow orchestration
- 在不改 application 层的前提下接入 HTTP、真实 MCP SDK/stdio transport

详细需求、边界和后续里程碑请参考：

```text
specs/minimal-modular-rag-project.md
```
