# Minimal Modular RAG Project

一个本地优先、模块化的 RAG 工程模板，支持文档摄取、检索、问答、MCP 工具暴露，以及基础评估与 trace 观测。它适合作为：

- 本地知识库问答项目的起点
- RAG / MCP / Agent-ready 分层设计的参考实现
- 需要快速接入私有文档检索能力的 Python 工程

- 中文规格文档：`specs/minimal-modular-rag-project.md`
- Python 工程配置：`pyproject.toml`
- 配置样例：`config/settings.yaml.example`

## 核心能力

- 文档摄取：支持 `text`、`pdf`，提供 `mrag-ingest` 和 `mrag-ingest-preview`
- 检索：支持 `dense` 和 `hybrid` 查询
- 问答：支持 `mrag-answer` 与交互式 `mrag-chat`
- 文档管理：支持列出、查看摘要、删除文档
- 可观测性：支持 JSONL traces 与 `mrag-traces`
- 评估：支持 retrieval / answer regression
- Dashboard：支持本地可视化总览、数据浏览、摄取管理、trace 查看和评估面板
- MCP：支持本地工具调用与 stdio 模式
- Agent-ready：提供最小 workflow runner 与工具注册层

## 内置 Provider

- loader: `text`、`pdf`
- embedding: `fake`、`openai`
- llm: `fake`、`openai`
- reranker: `fake`、`llm`、`cross_encoder`
- vector_store: `memory`、`local_json`、`chroma`

## 快速开始

```bash
uv venv
uv sync --extra dev
# 如需 chroma / cross-encoder：
uv sync --extra dev --extra vector --extra rerank
# 如需 Dashboard：
uv sync --extra dev --extra dashboard
```

也可以直接使用本地脚本准备一个 demo 环境：

```bash
bash scripts/start.sh
bash scripts/stop.sh
```

Dashboard 可以通过下面的命令启动：

```bash
bash scripts/start_dashboard.sh --config ./config/settings.yaml.example
```

内置页面：

- `Overview`
- `Data Browser`
- `Ingestion Manager`
- `Ingestion Traces`
- `Query Traces`
- `Evaluation Panel`

典型工作流：

1. 先通过 `mrag-ingest` 或 Dashboard 的 `Ingestion Manager` 导入文档
2. 在 `Data Browser` 查看文档、chunk 和 metadata
3. 在 `Ingestion Traces` / `Query Traces` 排查摄取与检索问题
4. 在 `Evaluation Panel` 运行 retrieval / answer / all 回归

## MCP Tools

当前内置 MCP tools：

- `query_knowledge`
- `list_collections`
- `get_document_summary`
- `list_documents`
- `delete_document`

## 测试

```bash
uv run pytest
uv run ruff check .
```

## 当前边界

- 还没有 HTTP API
- 还没有完整 OCR / Vision LLM / 多模态检索链路
- 还不是完整自主 Agent 平台，不包含 planner、memory、multi-agent orchestration

更多设计细节、边界说明和后续里程碑，请参考 `specs/minimal-modular-rag-project.md`。
