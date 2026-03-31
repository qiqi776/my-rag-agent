# Minimal Modular RAG Project

当前仓库处于 M2：在 M1 本地文本 RAG MVP 的基础上，已经补齐 provider 抽象和 factory 装配。

- 中文规格文档：`specs/minimal-modular-rag-project.md`
- Python 工程配置：`pyproject.toml`
- 配置样例：`config/settings.yaml.example`
- 源码目录：`src/`
- 测试目录：`tests/`

## 当前状态

当前已经完成的能力：

- `load -> split -> embed -> store` 的本地文本摄取链路
- `query -> dense retrieve -> top_k response` 的查询链路
- 基于 JSONL 的 ingestion / query traces
- 文档生命周期操作：`list` / `delete`
- CLI 入口：`mrag-ingest`、`mrag-query`、`mrag-docs`
- adapter base contracts：`BaseLoader`、`BaseEmbedding`、`BaseVectorStore`
- factory 装配：CLI 通过配置选择具体 provider

当前内置 provider：

- loader: `text`
- embedding: `fake`
- vector_store: `memory`（纯内存，仅适合单进程测试）
- vector_store: `local_json`（本地 JSON 快照，当前 CLI 默认使用）

当前的 vector store 边界：

- `InMemoryVectorStore` 现在只负责内存数据结构和检索逻辑，不再写文件
- `LocalJsonVectorStore` 负责把内存状态持久化到 `storage_path`
- 这样 application 层只依赖 `BaseVectorStore`，后续切换真实持久化后端时不需要改 service 层

当前未做：

- MCP / HTTP 接口
- 稀疏检索与融合
- Rerank
- 多模态处理
- Dashboard / Evaluation

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

示例命令：

```bash
uv run mrag-ingest ./docs --collection knowledge
uv run mrag-query "semantic embeddings" --collection knowledge
uv run mrag-docs list --collection knowledge
uv run mrag-docs delete <doc_id> --collection knowledge
```

## 测试

```bash
uv run pytest
uv run ruff check .
```

## 下一步扩展点

- 新增真实 embedding provider，并通过 `src/adapters/embedding/factory.py` 接入
- 新增真实 vector store 后端，并实现 `BaseVectorStore`
- 在不改 application 层的前提下接入 sparse retrieval、fusion、MCP

详细需求、边界和后续里程碑请参考：

```text
specs/minimal-modular-rag-project.md
```
