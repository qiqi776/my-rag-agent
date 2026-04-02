# Minimal Modular RAG Project

当前仓库处于 M4：在 M3 hybrid retrieval 的基础上，已经补齐稳定的 response schema 和 citations，并保留文档生命周期的 list/delete 契约。

- 中文规格文档：`specs/minimal-modular-rag-project.md`
- Python 工程配置：`pyproject.toml`
- 配置样例：`config/settings.yaml.example`
- 源码目录：`src/`
- 测试目录：`tests/`

## 当前状态

当前已经完成的能力：

- `load -> split -> embed -> store` 的本地文本摄取链路
- `query -> dense retrieve -> SearchOutput` 的查询链路
- `query -> dense retrieve -> sparse retrieve -> RRF fuse -> SearchOutput` 的 hybrid 查询链路
- 基于 JSONL 的 ingestion / query traces
- 文档生命周期操作：`list` / `delete`
- CLI 入口：`mrag-ingest`、`mrag-query`、`mrag-docs`
- adapter base contracts：`BaseLoader`、`BaseEmbedding`、`BaseVectorStore`
- factory 装配：CLI 通过配置选择具体 provider
- sparse retrieval 组件：`SparseRetriever`
- fusion 组件：`rrf_fuse`
- response 组件：`ResponseBuilder`、`SearchOutput`、`Citation`

当前内置 provider：

- loader: `text`
- embedding: `fake`
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

- MCP / HTTP 接口
- Rerank
- LLM answer synthesis
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
默认 retrieval mode 是 `dense`，也可以通过配置或 CLI 参数切换成 `hybrid`。
查询服务现在输出稳定的 `SearchOutput` 结构，包含 `results` 和 `citations`，后续可直接复用于 CLI、MCP 或 HTTP 接口。

示例命令：

```bash
uv run mrag-ingest ./docs --collection knowledge
uv run mrag-query "semantic embeddings" --collection knowledge
uv run mrag-query "semantic embeddings" --collection knowledge --mode hybrid
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
- 在不改 application 层的前提下接入 rerank、MCP、HTTP

详细需求、边界和后续里程碑请参考：

```text
specs/minimal-modular-rag-project.md
```
