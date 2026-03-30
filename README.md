# Minimal Modular RAG Project

这是一个参考 `MODULAR-RAG-MCP-SERVER` 架构思路、但以更小范围起步的模块化 RAG/Agent 基础项目。
当前仓库的第一版 MVP：本地文本摄取、dense 检索、文档生命周期和可观测性闭环。

- 中文规格文档：`specs/minimal-modular-rag-project.md`
- Python 工程配置：`pyproject.toml`
- 配置样例：`config/settings.yaml.example`
- 源码目录：`src/`
- 测试目录：`tests/`

## 当前状态

当前已经完成的第一版能力：

- `load -> split -> embed -> store` 的本地文本摄取链路
- `query -> dense retrieve -> top_k response` 的查询链路
- 基于 JSONL 的 ingestion / query traces
- 基于持久化快照的 in-memory vector store
- 文档生命周期操作：`list` / `delete`
- CLI 入口：`mrag-ingest`、`mrag-query`、`mrag-docs`

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
```

详细需求、边界和后续里程碑请参考：

```text
specs/minimal-modular-rag-project.md
```
