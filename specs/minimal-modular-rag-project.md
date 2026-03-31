# 规格说明：最小化模块化 RAG 项目

## 概述

本规格定义了一个新的 RAG 项目。它会借鉴现有 Modular RAG MCP Server 的结构设计，但以更小、更可控的范围起步。

目标不是逐功能复制原项目，而是构建一个具备相同架构优点的新项目：

- 清晰的领域契约
- 分离的摄取链路与检索链路
- 可插拔的外部 provider
- 足够薄的接口层
- 可追踪的执行过程

项目必须采用增量式交付。第一阶段里程碑是一个本地、仅 dense 检索的文本摄取与查询系统。稀疏检索、MCP、重排、多模态处理、Dashboard 和评估等高级能力，明确推迟到后续里程碑。

### 产品目标

构建一个模块化 RAG 基础工程，能够摄取本地文本文档、存储 chunk 表示，并通过稳定的应用层 API 和 CLI 返回相关结果。

### MVP 非目标

- PDF 解析
- 图像理解
- 重排
- 多 provider 集成
- Dashboard UI
- 评估框架
- 生产部署

## 需求

### 功能性需求

- FR-1: 项目应定义稳定的核心契约，包括 `Document`、`Chunk`、`ChunkRecord`、`ProcessedQuery` 和 `RetrievalResult`。
- FR-2: 项目应从版本化 YAML 文件加载配置，并在启动时校验必填字段。
- FR-3: 项目应支持将本地 `.txt` 和 `.md` 文档摄取到命名 collection 中。
- FR-4: MVP 写入链路应支持 `load -> split -> embed -> store`。
- FR-5: MVP 读取链路应支持 `query -> dense retrieve -> top_k response`。
- FR-6: 项目在增加 MCP 或 HTTP 接口之前，应先通过 CLI 暴露 MVP。
- FR-7: 项目应将摄取和查询过程的结构化执行 trace 写入 JSON Lines。
- FR-8: 在接口层扩展之前，项目应至少支持列出和删除已摄取文档这两类文档生命周期操作。
- FR-9: 项目应在 MVP 端到端闭环验证通过之后，再引入 provider 抽象。
- FR-10: 项目应支持后续扩展到稀疏检索、融合、重排、MCP 与多模态 transform，同时不破坏核心契约。

### 非功能需求

- NFR-1: 架构应清晰区分领域契约、应用编排、适配器、接口层和可观测性模块。
- NFR-2: MVP 测试应通过 fake 或 in-memory 适配器运行，且不依赖外部网络。
- NFR-3: 在引入外部 provider 之前，核心契约与纯编排逻辑应先由单元测试覆盖。
- NFR-4: 系统应能基于 `uv`、`pytest` 和 `ruff` 在本地稳定运行。
- NFR-5: 在引入真实持久化后端之前，必须先明确 ID、metadata key、collection 命名规则和存储布局。
- NFR-6: 接口层不得承载业务逻辑；它们只能做输入校验、调用应用服务、映射输出。
- NFR-7: 日志能力不得耦合 stdout 假设，以便未来安全接入 MCP stdio。

## 架构

### 分层模型

- `core`: 领域类型、配置、错误定义、trace 契约、共享规则
- `application`: 摄取、检索、文档生命周期的编排服务
- `adapters`: loader、embedding client、vector store、sparse index、LLM 等外部依赖实现
- `response`: 输出整形、引用、格式化契约
- `interfaces`: 先做 CLI，后续再加 MCP 或 HTTP
- `observability`: logger、trace 持久化，后续扩展 dashboard 和 evaluation

### 推荐项目结构

- `src/core/types.py`
- `src/core/settings.py`
- `src/core/errors.py`
- `src/core/trace.py`
- `src/application/ingest_service.py`
- `src/application/search_service.py`
- `src/application/document_service.py`
- `src/ingestion/pipeline.py`
- `src/retrieval/dense_retriever.py`
- `src/retrieval/sparse_retriever.py`
- `src/retrieval/fusion.py`
- `src/response/response_builder.py`
- `src/adapters/loader/base_loader.py`
- `src/adapters/loader/text_loader.py`
- `src/adapters/embedding/base_embedding.py`
- `src/adapters/embedding/fake_embedding.py`
- `src/adapters/vector_store/base_vector_store.py`
- `src/adapters/vector_store/in_memory_store.py`
- `src/adapters/*/factory.py`
- `src/interfaces/cli/ingest.py`
- `src/interfaces/cli/query.py`
- `src/interfaces/mcp/`
- `src/observability/logger.py`
- `src/observability/trace_store.py`
- `config/settings.yaml.example`
- `tests/unit`
- `tests/integration`
- `tests/e2e`

### 应用服务

- `IngestService`: 协调文件发现、加载、切分、embedding、存储与 trace 创建
- `SearchService`: 协调 query 预处理、检索、response 构建与 trace 创建
- `DocumentService`: 负责列出、查看、删除和重建索引等文档生命周期操作

### 数据与 ID 约定

- 文档 ID 格式：稳定哈希，或基于路径的确定性 ID
- Chunk ID 格式：`{doc_id}_{chunk_index:04d}_{content_hash}`
- 所有持久化记录必须包含的 metadata key：`source_path`、`collection`、`doc_id`、`chunk_index`
- 本地 MVP 的存储布局应至少包含 `data/raw/`、`data/db/` 和 `data/traces/`
- Trace 类型固定为 `ingestion` 和 `query`

## 交付计划

### 里程碑 M1：仅 Dense 的本地 MVP

范围：

- 本地 `.txt` 和 `.md` 摄取
- 确定性切分
- fake embedding 适配器
- in-memory vector store
- CLI ingest 与 query
- JSONL traces

主要交付物：

- `src/core/types.py`
- `src/core/settings.py`
- `src/application/ingest_service.py`
- `src/application/search_service.py`
- `src/adapters/loader/text_loader.py`
- `src/adapters/embedding/fake_embedding.py`
- `src/adapters/vector_store/in_memory_store.py`
- `src/interfaces/cli/ingest.py`
- `src/interfaces/cli/query.py`
- `tests/unit/*`
- `tests/integration/test_ingest_query_mvp.py`

验收意图：

- 本地文本文件可以被成功摄取
- 查询可以返回与摄取内容相关的 chunk
- 全部测试在无网络环境下通过

### 里程碑 M2：Provider 抽象与真实持久化

范围：

- 引入 `BaseLoader`、`BaseEmbedding`、`BaseVectorStore`
- 增加 factory
- 将 vector storage 从内存版切换到本地持久化后端

主要交付物：

- `src/adapters/*/base_*.py`
- `src/adapters/*/factory.py`
- 一个持久化 vector store 实现

验收意图：

- application 服务不依赖具体 provider
- 切换 adapter 时无需修改 service 层逻辑

### 里程碑 M3：稀疏检索与融合

范围：

- 增加 sparse index
- 增加 `SparseRetriever`
- 增加融合策略，第一版使用 RRF

主要交付物：

- `src/retrieval/sparse_retriever.py`
- `src/retrieval/fusion.py`
- `tests/unit/test_fusion.py`
- `tests/integration/test_hybrid_search.py`

验收意图：

- 同时支持 dense-only 和 hybrid retrieval
- 融合行为具备确定性并被测试覆盖

### 里程碑 M4：Response 层与文档生命周期

范围：

- 标准化查询输出
- 增加 citations
- 增加文档列表与删除能力

主要交付物：

- `src/response/response_builder.py`
- `src/application/document_service.py`
- `tests/unit/test_response_builder.py`
- `tests/integration/test_document_lifecycle.py`

验收意图：

- 查询输出结构在不同接口中保持稳定
- list/delete 操作能跨多个 store 协调完成

### 里程碑 M5：接口层扩展

范围：

- 保留 CLI
- 增加 MCP 或 HTTP 作为第一个网络接口

主要交付物：

- `src/interfaces/mcp/` 或 `src/interfaces/http/`
- 薄的 request/response 映射层
- 端到端 smoke tests

验收意图：

- 接口层仅调用 application services
- 相同的 search 和 document 流程可以同时通过 CLI 和接口 API 工作

### 里程碑 M6：高级检索与质量层

范围：

- reranker
- evaluation
- dashboard 或 trace explorer

主要交付物：

- `src/retrieval/reranker.py`
- `src/observability/dashboard/`
- `src/evaluation/`

验收意图：

- 可以对检索质量进行测量
- 无需手读原始 JSONL 也能检查 traces

### 里程碑 M7：多模态与增强模块

范围：

- PDF
- 图像提取
- 图像 caption
- metadata enrichment
- chunk refinement

主要交付物：

- `src/adapters/loader/pdf_loader.py`
- `src/ingestion/transforms/`
- 相关测试与 fixtures

验收意图：

- 多模态扩展能够以插件式方式接入现有 ingestion pipeline，而不改动核心契约

## 构建顺序

1. 在 `README.md` 中定义产品范围、非目标和目标用户。
2. 实现 `core` 类型及其单元测试。
3. 实现配置加载与校验。
4. 实现 logger 和 trace store。
5. 使用 fake adapters 打通 MVP 的 application services。
6. 增加 CLI 命令和端到端 MVP 集成测试。
7. 抽取基础 adapter 接口与 factories。
8. 增加真实持久化后端。
9. 增加稀疏检索与融合。
10. 增加 response builder 与文档生命周期操作。
11. 增加 MCP 或 HTTP 接口。
12. 增加 evaluation、dashboard、reranking 和多模态扩展。

## 测试策略

- 单元测试应覆盖核心类型、配置校验、fake adapters、dense 评分逻辑、fusion 逻辑和 response 构建。
- 集成测试应覆盖 ingest 后 query、collection 隔离、delete 后重查，以及 trace 生成。
- 端到端测试应覆盖 CLI ingest 命令、CLI query 命令，以及未来的 MCP 或 HTTP smoke test。

## 测试步骤

1. 按照本规格中的结构创建一个全新项目。
2. 在 `tests/fixtures/` 下添加一组小型 `.txt` 和 `.md` 语料。
3. 运行 `core` 契约与配置校验的单元测试。
4. 使用 fake adapters 运行 MVP 写入链路和读取链路的集成测试。
5. 执行 CLI ingest，对 fixture 语料进行摄取。
6. 执行 CLI query，并确认返回结果与摄取内容相关。
7. 验证 ingestion 和 query 都产生了 JSONL trace 文件。
8. 将 in-memory vector store 替换为持久化后端，并重新运行相同集成测试。
9. 增加稀疏检索，验证 dense-only 和 hybrid 两种模式都能通过确定性测试。
10. 验证以下边界情况：

- 空查询
- 不支持的文件类型
- 重复摄取同一文件
- collection 隔离
- 摄取后删除

## 验收标准

- 新项目可以按增量方式构建，而不是照搬原仓库结构。
- MVP 功能在无网络调用条件下可运行。
- 在替换 adapter 时，领域契约保持稳定。
- ingestion 与 retrieval 的编排逻辑位于接口层之外。
- CLI 是第一个完成的接口。
- 在开始做 dashboard 之前，trace 输出已经存在且可读。
- 系统具备清晰的后续扩展路径，可演进到 sparse retrieval、MCP 和 multimodal，而无需推翻核心契约。
