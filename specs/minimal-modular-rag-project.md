# 规格说明：最小化模块化 RAG 项目

## 概述

本规格定义了一个从本地可运行 MVP 演进到可扩展 RAG 基础工程的项目。

这个项目的定位不是“一次性做完所有 RAG 能力”，而是：

- 先做出可验证、可迭代的最小闭环
- 再逐步补齐 provider 抽象、hybrid retrieval、response contract
- 最后在此基础上接入 MCP / HTTP、rerank、evaluation、多模态 transform 等增强能力

- 已完成本地文本 ingest/query
- 已完成 dense-only 与 hybrid retrieval
- 已完成 provider abstraction 与本地持久化 vector store
- 已完成稳定的 response schema、citations、document lifecycle
- 已完成第一个协议接口层 MCP
- 尚未接入真实 LLM 生成、rerank、HTTP、多模态、dashboard/evaluation

### 产品目标

构建一个模块化、可观测、可测试、可持续扩展的 RAG 基础工程，使其能够：

- 摄取本地知识文档并建立稳定索引
- 通过应用层 API 返回结构化、可引用的检索结果
- 在不破坏核心契约的前提下逐步扩展到 MCP、生成、重排、评估和多模态

### 当前定位

这是一个 **RAG 基础工程**

当前重点在：

- 文档摄取
- 检索编排
- 输出契约
- 文档生命周期
- 可观测性
- 接口层薄封装

当前不包含：

- Planner
- Tool orchestration loop
- Multi-step agent reasoning
- Autonomous execution

### 非目标

在当前规格范围内，以下能力默认不作为 M4 前或 M4 当下的实现目标：

- PDF / Office / HTML loader
- 图像理解与多模态检索
- LLM answer synthesis
- Rerank
- Agent 工作流
- Dashboard UI
- 评估平台
- 分布式部署
- 外部网络依赖下的真实 provider 集成

## 设计原则

### 1. Incremental First

项目必须按里程碑递进，不允许一开始就引入 MCP、Rerank、多模态、Evaluation 等高复杂度模块。

### 2. Contracts Before Providers

优先定义稳定的核心类型和应用服务边界，再引入真实 provider。

### 3. Local-First Verification

每个阶段都必须能在本地、无外网依赖的条件下完成自动化验证。

### 4. Thin Interfaces

CLI、MCP、HTTP 都属于接口层，不承载业务逻辑，只做输入映射、调用服务、格式化输出。

### 5. Observable by Default

摄取与查询链路必须产出结构化 trace，便于后续接 dashboard、evaluation 和问题排查。

### 6. Graceful Expansion

新增 retrieval、response、interface、observability 能力时，不应破坏既有核心契约和已有测试。

## 需求

### 功能性需求

- FR-1: 项目应定义稳定的核心契约，包括 `Document`、`Chunk`、`ChunkRecord`、`ProcessedQuery`、`RetrievalResult`。
- FR-2: 项目应从 YAML 配置加载运行参数，并在启动时校验关键字段。
- FR-3: 项目应支持将本地 `.txt` 和 `.md` 文档摄取到命名 collection。
- FR-4: 摄取链路应至少支持 `load -> split -> embed -> store`。
- FR-5: 查询链路应至少支持 `query -> retrieve -> response`。
- FR-6: 查询链路应支持 `dense` 与 `hybrid` 两种 retrieval mode。
- FR-7: hybrid 模式应支持 `dense retrieve -> sparse retrieve -> RRF fuse`。
- FR-8: 查询输出应通过稳定的 response schema 返回，不应直接暴露裸检索结果作为接口契约。
- FR-9: response schema 应包含 citations，且 citation 至少包含 `chunk_id`、`doc_id`、`source_path`、`collection`、`chunk_index`、`score`。
- FR-10: 项目应支持至少两类文档生命周期操作：`list documents` 与 `delete document`。
- FR-11: 项目应把 ingest 与 query 的结构化 trace 写入 JSON Lines。
- FR-12: 在引入 MCP 或 HTTP 之前，CLI 必须保持可用，并作为默认验收入口。
- FR-13: application 层应只依赖抽象契约，不直接依赖具体 adapter 实现。
- FR-14: 项目应支持后续扩展到 rerank、MCP、HTTP、evaluation、多模态 transform，而不推翻现有分层。

### 非功能需求

- NFR-1: 架构必须清晰区分 `core`、`application`、`adapters`、`response`、`interfaces`、`observability`。
- NFR-2: MVP 与 M2-M4 测试必须可在无网络环境下通过。
- NFR-3: 纯编排逻辑应优先通过 unit tests 覆盖。
- NFR-4: integration tests 应覆盖跨实例、跨 service、跨存储的真实调用路径。
- NFR-5: e2e tests 应至少覆盖 CLI 主链路。
- NFR-6: 接口层不得包含 retrieval、response shaping、document lifecycle 业务逻辑。
- NFR-7: response contract 一旦稳定，应尽量作为 CLI、MCP、HTTP 的共享输出模型。
- NFR-8: trace 结构应足够稳定，使后续 dashboard / evaluation 可以直接消费。
- NFR-9: 当前本地持久化能力可以采用轻量实现，但职责边界必须清晰。
- NFR-10: 项目应通过 `uv`、`pytest`、`ruff` 在本地稳定运行。

## 架构

### 分层模型

- `core`
  - 领域类型、错误、配置、trace 契约、共享规则
- `application`
  - ingest、search、document lifecycle 的编排服务
- `adapters`
  - loader、embedding、vector store 等 provider 实现与 factory
- `retrieval`
  - dense、sparse、fusion、后续 rerank
- `response`
  - response schema、citation、输出整形
- `interfaces`
  - CLI 为先，后续扩展 MCP / HTTP
- `observability`
  - logger、trace store、后续 dashboard / evaluation

### 推荐项目结构

- `src/core/types.py`
- `src/core/settings.py`
- `src/core/errors.py`
- `src/core/trace.py`

- `src/application/ingest_service.py`
- `src/application/search_service.py`
- `src/application/document_service.py`

- `src/adapters/loader/base_loader.py`
- `src/adapters/loader/text_loader.py`
- `src/adapters/loader/factory.py`

- `src/adapters/embedding/base_embedding.py`
- `src/adapters/embedding/fake_embedding.py`
- `src/adapters/embedding/factory.py`

- `src/adapters/vector_store/base_vector_store.py`
- `src/adapters/vector_store/in_memory_store.py`
- `src/adapters/vector_store/local_json_store.py`
- `src/adapters/vector_store/factory.py`

- `src/retrieval/sparse_retriever.py`
- `src/retrieval/fusion.py`
- `src/retrieval/reranker.py`（后续）

- `src/response/response_builder.py`

- `src/interfaces/cli/ingest.py`
- `src/interfaces/cli/query.py`
- `src/interfaces/cli/documents.py`

- `src/interfaces/mcp/`（后续）
- `src/interfaces/http/`（后续，可选）

- `src/observability/logger.py`
- `src/observability/trace_store.py`
- `src/observability/dashboard/`（后续）
- `src/evaluation/`（后续）

- `config/settings.yaml.example`
- `tests/unit`
- `tests/integration`
- `tests/e2e`

### 当前推荐职责划分

#### Core

负责最稳定、最不应轻易变化的模型与规则：

- 类型定义
- 配置模型
- 错误类型
- trace context

#### Application

负责编排，不负责具体 provider 细节：

- `IngestService`
  - 文件发现
  - loader 调用
  - split 编排
  - embedding 编排
  - store 编排
  - ingestion trace

- `SearchService`
  - query normalization
  - retrieval mode 分派
  - dense / sparse / fusion 编排
  - response builder 调用
  - query trace

- `DocumentService`
  - list documents
  - delete document
  - 文档聚合视图

#### Adapters

负责和外部依赖或外部形态交互：

- loader
- embedding
- vector store
- 后续 LLM / reranker / evaluator provider

#### Retrieval

负责“怎么召回、怎么融合、怎么重排”的可替换检索算法逻辑。

#### Response

负责“如何把内部 retrieval 结果整形成稳定对外输出”。

#### Interfaces

负责命令行、协议适配、输入输出映射，不负责业务判断。

#### Observability

负责结构化 trace、日志输出，以及后续 dashboard / evaluation 的数据消费入口。

## 核心契约

### 核心类型

项目应至少包含以下稳定类型：

- `Document`
- `Chunk`
- `ChunkRecord`
- `ProcessedQuery`
- `RetrievalResult`

### Response 类型

在 M4 及之后，查询的正式输出契约应由 response 层定义，至少包含：

- `Citation`
- `SearchResultItem`
- `SearchOutput`

`SearchOutput` 至少应包含：

- `query`
- `normalized_query`
- `collection`
- `retrieval_mode`
- `result_count`
- `results`
- `citations`

### 文档生命周期类型

项目应至少包含：

- `DocumentSummary`
- `DeleteDocumentResult`

### 元数据约定

所有存储中的 chunk record 至少应包含：

- `source_path`
- `collection`
- `doc_id`
- `chunk_index`

建议保留的扩展字段：

- `start_offset`
- `end_offset`
- `title`
- `page`
- `rrf_sources`
- `rrf_source_ranks`

## 数据与存储约定

### ID 约定

- 文档 ID：确定性 ID，优先基于内容哈希
- Chunk ID：`{doc_id}_{chunk_index:04d}_{content_hash}`

### 本地目录布局

项目本地布局建议至少包含：

- `data/raw/`
- `data/db/`
- `data/traces/`

### 当前持久化边界

在当前项目里应明确区分两种 vector store 语义：

- `memory`
  - 纯内存
  - 适合 unit tests
  - 不负责进程间持久化

- `local_json`
  - 本地 JSON 快照持久化
  - 适合 CLI 多次运行之间共享状态
  - 不等同于生产级向量数据库

### Trace 约定

trace 类型固定为：

- `ingestion`
- `query`

query trace 在 M4 之后建议稳定记录如下 stage：

- `embed_query`
- `dense_retrieve`
- `sparse_retrieve`（hybrid 时）
- `rrf_fuse`（hybrid 时）
- 后续可增加 `rerank`
- 后续可增加 `build_response`

ingestion trace 建议稳定记录：

- `load`
- `split`
- `embed`
- `store`

## 当前已实现基线（M7）

### M1：仅 Dense 的本地 MVP

目标

- 先做出一个本地、无外部依赖、仅 dense 检索的最小闭环
- 优先验证 ingest/query 主链路，而不是提前抽象 provider 或扩展接口层
- 让核心类型、配置、trace、CLI 和集成测试先稳定下来
- 为后续 M2 的 provider abstraction 和 M3 的 hybrid retrieval 打基础

具体任务

1. 定义核心类型与基础规则：
   - src/core/types.py
   - src/core/errors.py
   - src/core/trace.py
     至少定义：
   - Document
   - Chunk
   - ChunkRecord
   - ProcessedQuery
   - RetrievalResult
     同时明确最小 metadata 约束：
   - source_path
   - collection
   - doc_id
   - chunk_index

2. 实现配置加载：
   - src/core/settings.py
   - config/settings.yaml.example
     配置至少覆盖：
   - default_collection
   - chunk_size
   - chunk_overlap
   - supported_extensions
   - dense_top_k
   - fake embedding dimensions
   - trace_file
     启动时必须校验关键字段，不能把明显错误拖到运行期。

3. 实现 loader：
   - src/adapters/loader/text_loader.py
     当前只支持：
   - .txt
   - .md
     loader 负责把文件读成统一的 Document，不负责切分。

4. 实现 embedding：
   - src/adapters/embedding/fake_embedding.py
     目标不是追求真实语义效果，而是：
   - 无外网依赖
   - 输出确定性
   - 适合单测和集成测试

5. 实现 vector store：
   - src/adapters/vector_store/in_memory_store.py
     当前只做：
   - upsert
   - dense query
     不要提前实现持久化或真实数据库接入。

6. 实现应用层主链路：
   - src/application/ingest_service.py
   - src/application/search_service.py
     ingest 至少完成：
   - 文件发现
   - load
   - split
   - embed
   - store
     search 至少完成：
   - query normalization
   - dense retrieve
   - top_k results

7. 实现 CLI：
   - src/interfaces/cli/ingest.py
   - src/interfaces/cli/query.py
     当前接口层只做：
   - 参数解析
   - 调用 service
   - 输出结果
     不要把业务逻辑写进 CLI。

8. 实现 trace：
   - src/observability/logger.py
   - src/observability/trace_store.py
     至少记录：
   - ingestion: load / split / embed / store
   - query: embed_query / dense_retrieve

9. 测试与验收：
   - tests/unit/\*
   - tests/integration/test_ingest_query_mvp.py
     单测先覆盖：
   - core types
   - settings
   - fake embedding
   - in_memory_store
   - 空查询错误
     集成测试至少覆盖：
   - ingest -> query -> trace

10. 文档：

- 更新 README
  明确说明当前只是：
- 本地文本 ingest/query MVP
- 仅 dense retrieval
- 无 provider abstraction
- 无 MCP / hybrid / rerank

约束

- 不要引入外网依赖、真实 embedding 服务、真实向量数据库
- 不要提前实现 provider abstraction，这一步的重点是先闭环
- 不要提前实现 MCP、HTTP、BM25、RRF、rerank、多模态
- 不要把 loader、embedding、store 设计成过度抽象的框架
- 接口层不得承载业务逻辑
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/pytest
- ./.venv/bin/ruff check .

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：
  - 做了什么
  - 为什么 M1 先不做 provider abstraction
  - 当前能力边界
  - 验证结果

已完成：

- 本地 `.txt` / `.md` 摄取
- 确定性切分
- fake embedding
- in-memory vector store
- CLI ingest / query
- JSONL traces

### M2：Provider 抽象与持久化边界整理

目标

- 保持现有 M1 行为、CLI 命令和测试语义不变
- application 层改为依赖抽象，不直接依赖具体 adapter
- 增加 factory，CLI 通过配置装配依赖
- 整理 vector store 的持久化边界，避免 “memory” 和 “本地文件快照” 语义混乱

具体任务

1. 新增抽象接口：
   - src/adapters/loader/base_loader.py
   - src/adapters/embedding/base_embedding.py
   - src/adapters/vector_store/base_vector_store.py
     抽象要覆盖当前 ingest/search/document_service 真实需要的方法，不要空泛设计。

2. 新增 factory：
   - src/adapters/loader/factory.py
   - src/adapters/embedding/factory.py
   - src/adapters/vector_store/factory.py
     当前先支持：
   - loader: text
   - embedding: fake
   - vector_store: memory（或你重命名后的更准确 provider 名）

3. 改造应用层和 CLI：
   - application 服务依赖抽象类型
   - interfaces/cli 只做参数解析、load_settings、factory 装配、结果输出
   - 保持 mrag-ingest / mrag-query / mrag-docs 的外部行为不变

4. 整理 vector store 设计：
   - 评估当前 InMemoryVectorStore 是否应该拆成“纯内存实现”和“文件持久化包装”
   - 如果值得拆，就拆并补测试
   - 如果不拆，也要统一命名和 README 说明，让职责边界清晰

5. 测试与验收：
   - 补 unit tests：base contract / factory / service decoupling
   - 补 integration tests：通过 factory 组装后 ingest/query/document lifecycle 仍正常
   - 不要删除已有有效测试；若发现测试假设有问题，修测试并说明原因

6. 文档：
   - 更新 README，说明当前 provider abstraction 状态、支持的 provider、下一步扩展点

约束

- 不要引入外网依赖、真实 embedding 服务、数据库服务
- 不要提前实现 MCP、BM25、RRF、rerank、多模态
- 接口层不得承载业务逻辑
- 优先小步重构，避免无意义的过度抽象
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/pytest
- ./.venv/bin/ruff check .

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：做了什么、为什么这样拆、风险点、验证结果

已完成：

- `BaseLoader`
- `BaseEmbedding`
- `BaseVectorStore`
- factory 装配
- `InMemoryVectorStore` 与 `LocalJsonVectorStore` 职责拆分
- application 层仅依赖抽象

### M3：稀疏检索与融合

目标

- 在不破坏 M2 抽象边界的前提下，把系统从单路 dense retrieval 升级为 dense + sparse 的 hybrid retrieval
- 保持 dense-only 模式继续可用
- 先把 sparse retrieval 与 fusion 行为做对，不提前引入 rerank 或真实外部检索组件
- 为后续 M4 的稳定 response contract 和 M6 的 rerank 打基础

具体任务

1. 明确 retrieval 边界：
   - 在现有 application / adapters 基础上新增 retrieval 层
   - 不把 sparse 检索逻辑塞回 vector store
   - 不把 fusion 逻辑塞进 CLI

2. 新增 sparse retriever：
   - src/retrieval/sparse_retriever.py
     当前实现要求：
   - 基于现有 chunk records 构建轻量 sparse index
   - 支持关键词级召回
   - 输出仍然统一为 RetrievalResult
     不要求这一步就做高性能持久化 sparse index。

3. 新增融合模块：
   - src/retrieval/fusion.py
     第一版固定采用：
   - RRF
     融合结果必须具备确定性，且排序行为可测试。

4. 改造 SearchService：
   - src/application/search_service.py
     当前至少支持两种 retrieval mode：
   - dense
   - hybrid
     hybrid 路径至少完成：
   - dense retrieve
   - sparse retrieve
   - RRF fuse
     同时保持 dense-only 行为与已有 CLI 兼容。

5. 扩展配置：
   - src/core/settings.py
   - config/settings.yaml.example
     至少增加：
   - retrieval.mode
   - retrieval.sparse_top_k
   - retrieval.rrf_k
     配置必须在启动时校验，不要把非法 retrieval mode 留到运行中再暴露。

6. 改造 CLI query：
   - src/interfaces/cli/query.py
     至少支持：
   - 默认读取配置中的 retrieval mode
   - 通过参数覆盖 mode
     CLI 仍然只做参数解析和输出，不复制 retrieval 逻辑。

7. trace 扩展：
   - query trace 需要增加：
   - sparse_retrieve
   - rrf_fuse
     这样后续才能区分 dense 与 hybrid 的真实执行过程。

8. 测试：
   - tests/unit/test_fusion.py
   - tests/integration/test_hybrid_search.py
     单测至少覆盖：
   - RRF 融合行为确定性
   - 单路与双路输入的排序稳定性
     集成测试至少覆盖：
   - hybrid 查询真的走了 sparse + fusion
   - dense mode override 仍然可用

9. 文档：
   - 更新 README
     明确说明当前已支持：
   - dense-only
   - hybrid retrieval
     同时说明当前 sparse 实现仍是轻量本地实现，不是生产级方案。

约束

- 不要引入 rerank、LLM answer synthesis、MCP、HTTP、多模态
- 不要把 sparse index 做成重量级新存储系统
- 不要为了 M3 回头破坏 M2 的抽象边界
- 不要让 CLI 或 response 层承担 retrieval 逻辑
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/pytest
- ./.venv/bin/ruff check .

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：
  - 做了什么
  - dense 与 hybrid 两条路径如何共存
  - 当前 sparse 实现的边界
  - 验证结果

已完成：

- `SparseRetriever`
- RRF fusion
- dense-only / hybrid retrieval mode
- hybrid integration tests

### M4：Response 层与文档生命周期

目标

- 增加 response 层，提供稳定的查询输出结构
- 增加 citations
- 保持现有 document list/delete 能力，但让其输出契约继续稳定
- 不破坏当前 dense / hybrid 查询能力
- 不引入真实 LLM 生成，不提前做 M5/M6 范围

具体任务

1. 新增 response 层核心文件：
   - src/response/response_builder.py
     这个文件里至少要有：
   - Citation
   - SearchResultItem
   - SearchOutput
   - ResponseBuilder
     结构要尽量简洁稳定，面向后续接口复用，不要过度设计。

2. 查询输出标准化：
   - ResponseBuilder 输入应至少包含：
     - ProcessedQuery
     - list[RetrievalResult]
     - retrieval_mode
   - 输出 SearchOutput 应至少包含：
     - query
     - normalized_query
     - collection
     - retrieval_mode
     - result_count
     - results
     - citations
   - Citation 至少包含：
     - chunk_id
     - doc_id
     - source_path
     - collection
     - chunk_index
     - score

3. 改造 SearchService：
   - 不再把“裸 results”视为最终对外输出
   - SearchService 内部接入 ResponseBuilder
   - 保持 dense / hybrid 两种模式行为不变
   - trace 保持现有逻辑，不要削弱

4. 改造 CLI query：
   - interfaces/cli/query.py 只消费稳定的 SearchOutput
   - CLI 可以继续打印文本，但不要再直接依赖原始 RetrievalResult 列表结构拼输出
   - 外部命令行为尽量保持兼容

5. DocumentService：
   - 保持现有 list/delete 行为
   - 如果需要，只做轻量整理，让 list/delete 输出契约更清晰
   - 不要在这一步引入 document detail / stats / reindex 等超出 M4 范围的能力

6. 测试：
   - 新增 tests/unit/test_response_builder.py
   - 覆盖：
     - dense 模式 response 输出
     - hybrid 模式 response 输出
     - citations 生成
     - 空结果时的稳定输出
   - 如果需要，补一个集成测试验证 search -> response builder 的真实链路
   - 不要删除已有有效测试

7. 文档：
   - 更新 README
   - 说明当前项目已经进入 M4
   - 说明查询输出结构已经稳定，并且后续可供 CLI / MCP / HTTP 复用

约束

- 不要引入真实 LLM、summary、answer synthesis
- 不要提前做 rerank、MCP、HTTP、多模态
- 接口层不得承载业务逻辑
- 避免无意义的过度抽象
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/ruff check .
- ./.venv/bin/pytest

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：
  - 做了什么
  - 为什么这样设计 response 层
  - 当前 residual risks
  - 验证结果

### M5：接口层扩展

目标

- 在不修改 application 业务逻辑的前提下，引入第一个协议接口
- 保留现有 CLI，并验证 CLI / MCP 能复用同一套 application services
- 复用现有 SearchOutput 与 document lifecycle contract，而不是在接口层重新定义业务结果
- 保持本地无外网依赖、测试可通过

具体任务

1. 明确 M5 范围：
   - 保留 CLI
   - 新增 MCP 接口层
   - 复用现有 application services
   - 复用现有 response/document contracts
   - 增加协议层测试
   - 本轮不做：
   - HTTP
   - rerank
   - LLM answer synthesis
   - 多模态
   - dashboard
   - evaluation
   - 官方 MCP SDK stdio transport 完整接入

2. 明确复用边界：
   - MCP 层不能直接做 retrieval
   - MCP 层不能自己拼业务 response
   - MCP 层不能复制 document lifecycle 逻辑
   - MCP 层只能做：
   - 参数校验
   - 依赖装配调用
   - 结果映射
   - 错误映射
   - 业务核心必须继续停留在：
   - src/application/search_service.py
   - src/application/document_service.py
   - src/response/response_builder.py

3. 新增 MCP 目录骨架：
   - src/interfaces/mcp/
   - src/interfaces/mcp/models.py
   - src/interfaces/mcp/dependencies.py
   - src/interfaces/mcp/mappers.py
   - src/interfaces/mcp/server.py
   - src/interfaces/mcp/tools/
   - src/interfaces/mcp/tools/query_knowledge.py
   - src/interfaces/mcp/tools/list_documents.py
   - src/interfaces/mcp/tools/delete_document.py

4. 设计 MCP 协议层最小数据模型：
   - 在 src/interfaces/mcp/models.py 中定义：
   - MCPTextContent
   - MCPTool
   - MCPToolResult
   - 要求：
   - 可以稳定 to_dict()
   - 不依赖外部 mcp SDK
   - 适合本地 smoke test

5. 统一依赖装配：
   - 在 src/interfaces/mcp/dependencies.py 中统一创建：
   - settings
   - TraceStore
   - embedding
   - vector_store
   - SearchService
   - DocumentService
   - 实现原则：
   - 只在这里调用 load_settings()
   - 只在这里调用 factory
   - 只在这里决定 trace 是否启用
   - tool 文件里不能重复写这些逻辑
   - 最终暴露 MCPDependencies 容器对象

6. 设计共享映射层：
   - 在 src/interfaces/mcp/mappers.py 中至少实现：
   - map_search_output(output: SearchOutput) -> MCPToolResult
   - map_document_list(documents: list[DocumentSummary]) -> MCPToolResult
   - map_delete_result(result: DeleteDocumentResult) -> MCPToolResult
   - map_error(message: str, code: str) -> MCPToolResult
   - 设计要求：
   - 同时输出人类可读内容 content
   - 同时输出机器可消费结构 structured_content
   - structured_content 必须保留稳定字段
   - SearchOutput 的 citations 必须完整透传
   - 不允许在 tool 文件里散落结果序列化逻辑

7. 设计 query MCP tool：
   - 在 src/interfaces/mcp/tools/query_knowledge.py 中实现 query_knowledge
   - 输入至少支持：
   - query
   - collection
   - top_k
   - mode
   - 处理流程固定为：
   - 校验参数
   - 调用 SearchService.search(...)
   - 将 SearchOutput 交给 map_search_output()
   - 返回 MCPToolResult
   - 禁止：
   - 在 tool 内做 dense / sparse / fusion
   - 在 tool 内自己拼 citation
   - 在 tool 内绕过 SearchService

8. 设计 list_documents MCP tool：
   - 在 src/interfaces/mcp/tools/list_documents.py 中实现 list_documents
   - 输入至少支持：
   - collection
   - 处理流程：
   - 校验参数
   - 调用 DocumentService.list_documents(...)
   - 将结果交给 map_document_list()
   - 返回 MCPToolResult
   - 输出至少包含：
   - doc_id
   - source_path
   - collection
   - chunk_count

9. 设计 delete_document MCP tool：
   - 在 src/interfaces/mcp/tools/delete_document.py 中实现 delete_document
   - 输入至少支持：
   - doc_id
   - collection
   - 处理流程：
   - 校验参数
   - 调用 DocumentService.delete_document(...)
   - 将结果交给 map_delete_result()
   - 返回 MCPToolResult
   - 输出至少包含：
   - doc_id
   - collection
   - deleted_chunks
   - deleted

10. 设计 MCP server：

- 在 src/interfaces/mcp/server.py 中实现本地 MCP server
- 至少包含：
- MCPServer
- create_mcp_server(config_path)
- register_tool(...)
- list_tools()
- call_tool(name, arguments)
- 职责：
- 注册 tools
- 保存 tool schema
- 分发调用
- 捕获异常
- 做统一错误映射
- 错误处理至少覆盖：
- 未知 tool
- 参数错误
- 配置错误
- query 为空
- retrieval mode 非法
- 其他内部错误
- 要求：
- 不泄露 traceback 到协议输出
- 只返回稳定的 MCP error payload

11. 提供本地 smoke CLI：

- 在 src/interfaces/mcp/server.py 中提供最小 CLI 入口
- 支持：
- list-tools
- call-tool
- 在 pyproject.toml 中注册：
- mrag-mcp = "src.interfaces.mcp.server:main"
- 这样可以直接本地验证：
- uv run mrag-mcp list-tools
- uv run mrag-mcp call-tool query_knowledge --arguments-json '{"query":"semantic embeddings","collection":"knowledge"}'

12. 单元测试清单：

- tests/unit/test_mcp_mappers.py
- tests/unit/test_query_tool.py
- tests/unit/test_list_documents_tool.py
- tests/unit/test_delete_document_tool.py
- 要覆盖：
- SearchOutput 映射后仍保留 citations
- document list 映射结构稳定
- delete result 映射结构稳定
- query tool 正确调用 SearchService
- list/delete tool 正确调用 DocumentService
- 参数错误能抛出或被捕获成协议错误

13. 集成测试清单：

- tests/integration/test_mcp_server.py
- 至少覆盖：
- 先 ingest 文档
- 再通过 create_mcp_server() 调用 query_knowledge
- 验证返回中有 SearchOutput 结构和 citations
- 再调用 list_documents
- 再调用 delete_document
- 最后验证 missing_tool 会返回错误 payload
- 这一步的意义是验证：
- MCP 层确实复用了 application services，而不是另起一套逻辑

14. E2E 测试清单：

- tests/e2e/test_mcp_smoke.py
- 至少覆盖：
- mrag-mcp list-tools
- mrag-mcp call-tool query_knowledge
- 目的不是验证完整协议栈，而是验证：
- CLI 入口可用
- tool registry 可用
- structured payload 可用

15. 文档：

- 更新 README
- 至少补这些内容：
- 当前项目已进入 M5
- 当前已支持 CLI + MCP
- MCP 层复用：
- SearchService
- DocumentService
- SearchOutput
- 新增命令：
- mrag-mcp
- 当前 MCP 边界：
- 本地 in-process tool server
- 不是官方 MCP SDK 的 stdio transport

16. M5 验收标准：

- CLI 继续工作
- MCP 至少支持：
- query
- list documents
- delete document
- SearchService 无需为 MCP 重写
- DocumentService 无需为 MCP 重写
- SearchOutput 作为共享契约被复用
- MCP 层只是协议映射层，不复制 retrieval / response / lifecycle 逻辑
- ruff 通过
- pytest 通过

17. M5 明确不做的项：

- 不做 HTTP
- 不做官方 MCP SDK stdio transport 全实现
- 不做 rerank
- 不做 LLM 生成
- 不做多模态
- 不做 dashboard
- 不做 evaluation
- 不做 agent loop

18. M5 完成后的残余风险：

- 当前 MCP 只是本地协议适配，不是正式 transport 层
- 参数校验是轻量级的，不是完整 JSON Schema engine
- 错误码还不是协议级完整体系
- response contract 虽然能复用，但还不是生成式 answer contract

约束

- 不要引入 HTTP 作为本轮主线
- 不要提前做 rerank、LLM answer synthesis、多模态、dashboard、evaluation
- 不要把业务逻辑塞进 MCP 层
- 不要破坏现有 CLI 行为
- 尽量复用现有 SearchOutput / DocumentService 契约
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/ruff check .
- ./.venv/bin/pytest

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：
  - 做了什么
  - MCP 层如何复用现有 application / response contract
  - 当前 residual risks
  - 验证结果

已完成：

- `src/interfaces/mcp/`
- `MCPDependencies`
- `MCPServer`
- `query_knowledge`
- `list_documents`
- `delete_document`
- 共享 MCP mapper
- `mrag-mcp`
- MCP unit / integration / e2e tests

## 已完成里程碑与后续计划

### M6：生成与精排层

目标

- 在当前 retrieval + response contract 基础上，补齐“G”与“精排”
- 把项目从“检索型 RAG 基础工程”推进到“带生成与重排的 RAG”
- 保持 dense / hybrid retrieval 两条路径继续可用
- 保持 citations 与输出契约的可追溯性
- 不在这一阶段引入复杂 Agent loop 或外部真实服务依赖

具体任务

1. 明确 M6 范围：
   - 保留现有 retrieval 能力
   - 新增 rerank
   - 新增 answer synthesis
   - 新增答案级输出契约
   - 保留现有 SearchOutput 不被强行改造成生成输出
   - 本轮不做：
   - 多模态 answer synthesis
   - Agent loop
   - dashboard
   - evaluation
   - 真实外部 LLM provider

2. 新增 reranker 模块：
   - src/retrieval/reranker.py
   - 或补充适配器层抽象：
   - src/adapters/reranker/base_reranker.py
   - src/adapters/reranker/fake_reranker.py
   - src/adapters/reranker/factory.py
   - 第一版要求：
   - 输出确定性
   - 可在本地无网络运行
   - 能对候选 RetrievalResult 重新排序
   - 不要求第一版就接真实 cross-encoder 或 LLM reranker

3. 新增 LLM 抽象：
   - src/adapters/llm/base_llm.py
   - src/adapters/llm/fake_llm.py
   - src/adapters/llm/factory.py
   - 第一版只需要 fake / stub 实现
   - 重点是先打通接口和测试，不是追求真实答案质量

4. 新增答案输出层：
   - src/response/answer_builder.py
   - 至少定义：
   - AnswerCitation
   - AnswerOutput
   - AnswerBuilder
   - AnswerOutput 至少应包含：
   - query
   - normalized_query
   - collection
   - retrieval_mode
   - answer
   - citations
   - supporting_results
   - 不要直接复用 SearchOutput 充当最终生成输出

5. 新增应用层服务：
   - 推荐新增：
   - src/application/answer_service.py
   - AnswerService 至少负责：
   - 调用 SearchService 获取候选结果
   - 调用 Reranker 做精排
   - 调用 LLM 生成答案
   - 调用 AnswerBuilder 构建最终输出
   - 不要让 SearchService 承担完整生成职责，避免职责膨胀

6. 明确查询与生成链路：
   - 当前至少支持：
   - retrieve -> rerank -> build answer
   - dense 与 hybrid 两种 retrieval mode 都必须能进入该链路
   - rerank 只处理候选集，不改变原始检索能力边界
   - citations 必须仍然能回溯到 supporting_results

7. CLI 设计：
   - 推荐新增独立入口：
   - src/interfaces/cli/answer.py
   - 在 pyproject.toml 注册：
   - mrag-answer = "src.interfaces.cli.answer:main"
   - 不推荐把 retrieval-only 与 answer flow 混在一个 CLI 命令的默认行为里
   - CLI 仍然只做参数解析、调用 service、打印输出

8. 配置扩展：
   - src/core/settings.py
   - config/settings.yaml.example
   - 至少增加：
   - llm.provider
   - reranker.provider
   - generation.max_context_results
   - generation.max_answer_chars
   - provider 当前默认仍然只支持 fake / stub

9. trace 扩展：
   - query / answer trace 至少增加：
   - rerank
   - assemble_context 或 build_prompt
   - generate_answer
   - 需要能区分 retrieval-only 和 answer flow
   - 可选择新增 answer trace type，或在 query trace metadata 中明确 mode

10. 测试：

- tests/unit/test_reranker.py
- tests/unit/test_fake_llm.py
- tests/unit/test_answer_builder.py
- tests/unit/test_answer_service.py
- tests/integration/test_query_with_generation.py
- 至少覆盖：
- dense + answer
- hybrid + answer
- citations 能追溯到 supporting_results
- 空结果时返回稳定空答案输出
- fake llm / fake reranker 无外网依赖

11. 文档：

- 更新 README
- 明确项目进入“带生成的 RAG”阶段
- 说明 SearchOutput 与 AnswerOutput 的区别
- 说明当前 generation 仍然是 fake / stub 级实现

12. M6 验收标准：

- 支持 retrieve -> rerank -> build answer
- dense / hybrid 两条路径都能生成答案
- citations 仍然可追溯
- 不破坏现有 mrag-query
- ruff 通过
- pytest 通过

约束

- 不要引入真实外部 LLM provider
- 不要引入官方 MCP answer tool 扩展
- 不要引入多模态 answer synthesis
- 不要引入 Agent tool loop
- 不要把生成逻辑塞进 SearchService 或接口层
- 保持 SearchOutput 作为 retrieval-level contract 的稳定性
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/ruff check .
- ./.venv/bin/pytest

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：
  - 做了什么
  - rerank 与 generation 如何接入现有 retrieval / response contract
  - 当前 residual risks
  - 验证结果

说明：

- M6 的目标是把当前“检索型 RAG 基础工程”推进到“真正带生成的 RAG”
- 不应在这一阶段引入复杂 Agent loop

### M7：Observability 与 Evaluation

目标

- 让 trace 从“能记录”升级到“能分析”
- 补齐 retrieval / answer 的基础评估能力
- 为后续持续调优建立数据闭环
- 在不改变当前 retrieval、answer、MCP 主链路职责的前提下新增可观测与回归能力

具体任务

1. 明确 M7 范围：
   - 重点做：
   - trace reader
   - trace explorer 或 dashboard 的最小文本入口
   - retrieval regression
   - answer regression
   - golden fixtures
   - 本轮不做：
   - 重量级 Web dashboard
   - 真实 LLM-as-judge
   - 新的 retrieval 算法
   - 多模态
   - Agent loop

2. 新增 trace reader：
   - src/observability/trace_reader.py
   - 至少支持：
   - 读取 JSONL trace
   - 按 trace_type 过滤
   - 按 trace_id 查看单次链路
   - 汇总 stage 耗时
   - 统计 result_count / answer_chars / 错误与空结果情况
   - 需要兼容：
   - ingestion
   - query
   - answer

3. 新增 trace explorer 或最小 dashboard 入口：
   - 推荐新增：
   - src/interfaces/cli/traces.py
   - 或：
   - src/observability/dashboard/
   - 至少提供：
   - mrag-traces list
   - mrag-traces show <trace_id>
   - mrag-traces stats
   - 输出可以先是文本或 JSON，不要求本轮做前端页面

4. 新增 evaluation 目录：
   - src/evaluation/
   - 至少拆分：
   - src/evaluation/fixtures.py
   - src/evaluation/retrieval_eval.py
   - src/evaluation/answer_eval.py
   - 如有必要可增加：
   - src/evaluation/models.py
   - 重点是先稳定 fixture 与 runner 契约

5. 定义 golden fixtures：
   - tests/fixtures/evaluation/
   - 或 data/evaluation/
   - retrieval fixture 至少包含：
   - query
   - collection
   - expected_doc_ids 或 expected_chunk_ids
   - top_k
   - answer fixture 至少包含：
   - query
   - collection
   - expected_keywords
   - expected_source_paths
   - 第一版 fixture 应小而稳定，适合本地回归

6. 实现 retrieval regression：
   - retrieval_eval.py 至少支持：
   - 读取 retrieval fixtures
   - 调用 SearchService
   - 计算 hit@k
   - 计算 recall@k
   - 可选增加 MRR 或简单 rank score
   - 输出应既可人读，也可程序消费
   - 不要把评估逻辑塞进 SearchService

7. 实现 answer regression：
   - answer_eval.py 至少支持：
   - 读取 answer fixtures
   - 调用 AnswerService
   - 检查 answer 非空
   - 检查 answer 命中 expected_keywords
   - 检查 citations / supporting_results / source_path 是否合理
   - 第一版不依赖外部 judge，优先 deterministic checks

8. 新增 evaluation CLI：
   - 推荐新增：
   - src/interfaces/cli/eval.py
   - 在 pyproject.toml 注册：
   - mrag-eval = "src.interfaces.cli.eval:main"
   - 至少支持：
   - mrag-eval retrieval
   - mrag-eval answer
   - mrag-eval all
   - CLI 只负责装配与输出，不承载指标计算逻辑

9. 配置与目录整理：
   - 如有必要，在 src/core/settings.py 补充 evaluation 路径配置
   - 若不新增配置，也至少在 README 与 fixture 目录中明确默认位置
   - trace 与 evaluation 的输入输出路径需要足够清晰，便于本地重复执行

10. 测试：

- tests/unit/test_trace_reader.py
- tests/unit/test_retrieval_eval.py
- tests/unit/test_answer_eval.py
- tests/integration/test_trace_explorer.py
- tests/integration/test_eval_regression.py
- 至少覆盖：
- trace 读取与过滤正确
- fixture 解析正确
- retrieval metrics 计算稳定
- answer regression 结果稳定
- CLI 能输出固定 summary

11. 文档：

- 更新 README
- 说明项目进入 M7：Observability 与 Evaluation
- 说明当前已支持：
- trace exploration
- retrieval regression
- answer regression
- 说明当前边界：
- 仍然不是在线 dashboard 平台
- 仍然不是复杂模型评测框架

12. M7 验收标准：

- 可以读取并分析 ingestion / query / answer traces
- 可以对 retrieval 质量做回归比较
- 可以对 answer 输出做基础回归比较
- fixture 格式稳定
- CLI 可运行
- 不改坏现有 mrag-query / mrag-answer / mrag-mcp
- ruff 通过
- pytest 通过

约束

- 不要引入重量级前端 dashboard
- 不要引入真实外部 LLM judge 或复杂在线评估平台
- 不要在这一阶段继续追加 retrieval / generation 新算法
- 不要把 evaluation 逻辑塞进 SearchService、AnswerService 或接口层
- trace reader 与 eval runner 必须建立在现有稳定 trace / response contract 之上
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/ruff check .
- ./.venv/bin/pytest

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：
  - 做了什么
  - trace reader / eval runner 如何复用当前 trace 与 response contract
  - 当前 residual risks
  - 验证结果

说明：

- evaluation 与 dashboard 应建立在稳定 trace 和 response contract 之上，而不是反过来牵引核心服务设计
- M7 的重点不是“继续加新功能”，而是让现有 retrieval / answer 能被持续观测、比较、回归验证

### 里程碑 M8：多模态与增强型 Ingestion

目标

- 在不破坏当前 retrieval、answer、document lifecycle 契约的前提下扩展 ingest 能力
- 引入 PDF loader 与可插拔 transform pipeline
- 让文档在进入 chunk / vector store 之前支持 metadata enrichment、页级拆分和轻量内容增强
- 为后续多模态处理与更复杂 ingestion workflow 预留稳定扩展点

具体任务

1. 明确 M8 范围：
   - 重点做：
   - PDF loader
   - ingestion pipeline
   - transform plugins
   - 页码级 metadata
   - 可选的 image captioning / chunk refinement 扩展点
   - 本轮不做：
   - 完整 OCR 平台
   - 真正的多模态检索
   - 外部图像大模型依赖
   - dashboard / evaluation 增强
   - Agent loop

2. 新增 PDF loader：
   - src/adapters/loader/pdf_loader.py
   - 如有必要补充：
   - src/adapters/loader/factory.py
   - 第一版至少支持：
   - 从 PDF 提取文本
   - 返回统一 Document 或页级中间结构
   - 为后续 metadata 保留：
   - source_path
   - page
   - title（如可得）
   - 不要求第一版就支持复杂版面分析

3. 定义 ingestion pipeline：
   - src/ingestion/pipeline.py
   - pipeline 负责：
   - 调用 loader
   - 运行 transforms
   - 生成最终可切分文本单元
   - 与现有 IngestService 对接时，应避免把 transform 逻辑硬塞进 service 主体
   - 推荐把 pipeline 作为 IngestService 的下游编排助手，而不是重新定义 ingest 主链

4. 新增 transform 目录：
   - src/ingestion/transforms/
   - 至少定义：
   - base transform contract
   - metadata enrichment transform
   - chunk refinement transform
   - 如需占位，可增加：
   - image captioning transform（stub / fake）
   - 重点是先把插件接口拉直，而不是一开始就引入重量级实现

5. 明确 transform contract：
   - 每个 transform 至少应支持：
   - 输入统一文档单元
   - 输出统一文档单元
   - 可附加 metadata
   - 可跳过或 no-op
   - contract 应保证 transform 可组合、可测试、可独立启停

6. 页级与来源级 metadata 支持：
   - 至少在 metadata 中保留：
   - source_path
   - collection
   - doc_id
   - page
   - chunk_index
   - 如 transform 增加内容增强，可附加：
   - section_title
   - caption
   - refinement_applied
   - 目标是让 citations 后续仍然能准确回溯

7. 与现有 ingest 链路集成：
   - src/application/ingest_service.py
   - 现有 ingest 主链仍应保持：
   - discover -> load -> split -> embed -> store
   - M8 应把增强步骤插入到：
   - load 与 split 之间
   - 或在 loader 输出与 chunking 之间
   - 不要改坏现有文本文件 ingest 行为

8. 配置扩展：
   - src/core/settings.py
   - config/settings.yaml.example
   - 至少增加：
   - loader.provider = text / pdf
   - ingestion.transforms.enabled
   - ingestion.transforms.order
   - 对每个 transform 的开关或基础参数
   - 配置必须支持：
   - 关闭所有 transform 时，仍然退化为当前 M7 的纯文本 ingest 行为

9. 新增测试夹具与样例文档：
   - tests/fixtures/ingestion/
   - 至少包含：
   - 一个 PDF 样例
   - 一个多页 PDF 样例
   - 一个 transform 前后可对比的文本样例
   - 样例应尽量小、稳定、适合本地仓库长期保留

10. 测试：

- tests/unit/test_pdf_loader.py
- tests/unit/test_ingestion_pipeline.py
- tests/unit/test_transforms.py
- tests/integration/test_pdf_ingestion.py
- tests/integration/test_transform_pipeline.py
- 至少覆盖：
- PDF 可成功读取
- transform 顺序可控
- transform 可 no-op
- metadata 在 pipeline 中不丢失
- 页码级 citations 可回溯
- 纯文本 ingest 行为未回归

11. 文档：

- 更新 README
- 说明项目进入 M8：多模态与增强型 Ingestion
- 说明当前支持：
- PDF loader
- transform pipeline
- metadata enrichment
- 说明当前边界：
- 仍不是完整多模态检索系统
- image captioning 可以先是 stub / fake

12. M8 验收标准：

- 支持 PDF ingest
- 支持至少一个可配置 transform pipeline
- 支持页级或来源级 metadata 回溯
- 不改坏现有 txt / md ingest
- SearchService / DocumentService / AnswerService 的契约不需要被推翻
- citations 仍可定位来源
- ruff 通过
- pytest 通过

约束

- 不要一开始就引入复杂 OCR、版面分析或外部多模态大模型依赖
- 不要把 transform 逻辑塞进 SearchService、AnswerService 或接口层
- 不要在这一阶段扩展多模态检索，而应专注于 ingestion 扩展点
- transform pipeline 必须在关闭时能退化为当前纯文本 ingest 行为
- 如果执行中断了，就重新审视当前仓库状态后继续完成，不要停在半成品分析

完成前必须执行完整验证

- ./.venv/bin/ruff check .
- ./.venv/bin/pytest

输出要求

- 直接改代码，不要只给方案
- 最终说明包括：
  - 做了什么
  - PDF loader / transform pipeline 如何接入现有 ingest 契约
  - 当前 residual risks
  - 验证结果

说明：

- 这一阶段借鉴参考项目的 transform 思路，但不要求一开始就引入完整图像处理链路
- M8 的重点是先定义 loader 与 transform 的扩展边界，再逐步把更多多模态处理能力接进去

### 里程碑 M9：Agent-Ready 扩展层

范围：

- 在 RAG 工程稳定后，提供可供 Agent 调用的上层能力
- 明确“RAG 基础层”和“Agent orchestration 层”的边界

主要交付物：

- `src/agent/` 或 `src/workflows/`
- tool registry
- query / document / answer 组合工具
- stateful workflow stubs
- 端到端示例

验收意图：

- Agent 使用现有 RAG 服务，而不是重新实现检索流程
- Agent 层可以被删除而不影响 RAG 核心能力
- RAG 工程仍可独立运行、独立测试

## 构建顺序

推荐构建顺序固定如下：

1. 核心类型、配置、错误、trace
2. ingest 与 dense retrieval MVP
3. provider abstraction 与持久化边界
4. sparse retrieval 与 fusion
5. response contract 与 document lifecycle
6. MCP / HTTP 接口
7. rerank 与 answer synthesis
8. evaluation / dashboard
9. 多模态与 transform pipeline
10. agent-ready 扩展

禁止顺序：

- 在 M1 前引入 MCP / HTTP
- 在 M2 前引入真实 provider 集成
- 在 M4 前让接口层自定义不稳定 response
- 在 M6 前把项目描述成完整生成式 RAG
- 在 RAG 主链路未稳定前直接转做 Agent

## 测试策略

### Unit Tests

覆盖：

- core types
- settings validation
- fake embedding
- vector store contract
- sparse retriever
- fusion
- response builder
- service abstraction

### Integration Tests

覆盖：

- ingest -> query
- cross-instance local_json store
- dense / hybrid retrieval
- document lifecycle
- search -> response builder 真实链路

### E2E Tests

覆盖：

- CLI ingest
- CLI query
- CLI docs list
- CLI docs delete

### 验证命令

每个阶段完成前必须至少执行：

```bash
./.venv/bin/ruff check .
./.venv/bin/pytest
```
