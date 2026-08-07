# AI 小镇

这是一个由同一大模型驱动三个 Agent 的双层人格手工实验系统。当前唯一业务
闭环是：

> 外部事件排队 → 生成并确认内层 → 生成并确认外层 →
> 把外层输出路由为接收者的新事件

每个 Agent 的内层与外层各有独立 system prompt 和完整已确认历史。系统不会
自动推进、自动记忆、压缩上下文或建立供 Agent 使用的公共时间线。完整产品
假设与信息边界以 [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) 为准。

## 环境要求

- Python `>=3.14,<3.15`
- [uv](https://docs.astral.sh/uv/)
- Node.js `>=24,<25`
- npm（随 Node.js 安装）

根目录的 `.python-version` 和 `.node-version` 声明了项目使用的主版本。

## 模型配置

根目录的 `models.toml` 是必需的本地配置并已被 Git 忽略。先从提交的示例复制：

Linux Bash：

```bash
cp models.example.toml models.toml
```

Windows PowerShell：

```powershell
Copy-Item models.example.toml models.toml
```

每个 `[[models]]` 严格包含四个字段：

```toml
[[models]]
model = "anthropic/claude-haiku-4.5"
protocol = "anthropic_messages"
base_url = "https://api.example.com/anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[[models]]
model = "deepseek-v4-flash"
protocol = "deepseek_responses"
base_url = "https://api.example.com/deepseek"
api_key_env = "DEEPSEEK_API_KEY"

[[models]]
model = "MiniMax-M3"
protocol = "minimax_responses"
base_url = "https://api.example.com/minimax"
api_key_env = "MINIMAX_API_KEY"
```

请把占位 URL 和模型名替换为兼容端点的实际值。`model` 大小写敏感且全局
唯一；TOML 顺序就是界面顺序，可以配置任意数量模型，也可以让多个模型使用
同一 protocol。当前正式内部 key 为 `anthropic_messages`、`deepseek_responses`
和 `minimax_responses`。protocol、端点和密钥不会出现在模型选择、scene 数据或
`GET /api/model-options` 中。

`api_key_env` 只引用环境变量名，真实 key 不得写进 `models.toml`。API key 按
模型粒度存储：每个 `[[models]]` 通过自己的 `api_key_env` 引用独立的环境变量，
多个模型可以复用同一个变量。可直接设置
进程环境变量，也可选择复制 [`.env.example`](.env.example) 为 `.env` 并填写：

```dotenv
ANTHROPIC_API_KEY=""
DEEPSEEK_API_KEY=""
MINIMAX_API_KEY=""
```

进程环境优先于 `.env`；`.env` 文件本身可选，但每个 `api_key_env` 引用的值
必须存在且非空。旧的六变量配置方式不再读取，也没有兼容回退。

空模型清单、重复 model、四个字段之外的未知字段、非法 URL、非法环境变量名、
缺失密钥或未注册 protocol 都会使 FastAPI 启动失败。错误只报告位置和类别，
不输出配置值或 secret。真实 `models.toml` 与 `.env` 均已被 Git 忽略。

## 安装与运行

首次安装：

```bash
uv python install 3.14
uv sync --project backend --locked
npm ci --prefix frontend
```

Linux Bash 可在仓库根目录执行：

```bash
./start
```

Windows PowerShell 分别启动：

```powershell
Set-Location backend
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

```powershell
Set-Location frontend
npm run dev
```

Vite 默认位于 `http://127.0.0.1:5173`，并把 `/api` 代理到
`http://127.0.0.1:8000`。后端健康检查为 `GET /api/health`。

## 使用流程

1. 选择模型并创建场景，再展开“Agent 与双层提示词”。
2. 分别为 A、B、C 编辑显示名、完整内层 prompt 与完整外层 prompt，保存
   设定。
3. 选择一个 Agent，在“待处理外部事件”中添加手工事件。
4. 点击“生成内层草稿”。可以编辑、重新生成或放弃；确认内层后，队首事件
   被消费并保存半回合。
5. 点击“生成外层草稿”。外层必须以 `To B: 正文` 开头（正文可多行）且
   不能发给自己。
6. 确认外层后，发送方保存规范化 `To B: 正文`，接收方队尾得到不可修改的
   `From A: 正文` 事件。
7. 继续手工选择 Agent 推进，或使用“回退最近确认”逐调用回退。

生成与重新生成每次只调用模型一次；确认不调用模型。未确认草稿、请求快照和
token usage 只存在浏览器，刷新即丢失。可见 reasoning 在确认时随草稿落盘到
场景历史，重新打开场景后仍可在时间线中展开查看；reasoning 不会进入后续
模型请求或上下文。已经确认内层的半回合保存在 JSON，重新打开场景后会恢复
到外层阶段。

手工事件只进入当前 Agent，按 FIFO 处理。仍在队列中的手工事件可编辑或删除；
由 Agent 外层消息生成的事件不可直接修改或删除。没有事件不能启动内层。

## 模型实际看到的内容

内层首轮输入：

```text
外部事件：
{本轮事件}
```

后续内层输入：

```text
外层人格上一轮对 Agent {接收者 ID}（{接收者当前姓名}）说：
{去掉 To X: 后的上回合外层正文}

外部事件：
{本轮事件}
```

接收者 ID 取自已确认的上一轮外层 turn，姓名取场景中该接收者当前的显示名。
发送方保存的外层 turn 仍是规范化的完整 `To X: 正文`，接收方事件仍是
`From X: 正文`。

外层输入：

```text
外部事件：
{本轮事件}

你内心有一个声音：
{本轮已确认内层输出}
```

每层请求只包含当前 Agent 当前层的 system prompt、该层全部已确认
input/output turns，以及上面的本轮输入。另一层 system prompt、另一层完整
历史、上述接收者 ID 与当前姓名以外的其他 Agent 状态、回退元数据和未确认
草稿都不会进入请求。

“模型请求预览”可明确选择内层或外层，并只展示协议无关的完整上下文。有序
`context` 固定为 system、完整历史的 user/assistant 交替项和 current user；
前端不从供应商 payload 反推上下文。外层没有对应已确认内层时，预览和生成
都会返回 `409`。

成功生成后，草稿区才展示本次 HTTP 调用实际发送的无凭据 JSON body 快照。
快照来自共享 client 的 request hook，不记录 URL、headers 或认证信息；失败
调用不返回调试 body。界面不增加运行时发信提示或其他模型未见的文本。

## 确认、冲突与回退

生成响应带有事件 ID、调用 ID 和协议无关状态令牌。令牌哈希场景模型、Agent、
人格层、完整事件、当前层 system prompt、当前层全部历史与本轮 input，不哈希
adapter payload。因此事件、prompt、同层历史、模型或阶段变化会使确认返回
`409`；后续内层输入中的接收者当前姓名变化也会使确认返回 `409`。单纯调整
供应商请求字段、缓存元数据或 adapter 实现不会。用户编辑后的非法外层输出
返回 `422`。界面在失败时保留草稿。

场景有一个只含调用引用的全局回退栈：

- 回退内层会删除该 turn，并把被消费事件放回原队首；
- 回退外层会删除该 turn 及其在接收者队列产生的事件，同时保留内层 turn。

只能回退全场景最近一次已确认模型调用。一次操作只回退一层，不会连锁执行。

显式未绑定或绑定模型不在当前进程配置中的场景仍可查看、编辑、管理事件和
回退，但不能新建预览或生成，相关请求返回 `409`。确认既不访问模型注册表也不
调用模型；模型从配置移除后，已有且业务状态仍有效的浏览器草稿仍可确认。
未绑定场景可通过界面或 `PUT /api/scenes/{scene_id}/model` 永久绑定一次。

## Pydantic AI Direct 与独立 prompt cache

项目保留自己的协议中性 `ModelBackend`，并通过 Pydantic AI Direct 每次恰好
发起一次请求。它不使用 `Agent`、工具、自动历史、持久会话、回退、重试或自动
压缩。

Anthropic 的两层分别使用公开的 5 分钟 block-level 缓存设置：

- `anthropic_cache_instructions='5m'` 标记当前层 system prompt；
- `anthropic_cache_messages='5m'` 标记末尾的 current user；
- 不标记最后一条 assistant，不使用 `CachePoint` 或 1 小时缓存。

current user 即使进入供应商临时缓存，在项目内仍是未确认、只由浏览器持有的
输入，不会写入场景或改变确认规则。

DeepSeek 与 MiniMax 共用 Responses-兼容映射：发送 `instructions`、完整
`input` 和 `reasoning.effort`（DeepSeek 为 `max`，MiniMax 为 `high`），MiniMax
另发 `service_tier="priority"`。两者都不设置 `max_output_tokens` 或 store/
truncation/reasoning context，不回放 reasoning ID，也不发送 conversation、
previous response ID 或 Anthropic 缓存元数据；厂商 API 天然无状态、达到模型
输出上限时直接报错，无需客户端截断。两种协议每次仍发送当前层完整历史，不做
截断或摘要，并统一展示缓存写入、缓存读取、未缓存输入和输出 token 指标；短
上下文指标为 0 属于正常情况。

后端按 TOML 顺序异步创建模型 backend，并在部分启动失败或正常退出时逆序、
幂等地关闭已创建资源；factory 构造中途失败也会先关闭 HTTP client。新增正式
协议只需实现 adapter、注册 factory 并通过共享 contract
tests，不需要修改业务 workflow、API 路由或前端。

## Scene schema v8

每个场景保存为 `data/scenes/<scene-id>.json`，采用同目录临时文件与原子
替换。当前 schema 为 v8；加载时一次性迁移 v6（先 v6→v7 再 v7→v8）与 v7
（v7→v8）文件，v1–v5 一律拒绝。

Agent 只包含：

- `id`、`name`
- `inner_context.system_prompt` 与 `inner_context.turns`
- `outer_context.system_prompt` 与 `outer_context.turns`
- `pending_events`

turn 保存调用 ID、事件 ID、显示顺序、实际 input 和确认 output，以及确认时
落盘的 `reasoning` 可见推理列表（可为空数组，模型不返回思考时为空）。内层
还保存被消费事件，外层还保存接收者和生成事件 ID。场景额外保存不可变
`model: string | null`、全局 `rollback_stack` 与单调递增 `next_sequence`。
`null` 只允许一次性绑定；回退栈和显示顺序不进入模型请求。

`data/` 是被 Git 忽略的运行时目录。删除后应用会在首次写入时重建
`data/scenes/`。

## API

- `GET /api/health`
- `GET /api/model-options`（元素严格只有 `model`，保持 TOML 顺序）
- `GET /api/scenes`
- `POST /api/scenes`（`{name, model}`）
- `GET /api/scenes/{scene_id}`
- `PUT /api/scenes/{scene_id}`
- `PUT /api/scenes/{scene_id}/model`
- `POST /api/scenes/{scene_id}/agents/{agent_id}/events`
- `PUT /api/scenes/{scene_id}/agents/{agent_id}/events/{event_id}`
- `DELETE /api/scenes/{scene_id}/agents/{agent_id}/events/{event_id}`
- `POST /api/scenes/{scene_id}/agents/{agent_id}/inner-drafts`
- `POST /api/scenes/{scene_id}/agents/{agent_id}/outer-drafts`
- `POST /api/scenes/{scene_id}/agents/{agent_id}/inner-confirmations`
- `POST /api/scenes/{scene_id}/agents/{agent_id}/outer-confirmations`
- `GET /api/scenes/{scene_id}/agents/{agent_id}/model-request-preview?layer=inner|outer`
- `POST /api/scenes/{scene_id}/rollback`

本机运行后可打开 `http://127.0.0.1:8000/docs` 查看接口文档。

## 自动验证

后端：

```bash
cd backend
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked python -m compileall app tests
uv run --locked pytest
```

前端：

```bash
cd frontend
npm test
npm run typecheck
npm run build
```

后端测试使用假的模型客户端，不会消耗真实 API 配额。
