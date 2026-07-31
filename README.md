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

复制根目录的 [`.env.example`](.env.example) 为 `.env`：

```dotenv
ANTHROPIC_BASE_URL="https://api.example.com/anthropic"
ANTHROPIC_API_KEY="sk-anthropic-xxxxx"
ANTHROPIC_MODEL="anthropic/claude-haiku-4.5"

RESPONSES_BASE_URL="https://api.example.com/v1"
RESPONSES_API_KEY="sk-responses-xxxxx"
RESPONSES_MODEL="gpt-5-mini"
```

- `ANTHROPIC_*` 配置 Anthropic Messages API 兼容端点和模型。
- `RESPONSES_*` 配置 OpenAI Responses API 兼容端点和模型。
- 两个 base URL 都会原样交给对应 Python SDK。
- 新场景必须从这两个具体模型中显式选择一个；场景内的两层和三个 Agent
  始终共用该绑定，绑定后不能更换。
- 同名进程环境变量优先于 `.env`。

任一配置缺失、空白，base URL 不是有效 HTTP(S) URL，或模型名与协议冲突时，
FastAPI 启动失败；错误只列变量名，不输出密钥或其他值。真实 `.env` 已被
Git 忽略。

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
5. 点击“生成外层草稿”。外层必须为一行 `To B: 正文` 且不能发给自己。
6. 确认外层后，发送方保存规范化 `To B: 正文`，接收方队尾得到不可修改的
   `From A: 正文` 事件。
7. 继续手工选择 Agent 推进，或使用“回退最近确认”逐调用回退。

生成与重新生成每次只调用模型一次；确认不调用模型。未确认草稿、请求快照、
token usage 和临时 reasoning 只存在浏览器，刷新即丢失。reasoning 不会保存
或进入后续上下文。已经确认内层的半回合保存在 JSON，重新打开场景后会恢复
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
外层人格：
{上回合外层输出}

外部事件：
{本轮事件}
```

外层输入：

```text
外部事件：
{本轮事件}

你内心有一个声音：
{本轮已确认内层输出}
```

每层请求只包含当前 Agent 当前层的 system prompt、该层全部已确认
input/output turns，以及上面的本轮输入。另一层 system prompt、另一层完整
历史、其他 Agent 状态、回退元数据和未确认草稿都不会进入请求。

“模型请求预览”可明确选择内层或外层，并展示可读上下文与同一份原始 JSON。
外层没有对应已确认内层时，预览和生成都会返回 `409`。

绑定模型为 Anthropic 时，原始请求使用 `system/messages`；绑定 Responses
模型时使用 `instructions/input/store=false`。界面的可读视图来自同一份协议
原始快照，不增加运行时发信提示或其他模型未见的文本。

## 确认、冲突与回退

生成响应带有事件 ID、调用 ID 和完整请求状态令牌。确认时后端重新构造当前
请求；如果期间队首事件、相关 prompt、同层历史或人格阶段发生变化，返回
`409`，且不写入 JSON。用户编辑后的非法外层输出返回 `422`。界面在失败时
保留草稿。

场景有一个只含调用引用的全局回退栈：

- 回退内层会删除该 turn，并把被消费事件放回原队首；
- 回退外层会删除该 turn 及其在接收者队列产生的事件，同时保留内层 turn。

只能回退全场景最近一次已确认模型调用。一次操作只回退一层，不会连锁执行。

显式未绑定或绑定模型不在当前进程配置中的场景仍可查看、编辑、管理事件和
回退，但预览、生成及确认两层调用均返回 `409`。未绑定 v6 场景可通过界面或
`PUT /api/scenes/{scene_id}/model` 永久绑定一次。

## 双协议与独立 prompt cache

Anthropic 的两层分别使用原生 block-level
`{"type":"ephemeral","ttl":"5m"}` 缓存：

- 当前层 system prompt 后有一个断点；
- 当前层历史非空时，最后一个已确认 output 后有一个滚动断点；
- 当前未确认输入不缓存。

Responses 使用提供商自动缓存，请求中不发送 Anthropic 的缓存元数据。两种
协议每次仍发送当前层完整历史，不做截断或摘要，并统一展示缓存写入、缓存
读取、未缓存输入和输出 token 指标；短上下文指标为 0 属于正常情况。

## Scene schema v6

每个场景保存为 `data/scenes/<scene-id>.json`，采用同目录临时文件与原子
替换。schema v6 不读取或迁移任何旧 schema。

Agent 只包含：

- `id`、`name`
- `inner_context.system_prompt` 与 `inner_context.turns`
- `outer_context.system_prompt` 与 `outer_context.turns`
- `pending_events`

turn 保存调用 ID、事件 ID、显示顺序、实际 input 和确认 output。内层还保存
被消费事件，外层还保存接收者和生成事件 ID。场景额外保存不可变
`model: string | null`、全局 `rollback_stack` 与单调递增 `next_sequence`。
`null` 只允许一次性绑定；回退栈和显示顺序不进入模型请求。

`data/` 是被 Git 忽略的运行时目录。删除后应用会在首次写入时重建
`data/scenes/`。

## API

- `GET /api/health`
- `GET /api/model-options`
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
uv run ruff format --check .
uv run ruff check .
uv run python -m compileall app tests
uv run pytest
```

前端：

```bash
cd frontend
npm test
npm run typecheck
npm run build
```

后端测试使用假的模型客户端，不会消耗真实 API 配额。
