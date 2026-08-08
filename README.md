# AI 小镇

这是一个由同一大模型驱动三个 Agent 的双层人格手工实验系统。当前唯一业务
闭环是：

> 外部事件排队 → 生成并确认内层 → 生成并确认外层 →
> 把外层输出路由为接收者的新事件

每个 Agent 的内层与外层各有独立完整历史；system prompt 由后端固定模板和
用户变量动态组装，不写入 scene。系统不会自动推进、自动记忆、压缩上下文或
建立供 Agent 使用的公共时间线。完整产品假设与信息边界以
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) 为准。

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
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 `
  --log-config logging.json --no-access-log
```

```powershell
Set-Location frontend
npm run dev
```

Vite 默认位于 `http://127.0.0.1:5173`，并把 `/api` 代理到
`http://127.0.0.1:8000`。后端健康检查为 `GET /api/health`。

## 服务器日志

`./start` 和上面的 Windows 命令都会加载 `backend/logging.json`。应用从同一份
已脱敏事件生成两种输出：

- 服务器终端显示带层级的人类可读格式：UTC 时间、等级、事件、消息和非空关联
  字段；普通字段紧凑排列，复杂对象和多行诊断缩进显示。TTY 自动着色，设置任意
  值的 `NO_COLOR` 可关闭颜色。
- 全部结构化应用事件以单行 JSON 写入根目录 `logs/ai-town.jsonl`。文件达到
  10 MiB 后轮转，保留 `ai-town.jsonl.1` 至 `.5`；目录和文件由后端按需创建，
  路径不受启动工作目录影响。

Uvicorn 自己的启动、reload 和关闭提示只写终端，重复 access log 仍关闭。
`http.request.started` 是 DEBUG，因此默认 INFO 终端中一个请求只显示完成事件；
业务事件和 `model.call.started` 仍为 INFO。JSON 记录固定包含 UTC `timestamp`、
`level`、`logger`、`event`、`message`，以及 `request_id`、`scene_id`、
`agent_id`、`layer`、`call_id`、`model`、`provider`。所有 HTTP 响应都通过
`X-Request-ID` 返回后端新生成的 UUID；客户端传入的同名请求头不会被采用。

成功事件只记录 ID、计数、耗时、token、正文长度和 SHA-256 等元数据，不记录
请求或响应正文。错误诊断不截断，并按 `failure_category` 区分外层输出协议、
供应商 HTTP、上游超时、连接失败、响应投影和其他内部错误。HTTP 状态与 body、
原始可见输出、provider response 或 traceback 会按故障类型完整写入；终端错误
还显示 `request_id`、`call_id` 和 JSONL 路径，便于定位同一记录。

两条通道在格式化前共同递归脱敏：已解析配置中的真实 API key、URL 编码后的
key，以及 Authorization、API-key、token 和 cookie 类字段都会替换为
`[REDACTED]`。除此之外，错误日志可能包含完整 prompt、事件、模型正文、原始
签名、加密 reasoning 或 redacted provider details。它们仍不会进入浏览器响应、
reasoning 投影、确认数据、scene 或后续模型上下文。

`logs/` 已整体被 Git 忽略。活动文件加 5 份备份通常约为 60 MiB；为保证单条
诊断不被截断，超大记录可能使占用暂时超过该值。服务器终端和 `logs/` 都应按
敏感实验数据管理，不能当作场景备份或提交到版本库。

## 使用流程

1. 选择模型并创建场景，再展开“Agent 提示词变量与互动角色”。
2. 为需要推进的 Agent 编辑姓名、代词、不可说出口的信念、内层记忆与外层
   记忆，并按其他人物的当前姓名填写人物简介和“称呼/使用场合”，保存设定。
   B/C 可以保持空配置。
3. 选择一个 Agent，在“待处理外部事件”中添加手工事件。
4. 点击“生成内层草稿”。可以编辑、重新生成或放弃；确认内层后，队首事件
   被消费并保存半回合。
5. 点击“生成外层草稿”。外层只能输出如 `对儿子说：正文` 的已配置语义发言，
   或精确的 `STOP`。
6. 确认语义发言后，发送方保存规范化语义输出，接收方队尾得到只含纯正文和
   来源元数据的不可修改事件；确认 `STOP` 只保存 turn，不创建事件。
7. 继续手工选择 Agent 推进，或使用“回退最近确认”逐调用回退。

生成与重新生成每次只调用模型一次；确认不调用模型。未确认草稿、请求快照和
token usage 只存在浏览器，刷新即丢失。可见 reasoning 在确认时随草稿落盘到
场景历史，重新打开场景后仍可在时间线中展开查看；reasoning 不会进入后续
模型请求或上下文。已经确认内层的半回合保存在 JSON，重新打开场景后会恢复
到外层阶段。

手工事件只进入当前 Agent，按 FIFO 处理。仍在队列中的手工事件可编辑或删除；
由 Agent 外层消息生成的事件不可直接修改或删除。手工事件逐字保存且不解析，
可直接输入 `儿子对你说：正文` 模拟回复。没有事件不能启动内层。

四个提示词变量、空人物简介与空互动字典都允许保存。某 Agent 只有在四个变量
均非空、至少有一个互动称呼，且每个拥有称呼的目标人物都有非空简介时，才能
预览、生成或确认；因此当前实验可以只配置并推进 A，让 B/C 仅作为手工模拟的
路由目标。互动目标按内部 A/B/C 固定顺序进入外层 system prompt，同一目标下
的称呼保持页面录入顺序，但模型看到的人物标识只有当前姓名，不含内部 ID。
三位人物去除首尾空白后的姓名必须互不相同，比较保持大小写敏感。

完整固定模板逐字定义见 `PROJECT_CONTEXT.md` 第 8 节。模板是后端 `.j2` 文件，
使用严格变量、关闭 HTML 转义且只渲染一次；用户文本中的 `{{ ... }}` 保持原样，
前端不保存或复制模板。

## 模型实际看到的内容

内层首轮输入：

```text
外部事件：
{本轮事件}
```

后续内层输入：

```text
上一轮：
你对{上一轮称呼}说：
{上一轮正文}

外部事件：
{本轮事件}
```

上一轮为 `STOP` 时，开头改为 `上一轮：你没有说话。`。发送方保存的外层
turn 是规范化语义输出或 `STOP`；接收方 Agent 事件只保存纯正文。

外层输入：

```text
外部事件：
{本轮事件}

你内心有一个声音：
{本轮已确认内层输出}
```

外层 system prompt 的“当下可互动角色”按人物分组，例如：

```text
【当下可互动角色】
以下每组信息分别说明一位可互动人物的姓名、你对其的简单认识，以及你可以使用的全部称呼和对应场合。

人物姓名：李国栋
人物简介：
你的儿子，最近工作不顺。

可用称呼及对应场合：
- 称呼：儿子
  使用场合：一般场合
- 称呼：国栋
  使用场合：对他感到生气时
```

外层被明确要求从这些称呼中选择符合当前场合的一项，再输出
`对{称呼}说：{正文}`；语义路由仍由称呼反查目标内部 ID。

每层请求只包含后端为当前 Agent 当前层动态组装的 system prompt、该层全部已确认
input/output turns，以及上面的本轮输入。另一层 system prompt、另一层完整
历史、内部 Agent ID、可互动人物姓名与当前 Agent 关系视角之外的其他 Agent
状态、回退元数据和未确认草稿都不会进入请求。

“模型请求预览”可明确选择内层或外层，并只展示协议无关的完整上下文。有序
`context` 固定为 system、完整历史的 user/assistant 交替项和 current user；
前端不从供应商 payload 反推上下文。外层没有对应已确认内层时，预览和生成
都会返回 `409`。

成功生成后，草稿区才展示本次 HTTP 调用实际发送的无凭据 JSON body 快照。
快照来自共享 client 的 request hook，不记录 URL、headers 或认证信息；失败
调用不返回调试 body。界面不增加运行时发信提示或其他模型未见的文本。

## 确认、冲突与回退

生成响应带有事件 ID、调用 ID 和协议无关状态令牌。令牌哈希场景模型、Agent、
人格层、完整事件、动态 system prompt、当前层全部历史、本轮 input，以及完整
提示词变量和有序互动字典，不哈希 adapter payload。因此任一变量或互动配置、
事件、同层历史、模型或阶段变化会使确认返回 `409`；目标人物改名还会改变引用
该人物的当前外层 system prompt，使相关外层草稿失效。改名不会改写已保存的
称呼或语义历史。供应商请求字段、缓存元数据或 adapter 实现变化不会单独使草稿
失效。用户编辑后的非法外层输出返回 `422`，界面在失败时保留草稿。

场景有一个只含调用引用的全局回退栈：

- 回退内层会删除该 turn，并把被消费事件放回原队首；
- 回退外层会删除该 turn；语义发言同时删除接收事件，`STOP` 没有事件可删，
  两者都保留内层 turn。

只能回退全场景最近一次已确认模型调用。一次操作只回退一层，不会连锁执行。

绑定模型不在当前进程配置中的场景仍可查看、编辑、管理事件和回退，但不能新建
预览或生成，相关请求返回 `409`。确认既不访问模型注册表也不调用模型；模型从
配置移除后，已有且业务状态仍有效的浏览器草稿仍可确认。场景始终保存不可变的
非空模型绑定，不存在未绑定或后续绑定流程。

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

## Scene contract versions

每个场景保存为 `data/scenes/<scene-id>.json`，采用同目录临时文件与原子
替换。当前 schema 为 `ai-town.scene/1.0`，格式严格为
`ai-town.scene/{major}.{minor}`。主版本对应持久化结构的不兼容变化；当前程序
接受所有合法 `ai-town.scene/1.x`，其他主版本因版本不兼容而拒绝。次版本可对应
提示词模板等非持久化行为变化；同一主版本之间不迁移、不补全，只严格校验当前
完整结构。缺字段、多字段或非法值都视为损坏且不改写文件。已有 `1.x` 场景读取
并保存后仍保留原次版本，不会自动升降。

实现者不得推断、自动递增或自行修改版本。任何相关变更开始前都必须询问用户，
并使用用户明确指定的目标版本；新场景使用代码中已确认的当前版本 `1.0`。包
版本、模型名称和依赖版本不属于这一 scene contract 版本体系。

Agent 只包含：

- `id`、`name`（姓名去除首尾空白后在场景内唯一，大小写敏感）
- `prompt_profile`：`pronoun`、`hidden_beliefs`、`inner_memories`、
  `outer_memories`
- `interactions`：其他 Agent ID 到
  `{description: string, addresses: {称呼: 使用场合}}` 的映射；简介可为空，
  称呼保持录入顺序
- `inner_context.turns` 与 `outer_context.turns`（不含 `system_prompt`）
- `pending_events`

turn 保存调用 ID、事件 ID、显示顺序、实际 input 和确认 output，以及确认时
落盘的 `reasoning` 可见推理列表（可为空数组，模型不返回思考时为空）。内层
还保存被消费事件；外层语义发言保存接收者和生成事件 ID，`STOP` 的两项均为
`null`。场景额外保存不可变且非空的 `model: string`、全局 `rollback_stack` 与
单调递增 `next_sequence`。所有契约字段必须显式存在；空 `reasoning`、turns、
队列和回退栈以 `[]` 保存，规定为空的路由字段以 `null` 保存。回退栈和显示顺序
不进入模型请求。

`data/` 是被 Git 忽略的运行时目录。删除后应用会在首次写入时重建
`data/scenes/`。

## API

- `GET /api/health`
- `GET /api/model-options`（元素严格只有 `model`，保持 TOML 顺序）
- `GET /api/scenes`
- `POST /api/scenes`（`{name, model}`）
- `GET /api/scenes/{scene_id}`
- `PUT /api/scenes/{scene_id}`
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
