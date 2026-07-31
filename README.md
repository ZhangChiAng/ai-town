# AI 小镇

这是每个场景由同一个大模型驱动三个 Agent 的最小实验系统。当前业务闭环是：

> 创建场景 → 编辑三个 Agent → 选择 Agent → 模型生成消息草稿 →
> 人工编辑或重新生成 → 确认发送 → 写入双方个人时间线 → JSON 保存 →
> 重启后继续

系统不会自动推进、自动更新记忆或建立全局事件。产品假设与信息边界以
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) 为准。

## 环境要求

- Python `>=3.14,<3.15`
- [uv](https://docs.astral.sh/uv/)
- Node.js `>=24,<25`
- npm（随 Node.js 安装）

根目录的 `.python-version` 和 `.node-version` 声明了项目使用的主版本。

## 模型配置

复制根目录的 [`.env.example`](.env.example) 为 `.env`，配置：

```dotenv
ANTHROPIC_BASE_URL="https://api.anthropic.com"
ANTHROPIC_API_KEY="sk-ant-xxxxx"
ANTHROPIC_MODEL="claude-haiku-4-5"
RESPONSES_BASE_URL="https://api.openai.com/v1"
RESPONSES_API_KEY="sk-xxxxx"
RESPONSES_MODEL="gpt-5"
```

- 六项全部必填。进程启动时同时创建 Anthropic 与 Responses 客户端；不同场景
  可以分别绑定其中任一具体模型。
- `ANTHROPIC_MODEL` 忽略大小写必须包含 `claude`；`RESPONSES_MODEL` 不得包含。
- 两个 `BASE_URL` 都必须是对应 SDK 的 API 根地址，并会原样传给 SDK。Anthropic
  官方根地址为 `https://api.anthropic.com`；Responses 官方根地址为
  `https://api.openai.com/v1`，OpenAI SDK 会在其后请求 `/responses`。不要填写
  `/chat/completions`，也不要把 `/responses` 本身加入根地址。
- 同名进程环境变量优先于 `.env`。

旧的 `BASE_URL/API_KEY/MODEL` 三项不再生效。六项中的任何一项缺失、空白，
URL 不是有效的 HTTP(S) URL，或模型名与协议冲突时，FastAPI 都会启动失败。
错误只列出变量名，不输出变量值。真实 `.env` 已被 Git 忽略，只有
`.env.example` 应进入版本库；已有本地 `.env` 需手动迁移，不会自动改写。

## 场景模型绑定

前端启动时读取 Anthropic 与 Responses 两项实际模型名。创建场景必须显式选择
`Claude — <ANTHROPIC_MODEL>` 或 `Responses — <RESPONSES_MODEL>`，界面不默认
选择。绑定保存的是具体模型名；同一场景的三个 Agent 共用它，创建后不可切换。

schema v1-v4 的旧场景以内存中的 `model: null` 加载，读取本身不改写文件。用户
可为其选择一次当前模型；绑定或其他后续显式保存才写入 schema v5。若部署配置
后来改了对应 `MODEL`，旧绑定不会自动跟随或允许改绑：场景仍可查看、编辑和人工
确认消息，但请求预览与草稿生成返回 `409`。

## 首次安装

在项目根目录执行：

```bash
uv python install 3.14
uv sync --project backend --locked
npm ci --prefix frontend
```

更新依赖时使用 `uv lock --project backend` 或 `npm install --prefix
frontend`，并提交相应锁文件。以上命令在 Windows PowerShell 中写法相同。

## 本地开发

### Linux Bash

```bash
./start
```

脚本会同时启动前后端；任一服务退出时会停止另一服务，按 `Ctrl+C` 也会统一
清理两个进程。脚本可以从任意工作目录调用。

### Windows PowerShell

终端 1：

```powershell
Set-Location backend
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

终端 2：

```powershell
Set-Location frontend
npm run dev
```

打开 Vite 输出的地址（默认 `http://127.0.0.1:5173`）。开发服务器会把
相对路径 `/api` 代理到 `http://127.0.0.1:8000`。

使用 VS Code Remote - SSH 时，把云端 `5173` 端口转发到本机，并以 VS
Code“端口”面板显示的地址为准。

后端健康检查为 `GET /api/health`，成功响应：

```json
{"status":"ok"}
```

## 单步模型回合

场景必须已绑定模型并保存，才能点击“生成草稿”。一次点击只调用一次场景绑定的
上游 API，最长等待 60 秒，两个 SDK 的自动重试都已关闭。Anthropic 请求不
包含工具定义或工具选择；Responses 请求设置 `store=false`，不使用工具、
`previous_response_id` 或模型专属缓存参数。每次请求末尾都会追加一条不
持久化的格式指令，要求模型从另外两位 Agent 中选择接收人，并只输出一行
`To B: 消息正文`。

Anthropic 响应允许 thinking block，但只能有一个可见 text block。Responses
响应必须为 `completed`，可以包含 reasoning item，但只能有一条 assistant
message，且只能有一个 `output_text`。两种协议的唯一可见文本都必须符合地址
格式；refusal、工具调用、多段可见文本、多行、空正文、无效接收人和发给自己
都会被拒绝。

上游实际返回的可读思维内容会在草稿旁原样显示：Anthropic 的 `thinking`、
Responses reasoning item 的 `summary_text` 与 `reasoning_text` 均保留来源类型。
草稿接口以 `reasoning: [{"type": "...", "text": "..."}]` 返回这些块。
这些内容只供观察，不可编辑；重新生成时覆盖，放弃或确认草稿时清空，不写入
场景 JSON、任何 Agent 的时间线或后续模型上下文。签名、`encrypted_content`
与 `redacted_thinking` 数据不会返回前端。应用不自行编造或强制开启思维模式；
若端点本次没有返回可读内容，界面会明确显示这一状态。

发给模型的上下文只有：

- 当前 Agent 已保存的 `system_prompt`，逐字作为唯一角色提示；Anthropic
  放在 `system`，Responses 放在 `instructions`；
- 当前 Agent 自己的已确认个人时间线，逐条映射为协议原生消息；
  收到的记录使用 `user`，发出的记录使用 `assistant`，每个 text block
  逐字使用时间线保存的 `content`，不补、删或改写 `To`/`From` 标签。

每条时间线记录保持独立的文本块和原有顺序；连续同角色记录会放入同一个 API
turn，以兼容要求 `user`/`assistant` 严格交替的网关。Anthropic 块类型是
`text`，Responses 块类型是 `input_text`。
无论时间线为空、以 `user` 结尾或以 `assistant` 结尾，格式指令都会作为末尾
独立 text block 出现；若需严格角色交替，它会并入末尾 `user` turn。该指令
不写入时间线；Anthropic 不把它作为显式缓存断点。四个拼接槽位不会在 user
context 中重复发送。场景名、其他 Agent 的私有设定和时间线、观察者信息、
未确认草稿也不会进入请求。生成只覆盖浏览器内当前 Agent 的临时草稿，不修改
场景 JSON。用户可直接编辑包括 `To B:` 在内的完整文本；确认时后端从文本解析
接收人，并把发送者的 `To B: 正文` 与接收者的 `From A: 正文` 原子写入各自
时间线。

已确认消息可以手动删除，但必须同时是发送者和接收者各自完整个人时间线的
绝对末条；夹在任一方后续消息之前的记录都不可删除。界面会在符合条件的双方
记录上显示“删除”，每次点击都展示参与者和正文并要求确认。一次确认只删除
一条，不会自动连锁删除；成功后会重新计算新的可删除消息。后端按共享
`message_id` 权威复核一发一收、参与者、视角化正文关系和双方栈顶位置，再同步移除两条
记录并原子保存，未参与的第三位 Agent 不受影响。消息不存在返回 `404`，配对
异常或不在双方栈顶返回 `409`，失败不会改写场景。

上游请求失败，或响应形状、可见草稿、reasoning、usage 不符合上述规则时，接口返回不包含
密钥及提供商响应正文的 `502`，并保留现有草稿与 usage。成功生成会以 INFO
级别记录场景 ID、Agent ID、模型名和四项 token usage，不记录提示词、正文或
API Key。

## Prompt cache 与统一 usage

Claude 草稿请求继续使用原生 Anthropic block-level prompt caching，并显式设置
`{"type":"ephemeral","ttl":"5m"}`：

- 最终系统提示词之后有一个断点；
- 时间线非空时，最后一条已确认记录的文本块之后有一个滚动断点；时间线为空时
  没有第二个断点。按需追加的运行时触发语不缓存。

相同状态下重新生成可复用完整前缀。确认新消息后，既有消息的角色与正文逐字
不变，滚动断点移到新的末条消息，Anthropic 可以继续复用稳定前缀。最终系统
提示词或时间线变化后，相关前缀会自然失效。仅修改四个拼接槽位而不修改最终
系统提示词时，槽位不会进入请求。请求不会加入时间戳、请求 ID 或旧草稿，也
不会为了达到缓存门槛而填充无意义文本。

Responses-compatible 端点使用端点自身的自动 prompt caching；应用不发送
Anthropic 断点、GPT 模型专属缓存参数或其他显式缓存控制，以兼容旧模型和
第三方网关。

草稿 API 的四个 usage 字段具有跨协议统一语义，界面显示：

- `缓存写入`：`cache_creation_input_tokens`
- `缓存读取`：`cache_read_input_tokens`
- `未缓存输入`：`input_tokens`
- `输出`：`output_tokens`

Anthropic 的四项值直接映射，`input_tokens` 只表示最后一个缓存断点后的
未缓存输入。Responses 的 `cache_write_tokens` 映射为缓存写入，
`cached_tokens` 映射为缓存读取，二者缺失或为 `null` 时为 0；未缓存输入等于
Responses 总输入减去这两项。Responses 的 `max_output_tokens` 为 2048，因为
兼容端点会把 reasoning 和最终消息共同计入该预算。不合法、负数或相互矛盾的
usage 会产生不含上游正文等敏感信息的 `502`。

Anthropic 的 Claude Haiku 4.5 最小可缓存前缀是 4096 tokens。较短上下文会
正常生成，但缓存写入和读取都可能为 0；这不是应用错误。具体限制以
[Anthropic prompt caching 文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
为准。

## 最终系统提示词

每个 Agent 的 `system_prompt` 是非空、可编辑、可持久化的权威角色提示；它会
逐字成为 Anthropic 的唯一 `system` 文本或 Responses 的 `instructions`。
新场景使用后端唯一模板初始化：

```text
【行为原则】
像真实的人一样交流。你不必坦白全部想法，可以试探、回避、推诿、隐瞒或撒谎，但你的表达应当符合你此刻的判断和目的。

【内在驱动】
你的欲望与恐惧会影响你如何理解别人说的话、注意哪些信息、相信什么，以及接下来选择说什么。它们不是需要直接复述的标签，也不要求你向别人解释自己的心理。面对含糊的信息时，按照这个人物的欲望、恐惧和既有记忆作出主观理解，而不是采用全知或完全客观的解释。

【人设】
{人设}

【欲望】
{欲望}

【恐惧】
{恐惧}

【记忆】
{当前压缩记忆}

【输出要求】
按当前回合指定的地址格式，只输出一行消息。消息正文只包含人物此刻真正会说出口的话，不要输出心理分析、推理过程或括号包裹的动作。
```

人设、欲望、恐惧、记忆保留为次要的“拼接素材”。编辑槽位不会自动修改最终
系统提示词；“从槽位重新拼接”会先要求覆盖确认，再调用无副作用的后端拼接
接口。取消或请求失败时保留原提示词。拼接完成后仍可自由编辑或删除默认规则，
但最终文本不能是空白。

## 模型请求预览

每个 Agent 的请求预览从已保存场景构造，不调用模型、不修改 JSON；场景有未
保存修改时界面明确标记该预览为旧版本。默认“可读模型上下文”按实际顺序逐字
展示 Anthropic 的 `system/messages` 或 Responses 的 `instructions/input`
中的系统指令、role 和所有文本块；“原始 JSON”展示所选协议的完整载荷。两者
共享同一次后端请求快照，不包含 API Key、Base URL 或第三方响应。

这是界面的统一可观测性原则：用户可读视图中的模型文本必须与实际请求逐字
同源。前端只负责布局，不生成模型未见的姓名、方向或解释文案，也不隐藏模型
已见的 `To`/`From` 或运行时指令。cache control 等传输元数据仍保留在原始
JSON 视图中。

## 场景数据与 API

每个场景保存为 `data/scenes/<scene-id>.json`。写入使用同目录临时文件和原子
替换；无法读取或结构损坏的 JSON 会明确报错，不会被忽略或覆盖。现有
`data/*.md` 仍只是实验行为参考，不会被应用加载或修改。

当前格式为 schema v5，包含不可变的 `model: string | null`。时间线只包含
`message` 记录，保留共享 `message_id`、收发方向、对方 Agent 与视角化最终正文。

读取 schema v1 文件时，后端会在内存中用旧有四个槽位生成最终系统提示词；
读取 v1/v2/v3 时，旧消息会在内存中补全 `type: "message"` 和相应的
`To`/`From` 前缀；读取 v1-v4 时统一返回 `model: null` 的 v5 表示。正确已有的
同类前缀不会重复添加。读取本身不会改写文件，只有一次性模型绑定或其他后续
显式写操作才会原子落盘为 v5。含已移除 `inner_voice` 记录的旧文件会返回明确
格式错误。

主要接口：

- `GET /api/health`
- `GET /api/model-options`
- `GET /api/scenes`
- `POST /api/scenes`（请求包含 `name` 与 `model`）
- `POST /api/system-prompts/compose`
- `GET /api/scenes/{id}`
- `PUT /api/scenes/{id}`
- `PUT /api/scenes/{id}/model`（仅限未绑定旧场景调用一次）
- `POST /api/scenes/{id}/messages`
- `DELETE /api/scenes/{id}/messages/{message_id}`
- `POST /api/scenes/{id}/agents/{agent_id}/message-drafts`
- `GET /api/scenes/{id}/agents/{agent_id}/model-request-preview`

本机运行后可打开 `http://127.0.0.1:8000/docs` 查看交互式接口文档。

## 自动验证

后端格式、静态检查和测试：

```bash
cd backend
uv run ruff format --check .
uv run ruff check .
uv run python -m compileall app tests
uv run pytest
```

前端测试、TypeScript 检查和生产构建：

```bash
cd frontend
npm test
npm run typecheck
npm run build
```

后端测试注入两种协议的假客户端，不会消耗真实 API 配额。

## 真实 API 验收

使用同一份六项配置分别创建绑定 Claude 与 Responses 的两个场景，并各完成一次
真实验收。每次先在请求预览确认协议载荷、私有信息边界和末尾运行时指令，再
生成一条合法草稿，核对实际返回的可读思维内容或明确空状态，再人工编辑并确认
发送；确认思维内容没有进入场景 JSON，且只有收发双方时间线发生变化。

Claude 缓存验收：

1. 为一个 Agent 准备超过所选 Claude 模型缓存门槛的最终系统提示词和个人
   时间线并保存；第一次生成确认“缓存写入”大于 0。
2. 在显式断点的 5 分钟 TTL 内重新生成，确认“缓存读取”大于 0，且请求不包含
   当前草稿。
3. 确认发送一条消息后再次生成；新末尾块产生新处理量，稳定前缀继续出现缓存
   读取。

Responses 缓存验收：

1. 打开绑定 `RESPONSES_MODEL` 的场景，确认原始预览使用
   `instructions/input/max_output_tokens/store`，其中 `store=false`，没有
   Anthropic 缓存元数据。
2. 连续生成相同前缀，按端点能力检查自动缓存 usage；端点不返回
   `cache_write_tokens` 或 `cached_tokens` 时，对应统一指标应为 0。
3. 确认响应只有一个 assistant `output_text`；incomplete、refusal、工具项或
   多段文本应得到统一的安全 `502`。

任一端点若前缀不足自身缓存门槛，写入和读取都为 0 是预期行为。不要通过复制
填充文本来制造缓存命中。
