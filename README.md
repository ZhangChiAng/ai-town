# AI 小镇

这是由同一个大模型驱动三个 Agent 的最小实验系统。当前业务闭环是：

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
BASE_URL="https://api.example.com/anthropic"
API_KEY="sk-xxxxx"
MODEL="anthropic/claude-haiku-4.5"
```

- `BASE_URL` 是原生 Anthropic API 根地址，会原样交给 Anthropic Python
  SDK；不要填写 OpenAI Compatible 的 `/chat/completions` 地址。
- `API_KEY` 是该 Anthropic 端点使用的密钥。
- `MODEL` 是端点接受的模型名。
- 同名进程环境变量优先于 `.env`。

三项中的任何一项缺失、空白，或 `BASE_URL` 不是有效的 HTTP(S) URL 时，
FastAPI 会启动失败。错误只列出变量名，不输出变量值。真实 `.env` 已被
Git 忽略，只有 `.env.example` 应进入版本库。

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

场景必须先保存，才能点击“生成草稿”。一次点击只调用一次 Anthropic
Messages API，最长等待 60 秒，SDK 自动重试已关闭。模型通过严格且强制的
`compose_message` 工具：

- 从另外两位 Agent 中选择一位接收者；
- 生成非空消息正文；
- 不能选择自己，也不能选择沉默。

发给模型的上下文只有：

- `system`：当前 Agent 已保存的 `system_prompt`，逐字作为唯一角色提示；
- 当前 Agent 自己的已确认个人时间线，逐条映射为原生 Messages API 消息：
  收到的记录使用 `user` 角色和 `From {counterpart_id}: {content}`，发出的记录
  使用 `assistant` 角色和 `To {counterpart_id}: {content}`。

每条时间线记录保持独立和原有顺序，连续同角色消息不会合并。空时间线产生空
`messages`，不会追加运行时提示语。四个拼接槽位不会在 user context 中重复
发送。身份与候选关系应由用户编辑的 system prompt 提供。场景名、其他 Agent
的私有设定和时间线、观察者信息、未确认草稿也不会进入请求。生成只覆盖浏览器
内当前 Agent 的临时草稿，不修改场景 JSON。用户仍可修改接收者和正文，也可以
完全手写。只有点击“确认发送”才会沿用消息接口，把匹配记录原子写入发送者和
接收者的时间线。

上游请求失败或返回无效工具结果时，接口返回不包含密钥及提供商响应正文的
`502`，并保留现有草稿与 usage。成功生成会以 INFO 级别记录场景 ID、Agent
ID、模型名和四项 token usage，不记录提示词、正文或 API Key。

## 5 分钟 prompt cache

草稿请求使用原生 Anthropic block-level prompt caching，并显式设置
`{"type":"ephemeral","ttl":"5m"}`：

- 最终系统提示词之后有一个断点；
- 时间线非空时，最后一条消息的文本块之后有一个滚动断点；时间线为空时没有
  第二个断点。

相同状态下重新生成可复用完整前缀。确认新消息后，既有消息的角色与正文逐字
不变，滚动断点移到新的末条消息，Anthropic 可以继续复用稳定前缀。最终系统
提示词或时间线变化后，相关前缀会自然失效。仅修改四个拼接槽位而不修改最终
系统提示词时，槽位不会进入请求。请求不会加入时间戳、请求 ID 或旧草稿，也
不会为了达到缓存门槛而填充无意义文本。

界面显示 Anthropic 返回的真实指标：

- `5 分钟缓存写入`：`cache_creation_input_tokens`
- `缓存读取`：`cache_read_input_tokens`
- `未缓存输入`：`input_tokens`
- `输出`：`output_tokens`

Anthropic 的 Claude Haiku 4.5 最小可缓存前缀是 4096 tokens。较短上下文会
正常生成，但缓存写入和读取都可能为 0；这不是应用错误。`input_tokens` 只表示
最后一个缓存断点后的未缓存输入，总输入量应把三项输入指标相加。具体限制以
[Anthropic prompt caching 文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
为准。

## 最终系统提示词

每个 Agent 的 `system_prompt` 是非空、可编辑、可持久化的权威角色提示。
新场景使用后端唯一模板初始化：

```text
【规则】
像真人一样说话，你不必全盘托出，可以推诿和回避甚至撒谎。不要输出括号包裹的动作，只输出说的话。不要有换行，所有话一口气说完。记住，一个人最本质的东西是他的欲望和恐惧。

【人设】
{人设}

【欲望】
{欲望}

【恐惧】
{恐惧}

【记忆】
{当前压缩记忆}
```

人设、欲望、恐惧、记忆保留为次要的“拼接素材”。编辑槽位不会自动修改最终
系统提示词；“从槽位重新拼接”会先要求覆盖确认，再调用无副作用的后端拼接
接口。取消或请求失败时保留原提示词。拼接完成后仍可自由编辑或删除默认规则，
但最终文本不能是空白。

## 请求预览与快照

每个 Agent 的“完整模型输入”面板包含两个互相独立的视图：

- “下一次请求预览”从已保存场景构造，不调用模型、不修改 JSON；场景有未保存
  修改时界面明确标记该预览为旧版本。
- “当前草稿实际请求”来自本次生成真正交给 Anthropic SDK 的载荷。重新生成会
  覆盖它，确认发送或放弃草稿会清除它，且它始终只保存在浏览器草稿状态中。

两个视图都展示 system、按请求顺序排列的 `user` / `assistant` 消息及完整
文本、工具输出约束、强制工具选择、缓存断点和完整原始 JSON。这里的“完整
请求”是 Anthropic Messages API 模型载荷，包括模型名和 token 上限，但不包括
API Key、Base URL 或第三方响应。提示词和快照不会写入日志。

## 场景数据与 API

每个场景保存为 `data/scenes/<scene-id>.json`。写入使用同目录临时文件和原子
替换；无法读取或结构损坏的 JSON 会明确报错，不会被忽略或覆盖。现有
`data/*.md` 仍只是实验行为参考，不会被应用加载或修改。

当前格式为 schema v2，Agent 新增非空 `system_prompt`。读取 schema v1 文件时，
后端会在内存中用旧有四个槽位生成最终系统提示词并返回 v2 表示，不会因读取而
改写文件。只有用户随后显式保存场景时，磁盘文件才升级为 v2；场景 ID 与个人
时间线保持不变。

主要接口：

- `GET /api/health`
- `GET /api/scenes`
- `POST /api/scenes`
- `POST /api/system-prompts/compose`
- `GET /api/scenes/{id}`
- `PUT /api/scenes/{id}`
- `POST /api/scenes/{id}/messages`
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

后端测试注入假 Anthropic 客户端，不会消耗真实 API 配额。

## 真实 API 缓存验收

1. 在测试场景中为一个 Agent 准备超过 Haiku 4.5 的 4096-token 缓存门槛的
   最终系统提示词和个人时间线，并保存场景。
2. 第一次点击“生成草稿”，确认“5 分钟缓存写入”大于 0。
3. 五分钟内点击“重新生成”，确认“缓存读取”大于 0；请求不会包含当前草稿。
4. 确认发送一条消息，再次选择同一 Agent 生成草稿。新的末尾时间线块会产生
   新处理量，同时先前时间线前缀应继续出现缓存读取。

若前缀不足模型门槛，写入和读取都为 0 是预期行为。不要通过复制填充文本来
制造缓存命中。
