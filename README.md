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

终端 1：

```bash
cd backend
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

终端 2：

```bash
cd frontend
npm run dev
```

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

- 当前 Agent 的 ID、姓名、人设、欲望、恐惧和当前压缩记忆；
- 另外两位 Agent 的 ID 与姓名；
- 当前 Agent 自己的已确认个人时间线。

场景名、其他 Agent 的私有设定和时间线、观察者信息、未确认草稿都不会进入
请求。生成只覆盖浏览器内当前 Agent 的临时草稿，不修改场景 JSON。用户仍可
修改接收者和正文，也可以完全手写。只有点击“确认发送”才会沿用消息接口，把
匹配记录原子写入发送者和接收者的时间线。

上游请求失败或返回无效工具结果时，接口返回不包含密钥及提供商响应正文的
`502`，并保留现有草稿与 usage。成功生成会以 INFO 级别记录场景 ID、Agent
ID、模型名和四项 token usage，不记录提示词、正文或 API Key。

## 5 分钟 prompt cache

草稿请求使用原生 Anthropic block-level prompt caching，并显式设置
`{"type":"ephemeral","ttl":"5m"}`：

- 固定工具与系统规则之后有一个断点；
- Agent 上下文和按确认顺序排列的独立时间线文本块之后有一个滚动断点。

相同状态下重新生成可复用完整前缀。确认新消息后，旧时间线块保持不变，
Anthropic 可以从先前断点继续复用前缀。Agent 设定或记忆被编辑并保存后，
相关前缀会自然失效。请求不会加入时间戳、请求 ID 或旧草稿，也不会为了达到
缓存门槛而填充无意义文本。

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

## 场景数据与 API

每个场景保存为 `data/scenes/<scene-id>.json`。写入使用同目录临时文件和原子
替换；无法读取或结构损坏的 JSON 会明确报错，不会被忽略或覆盖。现有
`data/*.md` 仍只是实验行为参考，不会被应用加载或修改。

主要接口：

- `GET /api/health`
- `GET /api/scenes`
- `POST /api/scenes`
- `GET /api/scenes/{id}`
- `PUT /api/scenes/{id}`
- `POST /api/scenes/{id}/messages`
- `POST /api/scenes/{id}/agents/{agent_id}/message-drafts`

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
   人设、欲望、恐惧、记忆和个人时间线，并保存场景。
2. 第一次点击“生成草稿”，确认“5 分钟缓存写入”大于 0。
3. 五分钟内点击“重新生成”，确认“缓存读取”大于 0；请求不会包含当前草稿。
4. 确认发送一条消息，再次选择同一 Agent 生成草稿。新的末尾时间线块会产生
   新处理量，同时先前时间线前缀应继续出现缓存读取。

若前缀不足模型门槛，写入和读取都为 0 是预期行为。不要通过复制填充文本来
制造缓存命中。
