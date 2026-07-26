# AI 小镇

这是 AI 小镇最小实验系统。当前已完成第一个业务垂直切片：

> 创建命名场景 → 编辑三个 Agent → 显式保存 → JSON 落盘 → 重启后重新打开

当前阶段不调用模型、不生成消息，也不自动更新记忆。产品假设和信息边界以
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) 为准。

## 环境要求

- Python `>=3.14,<3.15`
- [uv](https://docs.astral.sh/uv/)
- Node.js `>=24,<25`
- npm（随 Node.js 安装）

根目录的 `.python-version` 和 `.node-version` 分别声明了本项目使用的
Python 与 Node.js 主版本。

## 首次安装

在项目根目录执行：

```bash
uv python install 3.14
uv sync --project backend
npm install --prefix frontend
```

首次安装会生成 `backend/uv.lock` 与 `frontend/package-lock.json`。锁文件
应提交到版本库。锁文件存在后，可用下面的命令进行可复现安装：

```bash
uv sync --project backend --locked
npm ci --prefix frontend
```

以上命令在 Windows PowerShell 中写法相同。

## 本地开发

后端提供 `GET /api/health`，成功响应为：

```json
{"status":"ok"}
```

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

打开 Vite 输出的本地地址（默认是 `http://127.0.0.1:5173`）。开发服务器会把
相对路径 `/api` 代理到 `http://127.0.0.1:8000`。

## 场景数据

每个场景保存为 `data/scenes/<scene-id>.json`。新场景创建时立即写入初始文件，
此后的表单修改只在点击“保存场景”时落盘。写入使用同目录临时文件和原子替换；
无法读取或结构损坏的 JSON 会通过 API 明确报错，不会被忽略或覆盖。

当前场景格式包含：

- `schema_version`、UUID `id` 与场景名称；
- 固定顺序且固定 ID 为 A、B、C 的三个 Agent；
- 每个 Agent 的显示名、人设、欲望、恐惧、当前压缩记忆与空时间线。

现有 `data/*.md` 仍只是实验行为参考，不会被应用加载或修改。

## 场景 API

- `GET /api/scenes`
- `POST /api/scenes`
- `GET /api/scenes/{id}`
- `PUT /api/scenes/{id}`

可在后端运行后打开 `http://127.0.0.1:8000/docs` 查看交互式接口文档。

## 验证

后端测试：

```bash
cd backend
uv run pytest
```

前端测试、TypeScript 检查与生产构建：

```bash
cd frontend
npm test
npm run typecheck
npm run build
```
