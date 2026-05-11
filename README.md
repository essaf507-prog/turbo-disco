# Paper Style Transfer Pipeline

本项目用于处理一批英文 PDF 论文：先抽取正文、按语义友好的方式切分为 chunks，再调用大模型 API 分析名校论文的表达习惯，沉淀可复用的风格模板。之后用户输入自己的英文文章，系统会依据训练得到的模板进行学术论文风格润色，并直接返回润色结果。

> 代码以 Python 为主，可在 IntelliJ IDEA / PyCharm 中打开项目、配置虚拟环境后运行 CLI 或 HTTP API。

## 模块职责

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| PDF 读取与文字提取 | `paper_style_pipeline/pdf_reader.py` | 读取单个 PDF 或目录下的 PDF，提取每页文本并保留来源页码元数据。 |
| 文本切块 | `paper_style_pipeline/chunker.py` | 将论文文本按段落和最大 token 估算切成可送入模型的 chunk，避免打断句子和段落。 |
| API 客户端 | `paper_style_pipeline/llm_client.py` | 提供 OpenAI-compatible Chat Completions 客户端；通过环境变量配置 API 地址、Key 和模型。 |
| 语义/风格分析 | `paper_style_pipeline/style_analyzer.py` | 对每个 chunk 提取论文表达习惯、句式、连接词、论证方式和常用短语，并合并为风格画像。 |
| 风格画像存储 | `paper_style_pipeline/style_profile.py` | 定义可序列化的风格模板数据结构，支持保存/读取 JSON。 |
| 文章润色 | `paper_style_pipeline/polisher.py` | 将用户文章与风格画像一起提交给大模型，输出符合论文库风格的润色稿。 |
| 命令行入口 | `paper_style_pipeline/cli.py` | 提供训练风格模板、润色文章、启动 API 服务三个命令。 |
| HTTP 接口 | `paper_style_pipeline/api.py` | FastAPI 服务，暴露 `/train`、`/polish` 和 `/health` 接口，便于 IDEA 中调试或与前端/其他系统集成。 |
| 代码审查辅助 | `scripts/code_review.py` | 运行基础静态检查、单元测试与可选 ruff 检查，形成轻量代码审查结果。 |
| 测试 | `tests/` | 覆盖 chunk 切分、风格画像序列化和 fake LLM 流程。 |

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,pdf,dev]"
```

如果只想运行无网络单元测试，可以不安装额外依赖；PDF 读取需要 `pypdf`，HTTP 服务需要 `fastapi` 和 `uvicorn`。

## API 配置

本项目默认使用 OpenAI-compatible Chat Completions API。请通过环境变量配置：

```bash
export LLM_API_KEY="你的 API Key"
export LLM_BASE_URL="https://api.openai.com/v1"   # 可替换为兼容服务地址
export LLM_MODEL="gpt-4.1-mini"                  # 可替换为你的模型
```

也可以在 CLI 中通过参数覆盖模型：

```bash
paper-style train ./papers --output style_profile.json --model gpt-4.1-mini
```

## 使用方式

### 1. 训练论文库风格模板

```bash
paper-style train ./papers --output style_profile.json --chunk-size 1200
```

流程：

1. 读取 `./papers` 目录中的所有 PDF。
2. 从每页提取英文正文。
3. 切分为适合模型处理的 chunks。
4. 调用 API 分析每个 chunk 的语义、修辞和学术写作习惯。
5. 合并并保存 `style_profile.json`。

### 2. 润色自己的文章

```bash
paper-style polish --profile style_profile.json --input my_draft.txt --output polished.txt
```

如果不传 `--output`，润色结果会直接打印到终端。

### 3. 启动 HTTP API

```bash
paper-style serve --host 127.0.0.1 --port 8000
```

接口示例：

```bash
curl -X POST http://127.0.0.1:8000/polish \
  -H 'Content-Type: application/json' \
  -d '{"profile_path":"style_profile.json","text":"This paper talks about..."}'
```

## HTTP 接口

### `GET /health`

返回服务状态。

### `POST /train`

请求体：

```json
{
  "pdf_path": "./papers",
  "output_profile": "style_profile.json",
  "chunk_size": 1200,
  "overlap": 120
}
```

### `POST /polish`

请求体：

```json
{
  "profile_path": "style_profile.json",
  "text": "Your draft text..."
}
```

返回：

```json
{
  "polished_text": "..."
}
```

## 在 IDEA 中运行

1. 用 IntelliJ IDEA 打开项目根目录。
2. 安装 Python 插件或使用 PyCharm。
3. 配置项目 SDK 为 `.venv/bin/python`。
4. 在 Run Configuration 中选择：
   - CLI 训练：Module name = `paper_style_pipeline.cli`，Parameters = `train ./papers --output style_profile.json`
   - CLI 润色：Module name = `paper_style_pipeline.cli`，Parameters = `polish --profile style_profile.json --input my_draft.txt`
   - HTTP 服务：Module name = `paper_style_pipeline.cli`，Parameters = `serve --host 127.0.0.1 --port 8000`

## 代码审查

运行：

```bash
python scripts/code_review.py
```

审查内容包括：

- Python 语法编译检查；
- 单元测试；
- 如果安装了 ruff，则执行 lint；否则给出跳过提示。

## 注意事项

- 项目不会把 PDF 文本本地训练成模型权重，而是提取“风格画像”作为可复用模板；真正的语义分析和润色由外部 LLM API 完成。
- 请确认论文和 API 服务的使用符合版权、隐私与学校/机构政策。
- 输出是辅助润色结果，正式投稿前仍建议人工核查事实、引用、术语和学术诚信要求。
