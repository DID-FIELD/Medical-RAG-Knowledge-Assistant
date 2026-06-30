# 医疗知识 RAG 智能助手：基于 LangChain \+ Ollama \+ Chroma 的本地化知识库问答系统

---

# 医疗知识 RAG 智能助手

基于 LangChain \+ Ollama \+ Chroma 构建的本地化医疗知识库问答系统，全程离线运行，数据不出本地，支持 PDF 文档一键入库、语义检索、引用溯源。

## ✨ 功能特性

- 📄 **文档一键入库**：支持单个 PDF / 批量目录导入，自动语义分块

- 🔍 **语义精准检索**：基于向量相似度匹配，召回相关知识片段

- 🤖 **本地大模型推理**：基于 Ollama 部署 Qwen 系列模型，全程数据本地化

- 📑 **答案可溯源**：每条回答附带引用来源，支持查看原文片段

- 🖥️ **可视化交互界面**：Streamlit 开箱即用的 Web 问答界面

- 🔒 **完全离线**：所有模型与数据均在本地，无外网依赖

## 🛠️ 技术栈

|模块|技术选型|
|---|---|
|大模型框架|LangChain 新版（LCEL 表达式链）|
|本地大模型|Ollama \+ Qwen2\.5:3b|
|嵌入模型|nomic\-embed\-text（Ollama 本地）/ BGE\-small\-zh|
|向量数据库|Chroma|
|文档解析|PyPDF|
|Web 界面|Streamlit|

## 📁 项目目录结构

```Plain Text
Medical-RAG-Knowledge-Assistant
├── data/
│   └── medical_docs/      # 待导入的医疗PDF文档存放目录
├── src/                   # 核心源码
│   ├── __init__.py
│   ├── config.py          # 全局配置
│   ├── document_loader.py # PDF加载与语义分块
│   ├── embedding.py       # 嵌入模型封装
│   ├── vector_db.py       # 向量数据库读写
│   ├── retriever.py       # 检索器封装
│   ├── llm.py             # 大模型与Prompt
│   └── rag_pipeline.py    # RAG全流水线
├── scripts/
│   └── ingest.py          # 知识库批量入库脚本
├── vectorstore/
│   └── chroma_db/         # 向量库持久化目录
├── app.py                 # Streamlit Web 入口
├── requirements.txt       # 依赖清单
└── README.md
```

## 🚀 快速开始

### 1\. 环境准备

- Python 3\.10\+

- 安装 [Ollama](https://ollama.com/download) 客户端

### 2\. 安装依赖

```bash
# 创建虚拟环境
python -m venv rag_env
# 激活虚拟环境（Windows）
rag_env\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

### 3\. 拉取本地模型

```bash
# 拉取大语言模型
ollama pull qwen2.5:3b
# 拉取嵌入模型
ollama pull nomic-embed-text
```

### 4\. 构建知识库

1. 将 PDF 文档放入 `data/medical_docs/` 目录

2. 执行批量入库：

    ```bash
    python scripts/ingest.py
    ```

3. 也支持单个文件导入：

    ```bash
    python scripts/ingest.py --pdf 你的文档路径.pdf
    ```

### 5\. 启动 Web 界面

```bash
streamlit run app.py --server.headless true
```

启动后浏览器自动访问 `http://localhost:8501` 即可开始问答。

## ❓ 常见问题

1. **提示 ****`No module named 'langchain.text_splitter'`**
新版 LangChain 已拆分独立包，本项目依赖已适配，确保使用本项目 `requirements.txt` 安装即可。

2. **Ollama 命令找不到**
安装 Ollama 客户端后重启终端 / IDE，让系统环境变量生效。

3. **HuggingFace 模型下载超时**
推荐使用 Ollama 本地嵌入方案（默认）；如需使用 BGE 模型，可配置国内镜像：

    ```python
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    ```

4. **更换更大的模型**
修改 `src/config.py` 中 `LLM_MODEL` 配置，如改为 `qwen2.5:7b`，并执行 `ollama pull qwen2.5:7b` 即可。

## ⚠️ 免责声明

本系统仅用于医疗知识参考与学习交流，所有回答均来自已导入的文档内容，**不构成任何诊疗建议**。如有健康问题，请及时前往正规医疗机构就诊。

> （注：部分内容可能由 AI 生成）
