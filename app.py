import os
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.vector_db import load_vector_db
from src.config import VECTOR_DB_PATH, DATA_DIR
from scripts.ingest import ingest_single_pdf


def init_session():
    """初始化会话状态"""
    if "rag_pipeline" not in st.session_state:
        try:
            st.session_state.rag_pipeline = RAGPipeline()
            st.session_state.db_ready = True
        except Exception as e:
            st.session_state.db_ready = False
            st.session_state.db_error = str(e)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def get_db_info():
    """获取向量库基本信息"""
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    try:
        db = load_vector_db()
        # 获取文档数量
        collection = db._collection
        count = collection.count()
        return count
    except:
        return "未知"


def main():
    st.set_page_config(
        page_title="医疗知识助手",
        page_icon="🏥",
        layout="wide"
    )

    init_session()

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.title("🏥 医疗知识助手")
        st.divider()

        # 知识库状态
        st.subheader("知识库状态")
        db_count = get_db_info()
        if db_count is None:
            st.error("❌ 向量库未初始化")
        else:
            st.success(f"✅ 向量库就绪，共 {db_count} 个知识片段")

        st.divider()

        # PDF上传导入
        st.subheader("导入文档")
        uploaded_file = st.file_uploader("上传PDF文档", type="pdf")
        if uploaded_file:
            if st.button("开始导入", type="primary"):
                with st.spinner("正在解析并入库..."):
                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        # 判断是否追加
                        append = db_count is not None
                        ingest_single_pdf(tmp_path, append=append)
                        st.success("导入成功！")
                        # 重置流水线，重新加载向量库
                        st.session_state.rag_pipeline = RAGPipeline()
                        st.rerun()
                    except Exception as e:
                        st.error(f"导入失败: {e}")
                    finally:
                        os.unlink(tmp_path)

        st.divider()
        st.caption("本系统仅作知识参考，不构成诊疗建议")

    # ========== 主界面：对话区 ==========
    st.title("医疗知识问答")
    st.caption("基于本地知识库的RAG问答系统，所有答案均来自已导入的文档")

    # 渲染历史对话
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("查看引用来源"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**来源 {i}**：{src['file']} 第 {src['page']} 页")
                        st.text(src["content"])

    # 输入框
    if question := st.chat_input("请输入您的问题..."):
        if not st.session_state.db_ready:
            st.error("向量库未初始化，请先导入文档")
            st.stop()

        # 用户消息
        st.chat_message("user").markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成答案..."):
                try:
                    result = st.session_state.rag_pipeline.ask(question)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("查看引用来源"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"**来源 {i}**：{src['file']} 第 {src['page']} 页")
                                st.text(src["content"])
                except Exception as e:
                    st.error(f"生成失败: {e}")
                    answer = f"生成失败: {e}"
                    sources = []

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()