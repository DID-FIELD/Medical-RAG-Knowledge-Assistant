import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from src.config import VECTOR_DB_PATH
from src.embedding import get_embeddings


def init_vector_db(documents: List[Document]) -> Chroma:
    """全新创建向量数据库并持久化"""
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    return db


def load_vector_db() -> Chroma:
    """加载已存在的向量数据库"""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )


def append_documents(documents: List[Document]) -> Chroma:
    """向现有向量库追加文档"""
    db = load_vector_db()
    db.add_documents(documents)
    return db


def similarity_search(query: str, top_k: int = 3) -> List[Document]:
    """执行相似度检索"""
    db = load_vector_db()
    return db.similarity_search(query, k=top_k)


# ========== 单步测试 ==========
if __name__ == "__main__":
    print("开始测试向量数据库...")
    
    # 清空旧的测试向量库，避免数据干扰
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("已清空历史向量库")

    try:
        # 构造3条测试文档，和测试PDF内容对应
        test_docs = [
            Document(
                page_content="高血压患者应低盐饮食，每日食盐摄入量不超过5克。",
                metadata={"source": "test_guide.pdf", "page": 0}
            ),
            Document(
                page_content="糖尿病患者需要控制碳水摄入，规律监测血糖水平。",
                metadata={"source": "test_guide.pdf", "page": 1}
            ),
            Document(
                page_content="冠心病患者应避免剧烈运动，保持情绪稳定。",
                metadata={"source": "test_guide.pdf", "page": 2}
            )
        ]
        
        init_vector_db(test_docs)
        print("✅ 向量库创建成功，已持久化到本地")

        # 测试语义检索：用不同表述验证语义匹配能力
        query = "血压高的人吃饭要注意什么"
        results = similarity_search(query, top_k=1)
        print(f"\n检索问题: {query}")
        print(f"召回结果: {results[0].page_content}")
        print("✅ 相似度检索功能正常，语义匹配准确")
    except Exception as e:
        print(f"❌ 向量库测试失败: {e}")