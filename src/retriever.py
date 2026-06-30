import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from typing import List
from src.config import RETRIEVE_TOP_K
from src.vector_db import similarity_search


class Retriever:
    def __init__(self, top_k: int = RETRIEVE_TOP_K):
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """根据问题召回Top-K相关文档片段"""
        return similarity_search(query, top_k=self.top_k)


# ========== 单步测试 ==========
if __name__ == "__main__":
    print("开始测试检索器封装...")
    try:
        retriever = Retriever(top_k=2)
        results = retriever.retrieve("糖尿病怎么控制")
        print(f"✅ 检索成功，返回 {len(results)} 条结果")
        for i, doc in enumerate(results):
            print(f"[{i+1}] {doc.page_content}")
    except Exception as e:
        print(f"❌ 检索器测试失败: {e}")