import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.llm import get_llm, get_prompt_template
from src.vector_db import load_vector_db
from src.config import RETRIEVE_TOP_K


class RAGPipeline:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = get_prompt_template()
        self.db = load_vector_db()

        # 构建检索器
        self.retriever = self.db.as_retriever(
            search_kwargs={"k": RETRIEVE_TOP_K}
        )

        # LCEL 表达式链：官方推荐的新版RAG写法
        self.rag_chain = (
            {"context": self.retriever | self._format_documents, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_documents(docs):
        """将检索到的多个文档拼接成统一上下文"""
        return "\n\n".join([doc.page_content for doc in docs])

    def ask(self, question: str) -> dict:
        """一站式问答接口，返回答案+引用来源"""
        # 先召回相关文档
        source_docs = self.retriever.invoke(question)
        # 执行RAG生成
        answer = self.rag_chain.invoke(question)

        # 格式化来源信息
        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content,
                "file": doc.metadata.get("source", "未知文件"),
                "page": doc.metadata.get("page", 0) + 1
            })

        return {
            "answer": answer,
            "sources": sources
        }


# ========== 单步测试 ==========
if __name__ == "__main__":
    print("开始测试完整RAG流水线...")
    try:
        rag = RAGPipeline()
        result = rag.ask("高血压患者每天吃盐不能超过多少克？")
        
        print("✅ RAG流水线运行成功")
        print("\n" + "="*50)
        print("【问题】高血压患者每天吃盐不能超过多少克？")
        print("【答案】")
        print(result["answer"])
        print("="*50)
        print(f"\n引用来源数量: {len(result['sources'])}")
        for i, src in enumerate(result["sources"]):
            print(f"[{i+1}] {src['file']} 第{src['page']}页: {src['content'][:50]}...")
    except Exception as e:
        print(f"❌ RAG流水线测试失败: {e}")