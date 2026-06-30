import sys
import os
# 自动修复路径：把项目根目录加入Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(pdf_path: str) -> List[Document]:
    """加载单个PDF文件，解析文本并按中文语义分块"""
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # 中文优化的递归分块策略
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    chunked_docs = splitter.split_documents(raw_docs)
    return chunked_docs


# ========== 单步测试 ==========
if __name__ == "__main__":
    # 自动定位测试PDF路径
    test_pdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "medical_docs", "test_guide.pdf"
    )
    print(f"测试文件路径: {test_pdf}")

    if not os.path.exists(test_pdf):
        print("❌ 测试PDF不存在，请先放到 data/medical_docs/test_guide.pdf")
        sys.exit(1)

    try:
        chunks = load_pdf(test_pdf)
        print(f"✅ 加载成功，共生成 {len(chunks)} 个分块")
        print("\n第一个分块内容预览：")
        print("-" * 40)
        print(chunks[0].page_content[:150])
        print("-" * 40)
    except Exception as e:
        print(f"❌ 加载失败: {e}")