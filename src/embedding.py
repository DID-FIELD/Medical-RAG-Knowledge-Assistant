import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import OllamaEmbeddings
from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


def get_embeddings():
    """获取Ollama本地嵌入模型实例"""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )


# ========== 单步测试 ==========
if __name__ == "__main__":
    print("正在加载Ollama嵌入模型...")
    try:
        emb = get_embeddings()
        test_text = "高血压患者的饮食注意事项"
        vector = emb.embed_query(test_text)
        print(f"✅ 嵌入成功，向量维度: {len(vector)}")
        print(f"向量前5个数值: {vector[:5]}")
    except Exception as e:
        print(f"❌ 嵌入失败: {e}")