import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.config import LLM_MODEL, OLLAMA_BASE_URL, LLM_TEMPERATURE


RAG_PROMPT = """
你是医疗知识查询助手，只能使用【参考上下文】中的信息回答问题。
规则：
1. 上下文没有相关内容时，直接回答「知识库中暂无相关信息」
2. 禁止编造上下文以外的医疗内容
3. 回答简洁专业，结尾必须加上：*本回答仅作知识参考，不构成诊疗建议*

【参考上下文】
{context}

【用户问题】
{question}

【回答】
"""


def get_llm():
    """获取Ollama本地大模型实例"""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )


def get_prompt_template():
    """获取RAG提示词模板"""
    return PromptTemplate(
        template=RAG_PROMPT,
        input_variables=["context", "question"]
    )


# ========== 单步测试 ==========
if __name__ == "__main__":
    print("开始测试Ollama连接...")
    try:
        llm = get_llm()
        response = llm.invoke("你好，请用一句话自我介绍")
        print(f"✅ Ollama连接成功")
        print(f"模型回复: {response.content.strip()}")
    except Exception as e:
        print(f"❌ LLM测试失败: {e}")
        print("排查方向：")
        print("1. Ollama服务是否启动")
        print(f"2. 是否执行了 ollama pull {LLM_MODEL}")
        print(f"3. 地址 {OLLAMA_BASE_URL} 是否可访问")