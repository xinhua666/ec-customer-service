"""Step 1: 统一模型初始化 —— 全项目的模型入口。

只在这里初始化模型，其他模块一律从本文件导入，
后续换模型（DeepSeek -> GPT -> 智谱）只改一处。
"""
from langchain.chat_models import init_chat_model

from config.settings import LLM_API_BASE, LLM_API_KEY, LLM_MODEL


def get_llm(temperature: float = 0.2, streaming: bool = False):
    """获取统一的 Chat Model 实例。

    temperature 说明：
    - 客服问答用低温度（0.2），答案更稳定、少幻觉
    - 营销文案改写可以调高
    """
    return init_chat_model(
        LLM_MODEL,
        model_provider="openai",          # DeepSeek/智谱均兼容 OpenAI 协议
        base_url=LLM_API_BASE,
        api_key=LLM_API_KEY,
        temperature=temperature,
        streaming=streaming,
    )
