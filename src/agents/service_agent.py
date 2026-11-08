"""Step 4b: 带 RAG + 工具的客服 Agent（create_agent，v1 标准写法）。

运行: python -m src.agents.service_agent
"""
from langchain.agents import create_agent
from langchain_core.tools import tool

from src.core.llm import get_llm
from src.retrieval.kb import get_retriever
from src.retrieval.rag_chain import format_docs
from src.tools.shop_tools import TOOLS

SYSTEM = (
    "你是 SmallRig 跨境电商客服。\n"
    "能力分工：\n"
    "- 产品/售后政策问题：先用 retrieve_knowledge 查知识库，基于结果回答\n"
    "- 订单/物流问题：调用对应工具查询\n"
    "- 查不到的信息如实告知，建议转人工，禁止编造"
)


# 把检索器也包装成一个工具，交给 Agent 自主决定何时查知识库
def make_retriever_tool():
    retriever = get_retriever()

    @tool
    def retrieve_knowledge(query: str) -> str:
        """检索 SmallRig 产品手册与售后政策知识库。产品参数、退换货规则等问题先用这个。"""
        return format_docs(retriever.invoke(query))

    return retrieve_knowledge


agent = create_agent(
    get_llm(),
    tools=TOOLS + [make_retriever_tool()],
    system_prompt=SYSTEM,
)

if __name__ == "__main__":
    for q in [
        "订单 SO2026001 发货了吗？现在到哪了？",
        "补光灯 P200 的保修政策是什么？",
    ]:
        print(f"\n用户: {q}")
        result = agent.invoke({"messages": [("user", q)]})
        print(f"客服: {result['messages'][-1].content}")
