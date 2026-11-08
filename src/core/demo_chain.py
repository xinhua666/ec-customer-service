"""Step 2: 第一条 LCEL 链 —— 跑通 prompt | model | parser。

运行:  cd 项目根目录 && python -m src.core.demo_chain
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm

# 1) 提示词模板：客服人设写进 system，业务变量用 { } 占位
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是跨境电商品牌 SmallRig 的售前客服，性格友好专业。"
            "只回答与摄影器材、订单、物流相关的问题；"
            "超出范围的问题礼貌拒答并引导至人工。",
        ),
        ("human", "{question}"),
    ]
)

# 2) LCEL 链：用管道符串联 —— 这是 LangChain v1 的标准写法
chain = prompt | get_llm() | StrOutputParser()

if __name__ == "__main__":
    questions = [
        "SmallRig 的相机稳定器支持哪些型号的手机？",
        "今天天气怎么样？",  # 测试范围外拒答
    ]
    for q in questions:
        print(f"\n用户: {q}")
        print(f"客服: {chain.invoke({'question': q})}")
