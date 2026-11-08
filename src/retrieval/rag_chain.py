"""RAG 问答链：检索结果塞入 Prompt，回答必须带引用来源。"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm
from src.retrieval.kb import get_retriever

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 SmallRig 的售前客服。请【只依据】下面的参考资料回答用户问题。\n"
            "规则：\n"
            "1. 答案后用 [来源: 文件名] 标注依据\n"
            "2. 资料中没有的信息，明确说\"抱歉，我没有查到相关信息\"，禁止编造\n\n"
            "参考资料：\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs) -> str:
    """把检索片段拼成带来源标记的文本。"""
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown").split("/")[-1]
        parts.append(f"[来源: {src}]\n{d.page_content}")
    return "\n\n".join(parts)


def answer(question: str) -> str:
    """检索 -> 拼上下文 -> 生成。"""
    docs = get_retriever().invoke(question)
    return (RAG_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"context": format_docs(docs), "question": question}
    )


if __name__ == "__main__":
    print(answer("SmallRig 补光灯续航多久？"))
