"""Step 3: RAG —— 文档加载、切分、入库、检索。

数据准备：把产品文档（md/pdf/txt）放进 data/docs/
运行:  python -m src.retrieval.kb build     # 建库
      python -m src.retrieval.kb "补光灯支持什么接口"   # 测试检索
"""
import glob

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_API_BASE, EMBEDDING_API_KEY,
    EMBEDDING_MODEL, TOP_K,
)

DB_DIR = "data/chroma_db"


def get_embeddings() -> OpenAIEmbeddings:
    """Embedding 模型（智谱/阿里均兼容 OpenAI 协议）。"""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=EMBEDDING_API_KEY,
        openai_api_base=EMBEDDING_API_BASE,
    )


def load_docs():
    """加载 data/docs/ 下所有文档。"""
    docs = []
    for path in glob.glob("data/docs/*"):
        if path.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        else:
            docs += TextLoader(path, encoding="utf-8").load()
    return docs


def build():
    """切分 -> 向量化 -> 入库（每次全量重建，简单可靠）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(load_docs())
    print(f"共切分出 {len(chunks)} 个片段")

    Chroma.from_documents(chunks, get_embeddings(), persist_directory=DB_DIR)
    print(f"已入库 -> {DB_DIR}")


def get_retriever(k: int = TOP_K):
    """检索器：其他模块统一从这里拿。"""
    db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    return db.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "SmallRig 补光灯怎么充电"
        for i, doc in enumerate(get_retriever().invoke(q), 1):
            print(f"\n--- 片段 {i} (来源: {doc.metadata.get('source')}) ---")
            print(doc.page_content[:200])
