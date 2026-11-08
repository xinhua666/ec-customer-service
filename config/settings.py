import os
from dotenv import load_dotenv

load_dotenv()

# LLM 配置
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Embedding 配置
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")

# 检索参数（调优时改这里）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
