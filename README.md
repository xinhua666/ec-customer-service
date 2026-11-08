#电子客户服务

跨境电商智能客服与运营助手 —— 基于 LangChain 1.x + LangGraph 的实战项目。

## 功能
1. **RAG 知识库问答**：产品手册/售后政策检索（向量+BM25 混合检索 + 重排），答案带引用来源
2. **订单/物流查询**：Function Calling 工具调用
3. **退换货审核流**：LangGraph 状态图（规则校验 → 风险评估 → Human-in-the-loop）
4. **效果评估**：LangSmith 追踪 + 自建评测集

## 快速开始
```猛击
PIP安装-r要求.文本
cp .env.example .env   # 填入 API Key

Python-m src.core.demo_chain#基础链
python -m src.retrieval.kb build       # 建知识库
python -m src.retrieval.rag_chain # 问答 的时间
python -m src.tools.shop_tools         # 生成模拟订单库
python-m src.agents.service_agent#客服代理 客服代理
Python-m src.graph.refund_flow#退换货审核流
