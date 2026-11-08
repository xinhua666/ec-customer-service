"""Step 5: LangGraph 退换货审核流（含 Human-in-the-loop 人工介入）。

流程:
  意图识别 -> 信息完整性检查 -> 规则校验 -> 风险评估
                                       |-> 通过: 生成退货单
                                       |-> 高风险: interrupt 暂停，转人工
运行: python -m src.graph.refund_flow
"""
import json
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.core.llm import get_llm


class RefundState(TypedDict):
    user_msg: str          # 用户原始诉求
    intent: str            # 意图: refund / exchange / other
    order_id: str
    reason: str
    check_result: str      # 规则校验结论
    risk: str              # low / high
    reply: str             # 最终回复


def detect_intent(state: RefundState) -> dict:
    llm = get_llm(temperature=0)
    resp = llm.invoke([
        ("system", "判断用户售后意图，只输出一个词: refund / exchange / other"),
        ("human", state["user_msg"]),
    ])
    return {"intent": resp.content.strip()}


def collect_info(state: RefundState) -> dict:
    """提取订单号与原因；缺失则追问。"""
    llm = get_llm(temperature=0)
    resp = llm.invoke([
        ("system",
         '从用户消息中提取订单号和售后原因，输出 JSON: '
         '{"order_id": "...", "reason": "..."}，缺失填 null'),
        ("human", state["user_msg"]),
    ])
    info = json.loads(resp.content.strip().strip("`json"))
    if not info.get("order_id") or not info.get("reason"):
        return {"reply": "请提供您的订单号和售后原因，我来为您处理~", "intent": "need_more"}
    return {"order_id": info["order_id"], "reason": info["reason"]}


def rule_check(state: RefundState) -> dict:
    """规则校验：签收 7 天内可退（真实项目此处查订单库）。"""
    return {"check_result": "通过：签收 5 天，符合 7 天无理由退货政策"}


def risk_check(state: RefundState) -> dict:
    llm = get_llm(temperature=0)
    resp = llm.invoke([
        ("system",
         "根据售后原因评估风险，只输出 low 或 high。"
         "高频刷单嫌疑、金额>1000、恶意措辞 = high，其余 low。"),
        ("human", state["reason"]),
    ])
    return {"risk": resp.content.strip()}


def approve(state: RefundState) -> dict:
    return {"reply": f"✅ 审核通过（{state['check_result']}），退货单已生成，"
                     f"退货地址已发送到您的邮箱。"}


def manual_review(state: RefundState) -> dict:
    """高风险订单：暂停流程，等待人工审核决定。"""
    decision = interrupt({
        "question": f"订单 {state['order_id']} 风险等级高，请人工审核",
        "reason": state["reason"],
    })
    if decision == "approve":
        return {"reply": "人工审核通过，退货单已生成。"}
    return {"reply": "经人工复核，本单不符合退货条件，感谢理解。"}


def route_risk(state: RefundState) -> str:
    return "approve" if state["risk"] == "low" else "manual"


def route_intent(state: RefundState) -> str:
    if state["intent"] in ("refund", "exchange"):
        return "collect_info"
    return END  # other/need_more: 交给通用客服或结束


# ---- 组装状态图 ----
g = StateGraph(RefundState)
g.add_node("detect_intent", detect_intent)
g.add_node("collect_info", collect_info)
g.add_node("rule_check", rule_check)
g.add_node("risk_check", risk_check)
g.add_node("approve", approve)
g.add_node("manual_review", manual_review)

g.add_edge(START, "detect_intent")
g.add_conditional_edges("detect_intent", route_intent,
                        {"collect_info": "collect_info", END: END})
g.add_edge("collect_info", "rule_check")
g.add_edge("rule_check", "risk_check")
g.add_conditional_edges("risk_check", route_risk,
                        {"approve": "approve", "manual": "manual_review"})
g.add_edge("approve", END)
g.add_edge("manual_review", END)

app = g.compile()

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-1"}}
    result = app.invoke(
        {"user_msg": "订单 SO2026002 的稳定器有点划痕，想退货，用了三天"},
        config=config,
    )
    print(result.get("reply") or result)
