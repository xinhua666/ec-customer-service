"""Step 4: 工具调用 —— 订单/物流查询工具 + Agent。

先运行 python -m src.tools.shop_tools 生成模拟数据库。
"""
import sqlite3

from langchain_core.tools import tool

DB_PATH = "data/mock_shop.db"


def init_db():
    """生成模拟订单/物流数据（一次性）。"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript(
        """
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS logistics;
        CREATE TABLE orders(
            order_id TEXT PRIMARY KEY, user_id TEXT, product TEXT,
            amount REAL, currency TEXT, status TEXT, created_at TEXT);
        CREATE TABLE logistics(
            order_id TEXT, step TEXT, location TEXT, time TEXT);
        """
    )
    sample_orders = [
        ("SO2026001", "u001", "SmallRig 补光灯 P200", 399.0, "CNY", "已发货", "2026-10-20"),
        ("SO2026002", "u002", "手机稳定器 S60",  799.0, "CNY", "已签收", "2026-10-15"),
        ("SO2026003", "u003", "相机提笼套件",    259.0, "USD", "处理中", "2026-10-28"),
    ]
    c.executemany("INSERT INTO orders VALUES(?,?,?,?,?,?,?)", sample_orders)
    c.executemany(
        "INSERT INTO logistics VALUES(?,?,?,?)",
        [
            ("SO2026001", "已揽收", "深圳转运中心", "2026-10-21 09:00"),
            ("SO2026001", "运输中", "广州白云机场", "2026-10-22 14:30"),
            ("SO2026001", "国际段", "起飞飞往洛杉矶", "2026-10-23 22:00"),
        ],
    )
    conn.commit()
    conn.close()
    print("模拟数据已生成 ->", DB_PATH)


def _query(sql: str, params: tuple) -> str:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return str(rows) if rows else "未查询到记录"


# ---- 工具定义：docstring 会成为给模型的工具描述，务必写清楚 ----

@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态、商品、金额。订单号形如 SO2026001。"""
    return _query("SELECT * FROM orders WHERE order_id=?", (order_id,))


@tool
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流轨迹（揽收、运输、国际段、签收等节点）。"""
    return _query(
        "SELECT step, location, time FROM logistics WHERE order_id=? ORDER BY time",
        (order_id,),
    )


TOOLS = [query_order, query_logistics]

if __name__ == "__main__":
    init_db()
