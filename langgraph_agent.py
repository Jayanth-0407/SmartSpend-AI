import os
from datetime import datetime, timedelta
from collections import defaultdict
from typing import TypedDict, List, Optional
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from groq import Groq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

class GraphState(TypedDict):
    transactions: List[dict]        
    total_income: float             
    
    category_totals: dict           # {"Food Delivery": 18400, "Shopping": 6200}
    category_pct: dict              # {"Food Delivery": 21.6, "Shopping": 7.3}
    top_merchants: List[dict]       # top 5 merchants by spend

    patterns: List[str]             # human-readable insight strings
    zombie_subs: List[str]          # subscription merchants not seen in 60+ days
    post_salary_spike: bool         # True if spending spikes days 1–5 of month

    benchmark_flags: List[str]      # categories above peer benchmark
    savings_gap: float              # how far below 20% savings target

    coach_report: Optional[str]     # final LLM-written financial brief


#To find top 5 merchants by total spend.

def categorize_node(state: GraphState):
    transactions = state["transactions"]
    income = state["total_income"] or 85000  # fallback to 85k if not provided

    #Sum amounts per category
    totals: dict = defaultdict(float)
    merchant_totals: dict = defaultdict(float)

    for txn in transactions:
        amount = txn.get("amount", 0)
        category = txn.get("category", "Other")
        merchant = txn.get("merchant", "Unknown")

        if amount > 0:
            totals[category] += amount
            merchant_totals[merchant] += amount

    totals = dict(totals)

    category_pct = {
        cat: round(total / income * 100, 1)
        for cat, total in totals.items()
    }

    #Top 5 merchants by spend
    top_merchants = sorted(
        [{"merchant": m, "total": round(t, 2)} for m, t in merchant_totals.items()],
        key=lambda x: x["total"],
        reverse=True
    )[:5]

    return {
        **state,
        "category_totals": {k: round(v, 2) for k, v in totals.items()},
        "category_pct": category_pct,
        "top_merchants": top_merchants,
    }


# pattern_node
# Detect behavioral patterns Weekend spending spike, Zombie subscriptions,
# Post-salary shopping spike (days 1–5 of month)


def pattern_node(state: GraphState):
    transactions = state["transactions"]
    patterns: List[str] = []
    zombie_subs: List[str] = []
    post_salary_spike = False

    #Weekend vs weekday spend
    weekend_total = 0.0
    weekday_total = 0.0

    for txn in transactions:
        amount = txn.get("amount", 0)
        if amount <= 0:
            continue  # skip income transactions
        try:
            day = datetime.fromisoformat(str(txn["date"])).weekday()
            if day >= 5:  # Saturday=5, Sunday=6
                weekend_total += amount
            else:
                weekday_total += amount
        except (ValueError, KeyError):
            continue

    if weekday_total > 0:
        weekend_daily_avg = weekend_total / 2
        weekday_daily_avg = weekday_total / 5
        if weekend_daily_avg > weekday_daily_avg * 2:
            patterns.append(
                f"Weekend spending (₹{weekend_total:.0f} total) is over 2x "
                f"your weekday daily average — consider planning weekend budgets."
            )

    #Zombie subscription

    sub_transactions = [
        t for t in transactions
        if t.get("category", "").lower() in ("subscription", "subscriptions")
        and t.get("amount", 0) > 0
    ]

    if sub_transactions:
        all_dates = []
        for t in transactions:
            try:
                all_dates.append(datetime.fromisoformat(str(t["date"])))
            except (ValueError, KeyError):
                continue
        reference_date = max(all_dates) if all_dates else datetime.now()
        cutoff = reference_date - timedelta(days=60)

        sub_by_merchant: dict = defaultdict(list)
        for t in sub_transactions:
            try:
                sub_by_merchant[t["merchant"]].append(
                    datetime.fromisoformat(str(t["date"]))
                )
            except (ValueError, KeyError):
                continue

        for merchant, dates in sub_by_merchant.items():
            if max(dates) < cutoff:
                zombie_subs.append(merchant)

        if zombie_subs:
            patterns.append(
                f"Zombie subscriptions detected — you are paying for "
                f"{', '.join(zombie_subs)} but haven't used them in 60+ days."
            )

    # Post salary spike
    early_month_total = 0.0
    rest_of_month_total = 0.0

    for txn in transactions:
        amount = txn.get("amount", 0)
        if amount <= 0:
            continue
        try:
            day_of_month = datetime.fromisoformat(str(txn["date"])).day
            if day_of_month <= 5:
                early_month_total += amount
            else:
                rest_of_month_total += amount
        except (ValueError, KeyError):
            continue

    # Flag if first 5 days account for > 30% of total spend
    total_spend = early_month_total + rest_of_month_total
    if total_spend > 0 and (early_month_total / total_spend) > 0.30:
        post_salary_spike = True
        patterns.append(
            f"Post-salary spending spike — ₹{early_month_total:.0f} "
            f"({early_month_total/total_spend*100:.0f}% of your monthly spend) "
            f"happens in the first 5 days after salary credit."
        )

    return {
        **state,
        "patterns": patterns,
        "zombie_subs": zombie_subs,
        "post_salary_spike": post_salary_spike,
    }

# benchmark_node
# Compares user's category_pct against hardcoded benchmarks
BENCHMARKS = {
    "Food Delivery":  10,   # Swiggy, Zomato
    "Groceries":      8,    # BigBasket, DMart, local kiranas
    "Dining Out":     6,    # restaurants, cafes
    "Shopping":       8,    # Amazon, Flipkart, apparel
    "Subscription":   2,    # Netflix, Spotify, etc.
    "Transport":      8,    # Uber, Ola, petrol, metro
    "Entertainment":  4,    # movies, events, gaming
    "Healthcare":     3,    # pharmacy, doctor visits
}

SAVINGS_TARGET_PCT = 20  #minimum recommended savings % of income


def benchmark_node(state: GraphState):
    category_pct = state["category_pct"]
    income = state["total_income"] or 85000
    benchmark_flags: List[str] = []

    for category, benchmark in BENCHMARKS.items():
        actual = category_pct.get(category, None)
        if actual is None:
            for key in category_pct:
                if category.lower() in key.lower() or key.lower() in category.lower():
                    actual = category_pct[key]
                    break

        if actual is not None and actual > benchmark + 5:
            overspend_amount = round((actual - benchmark) / 100 * income)
            benchmark_flags.append(
                f"{category}: you spend {actual}% vs peer benchmark of {benchmark}% "
                f"— that's ₹{overspend_amount:,} extra per month."
            )

    #To Compute savings gap
    total_spent_pct = sum(
        pct for cat, pct in category_pct.items()
        if cat.lower() != "savings"
    )
    estimated_savings_pct = max(0, 100 - total_spent_pct)
    savings_gap = round(
        max(0, SAVINGS_TARGET_PCT - estimated_savings_pct), 1
    )

    if savings_gap > 0:
        savings_gap_amount = round(savings_gap / 100 * income)
        benchmark_flags.append(
            f"Savings: you are saving ~{estimated_savings_pct:.1f}% vs the "
            f"recommended {SAVINGS_TARGET_PCT}% — "
            f"₹{savings_gap_amount:,}/month gap to close."
        )

    return {
        **state,
        "benchmark_flags": benchmark_flags,
        "savings_gap": savings_gap,
    }


#coach_node
#To call the LLM

def coach_node(state: GraphState):
    top_merchants_str = ", ".join(
        f"{m['merchant']} (₹{m['total']:,.0f})"
        for m in state.get("top_merchants", [])
    ) or "Not available"

    patterns_str = (
        "\n".join(f"- {p}" for p in state["patterns"])
        if state["patterns"]
        else "- No unusual patterns detected"
    )

    zombie_str = (
        ", ".join(state["zombie_subs"])
        if state["zombie_subs"]
        else "None"
    )

    flags_str = (
        "\n".join(f"- {f}" for f in state["benchmark_flags"])
        if state["benchmark_flags"]
        else "- All categories within peer benchmarks"
    )

    category_str = "\n".join(
        f"  {cat}: ₹{amt:,.0f} ({state['category_pct'].get(cat, 0)}% of income)"
        for cat, amt in sorted(
            state["category_totals"].items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    prompt = f"""You are a financial coach. You have access to the CURRENT upload 
    and the HISTORY of this user's account.

    CRITICAL INSTRUCTION FOR FIRST INTERACTION:
    Whenever the user initiates the conversation (e.g., says "hi", "hello", or asks their first question about a newly uploaded statement), you MUST begin your response with exactly this phrase:
    
    "Hi! I'm your Smart Spend Coach. Ask me anything about your uploaded statement!\n\n"
    
    After providing this exact greeting, proceed to give the financial insights and breakdowns based on the user's data.
    
    Analyze the spending data below and write a Weekly Financial Brief.

    If there is history in the state, compare today's data to the past.

    RULES:
    - Maximum 160 words for the brief
    - Be specific — use the exact rupee amounts and percentages provided
    - Tone: supportive but honest, like a smart friend who knows finance
    - End with exactly 3 numbered action items, each under 20 words
    - Do NOT give generic advice — every sentence must reference the actual data

    ─── SPENDING DATA ───────────────────────────────────
    Monthly Income: ₹{state['total_income']:,.0f}

    Category Breakdown:
    {category_str}

    Top 5 Merchants:
    {top_merchants_str}

    ─── BEHAVIORAL PATTERNS ─────────────────────────────
    {patterns_str}

    ─── ZOMBIE SUBSCRIPTIONS ────────────────────────────
    {zombie_str}

    ─── BENCHMARK FLAGS ─────────────────────────────────
    {flags_str}
    ─────────────────────────────────────────────────────

    Write the brief now:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,   #creativity
    )

    coach_report = response.choices[0].message.content.strip()

    return {
        **state,
        "coach_report": coach_report,
    }


#Building a Linear StateGraph

def build_graph():
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("categorize", categorize_node)
    workflow.add_node("patterns",   pattern_node)
    workflow.add_node("benchmark",  benchmark_node)
    workflow.add_node("coach",      coach_node)

    # Wire edges — left to right, one direction
    workflow.set_entry_point("categorize")
    workflow.add_edge("categorize","patterns")
    workflow.add_edge("patterns","benchmark")
    workflow.add_edge("benchmark","coach")
    workflow.add_edge("coach",END)

    conn = sqlite3.connect("coach_memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)

graph = build_graph()

#bridge between the langgraph brain and fastapi server

def run_financial_coach(    
        transactions: list[dict],
        user_id:str,
        total_income: float = 85000.0
    ):
    
    initial_state: GraphState = {
        "transactions":    transactions,
        "total_income":    total_income,
        # keeping all other fields start as None/empty because nodes will fill them
        "category_totals": {},
        "category_pct":    {},
        "top_merchants":   [],
        "patterns":        [],
        "zombie_subs":     [],
        "post_salary_spike": False,
        "benchmark_flags": [],
        "savings_gap":     0.0,
        "coach_report":    None,
    }

    config={"configurable":{"thread_id":user_id}}

    final_state = graph.invoke(initial_state,config=config)

    return {
        "coach_report":    final_state["coach_report"],
        "patterns":        final_state["patterns"],
        "benchmark_flags": final_state["benchmark_flags"],
        "zombie_subs":     final_state["zombie_subs"],
        "category_totals": final_state["category_totals"],
        "category_pct":    final_state["category_pct"],
        "top_merchants":   final_state["top_merchants"],
        "savings_gap":     final_state["savings_gap"],
    }
