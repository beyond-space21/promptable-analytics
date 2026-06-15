from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool, QueryEngineTool
import asyncio
import re
import clickhouse_connect

# ========================= CONFIG =========================
CH_HOST = "localhost"
CH_PORT = 8123
CH_USER = "admin"
CH_PASSWORD = "hiffiofsuperlabs"
# RAG knowledge base (vector store — docs only)
RAG_DATABASE = "rag_knowledge"
CH_TABLE = "clickhouse_docs_v1"
CHARTJS_TABLE = "chartjs_docs_v1"

# Target database/table for generated SQL queries
QUERY_DATABASE = "analytics"
QUERY_TABLE = "request_logs"  # optional: set e.g. "events" to hint the default table

# Query limits — keep ClickHouse + UI responsive
SQL_MAX_ROWS = 100
SQL_DISPLAY_ROWS = 12
SQL_EXECUTION_TIMEOUT_SEC = 30
UI_TABLE_MAX_ROWS = 20

# Agent Config
AGENT_MAX_ITERATIONS = 25
FAST_PATH_ENABLED = True

# Query intent: chart | analytical | simple_metric
_CHART_SIGNALS = (
    "chart", "graph", "plot", "pie", "bar chart", "line chart", "histogram",
    "visualiz", "display as", "draw", "doughnut", "donut", "polar area",
    "as a bar", "as a line", "as a pie",
)
_ANALYTICAL_PATTERNS = (
    r"\bhow\b",
    r"\bwhy\b",
    r"\bexplain\b",
    r"\bwhat drives\b",
    r"\bwhat causes\b",
    r"\bconsuming\b",
    r"\bconsume\b",
    r"\bbreak down\b",
    r"\banalyse\b",
    r"\banalyze\b",
    r"\bunderstand\b",
    r"\binsight\b",
    r"\broot cause\b",
    r"\bcontribut",
    r"\bdriving\b",
    r"\baccount for\b",
    r"\bwhere is\b.*\btime\b",
    r"\bwhere does\b",
)

# LLM Config
LLM_MODEL = "gpt-4o"
AUX_LLM_MODEL = "gpt-4o-mini"  # fallback SQL + summaries
EMBED_MODEL = "text-embedding-3-large"

# OpenAI Pricing (update as needed - June 2026 rates)
PRICING = {
    "gpt-4o": {
        "input": 2.50,      # $ per 1M tokens
        "output": 10.00
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60
    },
    # Add more models as needed
}

# =========================================================

llm = OpenAI(model=LLM_MODEL, temperature=0.0)
aux_llm = OpenAI(model=AUX_LLM_MODEL, temperature=0.0)

embed_model = OpenAIEmbedding(model=EMBED_MODEL)

# Connect to vector store
client = clickhouse_connect.get_client(
    host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD
)

vector_store = ClickHouseVectorStore(
    clickhouse_client=client,
    table=CH_TABLE,
    database=RAG_DATABASE,
    dimension=3072 if "large" in EMBED_MODEL else 1536,
)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

# =============== TOOLS ===============
def get_ch_client(database: str = QUERY_DATABASE):
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=database,
    )

def get_table_schema(table_name: str) -> str:
    conn = get_ch_client()
    result = conn.query(f"DESCRIBE TABLE {QUERY_DATABASE}.{table_name}")
    return str(result.result_rows)

def list_tables() -> str:
    conn = get_ch_client()
    result = conn.query(f"SHOW TABLES FROM {QUERY_DATABASE}")
    return str(result.result_rows)

def validate_query(sql: str) -> str:
    conn = get_ch_client()
    try:
        result = conn.query(f"EXPLAIN {_apply_sql_row_cap(sql)}")
        return "Valid ✓\n" + str(result.result_rows)
    except Exception as e:
        return f"Invalid: {str(e)}"


def _apply_sql_row_cap(sql: str, max_rows: int = SQL_MAX_ROWS) -> str:
    """Wrap any SELECT so ClickHouse never returns more than max_rows."""
    sql = sql.strip().rstrip(";")
    return f"SELECT * FROM ({sql}) AS _limited LIMIT {max_rows}"


def _ch_query_settings() -> dict[str, str | int]:
    return {
        "max_execution_time": SQL_EXECUTION_TIMEOUT_SEC,
        "max_result_rows": str(SQL_MAX_ROWS),
        "result_overflow_mode": "break",
    }


def _format_rows_for_display(rows: list, total_rows: int | None = None) -> str:
    """Compact string for agent/UI — never dump huge result sets."""
    shown = rows[:SQL_DISPLAY_ROWS]
    text = str(shown)
    total = total_rows if total_rows is not None else len(rows)
    if total > len(shown):
        text += f"\n... ({total} rows total, showing {len(shown)})"
    if len(text) > 900:
        text = text[:900] + "\n... (truncated)"
    return text


def _format_rows_for_display_from_text(text: str) -> str:
    if not text or text.startswith("Error:"):
        return text
    if len(text) <= 900:
        return text
    return text[:900] + "\n... (truncated for display)"


def execute_sql(sql: str) -> str:
    conn = get_ch_client()
    try:
        safe_sql = _apply_sql_row_cap(sql)
        result = conn.query(safe_sql, settings=_ch_query_settings())
        rows = result.result_rows
        return _format_rows_for_display(rows, total_rows=len(rows))
    except Exception as e:
        return f"Error: {str(e)}"


# Cache schema so the agent can skip discovery calls
_TABLE_SCHEMA = get_table_schema(QUERY_TABLE) if QUERY_TABLE else ""

FALLBACK_SQL_PROMPT = """Write exactly ONE ClickHouse SELECT query for the user's question.

Table: `{db}`.`{table}`
Schema:
{schema}

Question: {query}

Rules:
- Output ONLY raw SQL. No markdown, no explanation.
- For "top N" / "famous" / "popular" / "most active" requests/URLs/endpoints:
  - If the user asks about response time or latency: GROUP BY url, avg(response_time_ms) AS avg_response_time_ms, ORDER BY count() DESC LIMIT N.
  - Otherwise: GROUP BY url, count() AS cnt, ORDER BY cnt DESC LIMIT N.
- For time/frequency: GROUP BY time bucket, ORDER BY bucket LIMIT {sql_max_rows}.
- NEVER SELECT * from raw logs. Always GROUP BY + LIMIT (max {sql_max_rows}).
- Return two columns: a label (url, status, date, etc.) and a numeric metric.
"""


def _extract_sql(text: str) -> str:
    import re

    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    idx = text.upper().find("SELECT")
    if idx >= 0:
        return text[idx:].strip().rstrip(";")
    return text.strip().rstrip(";")


async def _fallback_sql_query(user_query: str) -> tuple[str, str, float, dict[str, int]] | None:
    """Generate and run SQL directly when the agent fails to execute."""
    if not QUERY_TABLE:
        return None

    prompt = FALLBACK_SQL_PROMPT.format(
        db=QUERY_DATABASE,
        table=QUERY_TABLE,
        schema=_TABLE_SCHEMA,
        query=user_query,
        sql_max_rows=SQL_MAX_ROWS,
    )
    response = await aux_llm.acomplete(prompt)
    cost, usage = _cost_and_usage_from_response(response)
    sql = _extract_sql(str(response))
    if not sql.upper().startswith("SELECT"):
        return None

    result = execute_sql(sql)
    if not result or result.startswith("Error:"):
        return None
    return sql, result, cost, usage

_query_target = (
    f"{QUERY_DATABASE}.{QUERY_TABLE}"
    if QUERY_TABLE
    else f"database `{QUERY_DATABASE}`"
)
_agent_system_prompt = f"""You are a ClickHouse analytics agent with live tool access. You MUST run SQL via `execute_sql` — never describe steps the user should take themselves.

## Table (already known — do NOT call list_tables)
`{QUERY_DATABASE}.{QUERY_TABLE}` schema:
{_TABLE_SCHEMA}

## Required workflow
1. Write a SELECT query for the user's question.
2. Call `execute_sql` with that query. Optional: `validate_query` first only if unsure.
3. After execute_sql returns data, give a 1-2 sentence answer citing the results. STOP.

## SQL rules
- "top N" / "N most …" → GROUP BY + ORDER BY metric DESC + LIMIT N.
- "famous/popular/most active requests/URLs/endpoints" → GROUP BY url, count() AS cnt, ORDER BY cnt DESC.
- If user asks for response time/latency on those URLs → GROUP BY url, avg(response_time_ms) AS avg_response_time_ms, ORDER BY count() DESC.
- Return label column + numeric metric column (chart-friendly).
- ALWAYS aggregate (GROUP BY) for analytics. NEVER `SELECT *` on raw logs.
- ALWAYS include LIMIT (max {SQL_MAX_ROWS}). Queries without LIMIT are capped automatically.
- Use `clickhouse_docs` only if a query error needs syntax help.

## Forbidden
- Do NOT explain how to write SQL without calling execute_sql.
- Do NOT say you cannot execute queries — you have execute_sql.
- Do NOT output Chart.js code (charts are rendered automatically).

You must call execute_sql before answering."""

_agent_analytical_prompt = f"""You are a ClickHouse analytics investigator with live tool access. The user asked an analytical question — you MUST explore the data with multiple SQL queries before answering.

## Table (already known — do NOT call list_tables)
`{QUERY_DATABASE}.{QUERY_TABLE}` schema:
{_TABLE_SCHEMA}

## Required workflow
1. Plan what data you need (totals, breakdowns, shares, comparisons, time patterns).
2. Call `execute_sql` for EACH needed query — typically 2-4 queries for analytical questions.
3. After collecting results, write a 3-5 sentence insight citing specific numbers from your queries. STOP.

## Analytical SQL patterns
- Total latency consumed: `sum(response_time_ms)` or `count() * avg(response_time_ms)` grouped by url, status_code, or method.
- Share of total: compute totals then compare top contributors vs rest.
- Time patterns: `toStartOfHour(timestamp)` or `toDate(timestamp)` buckets.
- Slow vs frequent: compare `avg(response_time_ms)` with `count()` for the same dimension.

## SQL rules
- ALWAYS aggregate (GROUP BY). NEVER `SELECT *` on raw logs.
- ALWAYS include LIMIT (max {SQL_MAX_ROWS}).
- Use `clickhouse_docs` only after execute_sql errors.

## Forbidden
- Do NOT stop after a single query if the question needs breakdown or comparison.
- Do NOT describe SQL without calling execute_sql.
- Do NOT say you cannot execute queries.
- Do NOT output Chart.js code (charts are optional and added separately if appropriate).

Investigate thoroughly, then explain what the data shows."""

tools = [
    QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="clickhouse_docs",
        description="Search ClickHouse docs for SQL syntax. Use only after execute_sql errors.",
    ),
    FunctionTool.from_defaults(fn=validate_query),
    FunctionTool.from_defaults(fn=execute_sql),
]

# =============== AGENT ===============
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)

agent = ReActAgent(
    tools=tools,
    llm=llm,
    verbose=False,
    system_prompt=_agent_system_prompt,
    early_stopping_method="generate",
)

analytical_agent = ReActAgent(
    tools=tools,
    llm=llm,
    verbose=False,
    system_prompt=_agent_analytical_prompt,
    early_stopping_method="generate",
)


def classify_query_intent(query: str) -> str:
    """Return chart | analytical | simple_metric."""
    q = query.lower()
    if any(signal in q for signal in _CHART_SIGNALS):
        return "chart"
    if any(re.search(pattern, q) for pattern in _ANALYTICAL_PATTERNS):
        return "analytical"
    return "simple_metric"

def calculate_cost(usage, model: str = LLM_MODEL) -> float:
    """Calculate cost from OpenAI usage object."""
    if not usage:
        return 0.0

    model_pricing = PRICING.get(model, {"input": 2.50, "output": 10.00})
    input_cost = (usage.prompt_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (usage.completion_tokens / 1_000_000) * model_pricing["output"]
    return input_cost + output_cost


def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _usage_dict(usage) -> dict[str, int]:
    if not usage:
        return _empty_usage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _add_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": a["prompt_tokens"] + b["prompt_tokens"],
        "completion_tokens": a["completion_tokens"] + b["completion_tokens"],
        "total_tokens": a["total_tokens"] + b["total_tokens"],
    }


def _usage_from_raw(raw) -> dict[str, int]:
    """Extract token usage from an OpenAI raw response (dict or object)."""
    if raw is None:
        return _empty_usage()
    if isinstance(raw, dict):
        usage = raw.get("usage") or raw
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
            total = int(usage.get("total_tokens", 0) or 0) or (prompt + completion)
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
    usage = getattr(raw, "usage", None)
    if usage is not None:
        return _usage_dict(usage)
    return _empty_usage()


def _model_from_raw(raw, default_model: str = LLM_MODEL) -> str:
    if isinstance(raw, dict):
        return str(raw.get("model") or default_model)
    return str(getattr(raw, "model", default_model) or default_model)


def _cost_from_raw(raw, default_model: str = LLM_MODEL) -> float:
    usage = _usage_from_raw(raw)
    if usage["total_tokens"] <= 0:
        return 0.0
    model = _model_from_raw(raw, default_model)
    pricing_model = AUX_LLM_MODEL if "mini" in model else LLM_MODEL
    class _Usage:
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
    return calculate_cost(_Usage(), pricing_model)


def _cost_and_usage_from_response(response, default_model: str = AUX_LLM_MODEL) -> tuple[float, dict[str, int]]:
    try:
        raw = getattr(response, "raw", None)
        usage = getattr(raw, "usage", None) if raw is not None else None
        if usage is None:
            return 0.0, _empty_usage()
        model = getattr(raw, "model", default_model) or default_model
        pricing_model = AUX_LLM_MODEL if "mini" in str(model) else LLM_MODEL
        return calculate_cost(usage, pricing_model), _usage_dict(usage)
    except Exception:
        return 0.0, _empty_usage()


def _should_use_fast_path(query: str) -> bool:
    if not FAST_PATH_ENABLED:
        return False
    if classify_query_intent(query) == "analytical":
        return False
    q = query.lower()
    if any(w in q for w in ("join", "compare", "correlat", "union", "subquery", "between")):
        return False
    return bool(
        re.search(
            r"\b(top|most|count|chart|active|frequency|status|average|avg|sum|requests|endpoints|urls|famous|popular|latency|response)\b",
            q,
        )
    )


def _message_text(message) -> str:
    if message is None:
        return ""
    if hasattr(message, "content") and message.content:
        return str(message.content)
    return str(message)


def _extract_sql_result_text(event: ToolCallResult) -> str:
    text = (event.tool_output.content or "").strip()
    if text.startswith("Error:"):
        return ""
    return text


def format_event(event, *, truncate: bool = True) -> dict | None:
    if isinstance(event, ToolCall):
        if event.tool_name == "execute_sql" and "sql" in event.tool_kwargs:
            text = str(event.tool_kwargs["sql"]).strip()
        else:
            text = ", ".join(f"{k}={v!r}" for k, v in event.tool_kwargs.items())
        return {
            "type": "tool_call",
            "title": f"Tool call: {event.tool_name}",
            "text": text,
        }

    if isinstance(event, ToolCallResult):
        text = event.tool_output.content or ""
        if event.tool_name == "execute_sql" and truncate:
            text = _format_rows_for_display_from_text(text)
        elif truncate and len(text) > 600:
            text = text[:600] + "\n..."
        return {
            "type": "tool_result",
            "title": f"Tool result: {event.tool_name}",
            "text": text,
        }

    if isinstance(event, AgentOutput):
        text = _message_text(event.response).strip()
        if not text:
            return None
        return {
            "type": "reasoning",
            "title": "Agent reasoning",
            "text": text,
        }

    return None


def _final_result(agent_output) -> dict:
    response_text = _message_text(agent_output.response)
    raw = getattr(agent_output, "raw", None)
    return {
        "response": response_text,
        "cost": _cost_from_raw(raw, LLM_MODEL),
        "tokens": _usage_from_raw(raw),
    }


async def _events_with_heartbeat(handler, interval: float = 2.0):
    """Yield agent events, injecting status pulses when the agent is silent."""
    queue: asyncio.Queue = asyncio.Queue()
    finished = False

    async def pull_events() -> None:
        nonlocal finished
        async for event in handler.stream_events():
            await queue.put(("event", event))
        finished = True
        await queue.put(("done", None))

    async def pulse() -> None:
        elapsed = 0
        while not finished:
            await asyncio.sleep(interval)
            if finished:
                break
            elapsed += interval
            await queue.put(("pulse", elapsed))

    puller = asyncio.create_task(pull_events())
    pulser = asyncio.create_task(pulse())

    try:
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                break
            if kind == "pulse":
                yield {"kind": "status", "text": f"Working… ({int(payload)}s)"}
            else:
                yield {"kind": "event", "event": payload}
    finally:
        pulser.cancel()
        await puller


async def _run_fast_path(query: str) -> tuple[list[dict], list[str], str, float, dict[str, int], int] | None:
    """Direct SQL for simple queries — skips the ReAct agent entirely."""
    fallback = await _fallback_sql_query(query)
    if not fallback:
        return None
    sql, sql_result, cost, usage = fallback
    display = _format_rows_for_display_from_text(sql_result)
    steps = [
        {"type": "tool_call", "title": "Tool call: execute_sql", "text": sql},
        {
            "type": "tool_result",
            "title": "Tool result: execute_sql",
            "text": display,
        },
    ]
    return steps, [sql_result], "Query executed.", cost, usage, 1


async def run_query_stream(query: str):
    """Yield step dicts as they happen, then a final done dict."""
    from visualization import build_visualization

    query_intent = classify_query_intent(query)
    steps: list[dict] = []
    sql_results: list[str] = []
    extra_cost = 0.0
    total_tokens = _empty_usage()
    llm_calls = 0
    query_path = "agent"

    if _should_use_fast_path(query):
        yield {"kind": "status", "text": "Running query…"}
        fast = await _run_fast_path(query)
        if fast:
            steps, sql_results, response_text, fast_cost, fast_tokens, llm_calls = fast
            for step in steps:
                yield {"kind": "step", "step": step}
            query_path = "fast"
            result = {
                "response": response_text,
                "cost": fast_cost,
                "tokens": fast_tokens,
                "llm_calls": llm_calls,
                "query_path": query_path,
                "query_intent": query_intent,
            }
        else:
            fast = None
    else:
        fast = None

    if not (_should_use_fast_path(query) and fast):
        if query_intent == "analytical":
            query_path = "analytical"
            yield {"kind": "status", "text": "Investigating…"}
            active_agent = analytical_agent
        else:
            yield {"kind": "status", "text": "Thinking…"}
            active_agent = agent

        handler = active_agent.run(
            user_msg=query,
            max_iterations=AGENT_MAX_ITERATIONS,
            early_stopping_method="generate",
        )

        agent_cost = 0.0
        try:
            async for item in _events_with_heartbeat(handler):
                if item["kind"] == "status":
                    yield item
                    continue
                event = item["event"]
                if isinstance(event, AgentOutput):
                    usage = _usage_from_raw(event.raw)
                    if usage["total_tokens"] > 0:
                        llm_calls += 1
                        agent_cost += _cost_from_raw(event.raw, LLM_MODEL)
                        total_tokens = _add_usage(total_tokens, usage)

                if isinstance(event, ToolCallResult) and event.tool_name == "execute_sql":
                    full_text = _extract_sql_result_text(event)
                    if full_text:
                        sql_results.append(full_text)

                step = format_event(event, truncate=True)
                if step:
                    steps.append(step)
                    yield {"kind": "step", "step": step}

            agent_output = await handler
        except Exception as exc:
            if not sql_results:
                raise
            agent_output = type("PartialOutput", (), {"response": str(exc), "raw": None})()

        final_usage = _usage_from_raw(getattr(agent_output, "raw", None))
        if final_usage["total_tokens"] > 0 and total_tokens["total_tokens"] == 0:
            total_tokens = final_usage
            agent_cost = _cost_from_raw(getattr(agent_output, "raw", None), LLM_MODEL)
            llm_calls = max(llm_calls, 1)

        result = _final_result(agent_output)

        if not sql_results:
            yield {"kind": "status", "text": "Running direct SQL fallback…"}
            fallback = await _fallback_sql_query(query)
            if fallback:
                sql, sql_result, fb_cost, fb_tokens = fallback
                extra_cost += fb_cost
                total_tokens = _add_usage(total_tokens, fb_tokens)
                if fb_tokens["total_tokens"] > 0:
                    llm_calls += 1
                sql_results.append(sql_result)
                steps.append({
                    "type": "tool_call",
                    "title": "Tool call: execute_sql (fallback)",
                    "text": sql,
                })
                steps.append({
                    "type": "tool_result",
                    "title": "Tool result: execute_sql (fallback)",
                    "text": _format_rows_for_display_from_text(sql_result),
                })
                yield {"kind": "step", "step": steps[-2]}
                yield {"kind": "step", "step": steps[-1]}
                result["response"] = "Query executed via fallback."

        result["cost"] = agent_cost + extra_cost
        result["tokens"] = total_tokens
        result["llm_calls"] = llm_calls
        result["query_path"] = query_path
        result["query_intent"] = query_intent
    elif "tokens" not in result:
        result["tokens"] = _empty_usage()
        result["query_intent"] = query_intent

    if query_intent == "analytical":
        yield {"kind": "status", "text": "Synthesizing insight…"}
    else:
        yield {"kind": "status", "text": "Building visualization…"}

    visualization, viz_cost, viz_tokens = await build_visualization(
        user_query=query,
        agent_response=result["response"],
        steps=steps,
        llm=aux_llm,
        sql_results=sql_results,
        query_intent=query_intent,
    )
    result["cost"] = result.get("cost", 0.0) + viz_cost
    result["tokens"] = _add_usage(result.get("tokens", _empty_usage()), viz_tokens)
    if viz_tokens.get("total_tokens", 0) > 0:
        result["llm_calls"] = result.get("llm_calls", 0) + 1

    yield {"kind": "done", **result, "visualization": visualization}


async def run_query(query: str) -> dict:
    steps: list[dict] = []
    result: dict | None = None

    async for event in run_query_stream(query):
        if event["kind"] == "step":
            steps.append(event["step"])
        else:
            result = event

    assert result is not None
    return {
        "response": result["response"],
        "cost": result["cost"],
        "tokens": result.get("tokens", _empty_usage()),
        "llm_calls": result.get("llm_calls"),
        "query_path": result.get("query_path"),
        "query_intent": result.get("query_intent"),
        "steps": steps,
        "visualization": result.get("visualization"),
    }


# ===================== MAIN =====================
async def main() -> None:
    print(f"🚀 ClickHouse NL2SQL Agent Started (Model: {LLM_MODEL})")
    print(f"💰 Cost tracking enabled\n")

    total_cost = 0.0

    while True:
        query = input("\n🔍 Enter natural language query (or 'exit'): ")
        if query.lower() in ["exit", "quit", "q"]:
            print(f"\n📊 Total session cost: ${total_cost:.6f}")
            break

        print("Thinking...")
        result = await run_query(query)
        for step in result["steps"]:
            print(f"\n--- {step['title']} ---")
            if step["text"]:
                print(step["text"])
        response_text = result["response"]
        cost = result["cost"]

        total_cost += cost

        print("\n" + "=" * 80)
        print("✅ Final Response:")
        print(response_text)
        print("=" * 80)
        print(f"💰 Query Cost: ${cost:.6f}")
        print(f"📈 Session Total Cost: ${total_cost:.6f}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())