import ast
import logging
import re
from datetime import date, datetime
from typing import Any

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CHART_PALETTE = [
    "rgba(79, 140, 255, 0.75)",
    "rgba(62, 207, 142, 0.75)",
    "rgba(240, 160, 48, 0.75)",
    "rgba(255, 107, 107, 0.75)",
    "rgba(168, 85, 247, 0.75)",
    "rgba(45, 212, 191, 0.75)",
    "rgba(251, 113, 133, 0.75)",
    "rgba(250, 204, 21, 0.75)",
]

SUMMARY_PROMPT = PromptTemplate(
    """Write a 2-3 sentence analytical summary of these query results for a business user.

Question: {user_query}
Answer: {agent_response}
Data: {sql_data}

Be concise. Highlight the main insight, top value, or trend. No bullet points."""
)


_META_PHRASES = (
    "cannot execute",
    "unfortunately",
    "steps involved",
    "you would take",
    "query executed",
    "fallback",
    "identify the table",
)


def _is_good_agent_summary(text: str) -> bool:
    if not text or len(text) > 450:
        return False
    lower = text.lower()
    if any(p in lower for p in _META_PHRASES):
        return False
    return len(text.split()) >= 6


def _summary_from_rows(rows: list[list[Any]] | None, user_query: str) -> str:
    if not rows:
        return ""
    pairs = _extract_label_value_pairs(rows, row_limit=3)
    if not pairs:
        return ""
    labels, values = pairs
    top_label, top_val = labels[0], values[0]
    parts = [f"The top result is {top_label} with {top_val:,.0f}."]
    if len(labels) > 1:
        parts.append(f"Next is {labels[1]} ({values[1]:,.0f}).")
    if "pie" in user_query.lower() or "share" in user_query.lower():
        total = sum(values)
        if total:
            parts.append(f"The leader holds {100 * top_val / total:.1f}% of the total shown.")
    return " ".join(parts)


class SummaryResult(BaseModel):
    summary: str = Field(description="2-3 sentence analytical summary")


def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _usage_from_llm_response(response) -> tuple[float, dict[str, int]]:
    try:
        raw = getattr(response, "raw", None)
        usage = getattr(raw, "usage", None) if raw else None
        if not usage:
            return 0.0, _empty_usage()
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
        cost = (prompt * 0.15 + completion * 0.60) / 1_000_000
        return cost, {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
    except Exception:
        return 0.0, _empty_usage()


def extract_sql_results(steps: list[dict]) -> list[str]:
    results: list[str] = []
    for step in steps:
        if step.get("type") != "tool_result":
            continue
        title = step.get("title", "")
        if "execute_sql" not in title:
            continue
        text = (step.get("text") or "").strip()
        if text and not text.startswith("Error:"):
            results.append(text)
    return results


def parse_sql_rows(text: str) -> list[list[Any]] | None:
    if not text:
        return None
    body = text.split("\n...")[0].strip()

    rows = _parse_datetime_rows(body)
    if rows:
        return rows

    try:
        parsed = ast.literal_eval(body)
    except (SyntaxError, ValueError):
        parsed = None

    if parsed is None:
        parsed_rows = _parse_rows_fallback(body)
        if parsed_rows:
            return parsed_rows
        return _parse_datetime_rows(text)

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], (list, tuple)):
        return [list(row) for row in parsed]
    if isinstance(parsed, list):
        return [[item] for item in parsed]
    return None


def _parse_datetime_rows(text: str) -> list[list[Any]] | None:
    """Extract (datetime, value) pairs even from truncated SQL result strings."""
    pattern = re.compile(
        r"datetime\.datetime\((\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+),\s*(\d+))?(?:,\s*(\d+))?\),\s*(\d+)\)"
    )
    rows: list[list[Any]] = []
    for match in pattern.finditer(text):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        hour = int(match.group(4) or 0)
        minute = int(match.group(5) or 0)
        value = int(match.group(7))
        label = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
        rows.append([label, value])
    return rows if len(rows) >= 2 else None


def _parse_rows_fallback(text: str) -> list[list[Any]] | None:
    """Handle str(rows) that includes datetime(...) literals from ClickHouse."""
    cleaned = re.sub(
        r"datetime\.(?:datetime|date)\([^)]+\)",
        "'DATE'",
        text,
    )
    try:
        parsed = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], (list, tuple)):
        return [list(row) for row in parsed]
    return None


def infer_columns(rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    if width == 1:
        return ["value"]
    if width == 2:
        return ["label", "value"]
    return [f"col_{i + 1}" for i in range(width)]


def rows_to_table(rows: list[list[Any]] | None, max_rows: int = 20) -> dict | None:
    if not rows:
        return None
    columns = infer_columns(rows)
    return {
        "columns": columns,
        "rows": [[_cell(v) for v in row] for row in rows[:max_rows]],
        "truncated": len(rows) > max_rows,
        "total_rows": len(rows),
    }


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _format_label(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d") if isinstance(value, date) and not isinstance(value, datetime) else value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _looks_like_dates(labels: list[str]) -> bool:
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    hits = sum(1 for label in labels if date_pattern.match(label) or label == "DATE")
    return hits >= max(2, len(labels) // 2)


def _parse_visual_intent(user_query: str) -> dict[str, Any]:
    """Extract chart type and row limit from the user's natural language request."""
    q = user_query.lower()
    intent: dict[str, Any] = {"chart_type": None, "limit": None}

    if re.search(r"\bpie[\s-]?chart\b|\bpie graph\b|\bas a pie\b", q):
        intent["chart_type"] = "pie"
    elif re.search(r"\bdoughnut[\s-]?chart\b|\bdonut[\s-]?chart\b|\bas a doughnut\b", q):
        intent["chart_type"] = "doughnut"
    elif re.search(r"\bline[\s-]?chart\b|\bas a line\b", q):
        intent["chart_type"] = "line"
    elif re.search(r"\bbar[\s-]?chart\b|\bas a bar\b|\bhistogram\b", q):
        intent["chart_type"] = "bar"
    elif re.search(r"\bscatter[\s-]?chart\b|\bscatter plot\b", q):
        intent["chart_type"] = "scatter"

    for pattern in (
        r"\b(?:top|most)\s+(\d+)\b",
        r"\b(\d+)\s+(?:most|top|biggest|largest)\b",
        r"\bshow\s+(\d+)\b",
    ):
        match = re.search(pattern, q)
        if match:
            intent["limit"] = int(match.group(1))
            break

    return intent


def _truncate_label(label: str, max_len: int = 48) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _infer_chart_type(labels: list[str], user_query: str, explicit_type: str | None = None) -> str:
    if explicit_type:
        return explicit_type

    q = user_query.lower()
    if any(
        w in q
        for w in (
            "over time",
            "per day",
            "daily",
            "weekly",
            "monthly",
            "trend",
            "timeline",
            "by date",
            "frequently",
            "frequency",
            "how often",
            "over hour",
            "per hour",
        )
    ):
        return "line"
    if _looks_like_dates(labels):
        return "line"
    if any(w in q for w in ("distribution", "proportion", "percentage", "share", "breakdown", "pie", "part of")):
        return "pie" if len(labels) <= 12 else "bar"
    return "bar"


def _extract_label_value_pairs(
    rows: list[list[Any]],
    row_limit: int = 20,
) -> tuple[list[str], list[float]] | None:
    if not rows or len(rows) < 2:
        return None

    labels: list[str] = []
    values: list[float] = []

    for row in rows[:row_limit]:
        if not row:
            continue

        value_idx = None
        for idx in range(len(row) - 1, -1, -1):
            if _is_numeric(row[idx]):
                value_idx = idx
                break
        if value_idx is None:
            continue

        label_idx = 0 if value_idx != 0 else (1 if len(row) > 1 else None)
        if label_idx is None:
            continue

        labels.append(_truncate_label(_format_label(row[label_idx])))
        values.append(float(row[value_idx]))

    if len(labels) < 2:
        return None
    return labels, values


def build_chart_from_rows(
    rows: list[list[Any]] | None,
    user_query: str,
    intent: dict[str, Any] | None = None,
) -> dict | None:
    if not rows:
        return None

    intent = intent or _parse_visual_intent(user_query)
    row_limit = intent.get("limit") or 20

    pairs = _extract_label_value_pairs(rows, row_limit=row_limit)
    if not pairs:
        return None

    labels, values = pairs
    chart_type = _infer_chart_type(labels, user_query, explicit_type=intent.get("chart_type"))
    q_lower = user_query.lower()
    if "response time" in q_lower or "latency" in q_lower:
        dataset_label = "Avg response time (ms)"
    elif "count" in q_lower or "active" in q_lower or "famous" in q_lower or "popular" in q_lower:
        dataset_label = "Count"
    else:
        dataset_label = "Value"

    dataset: dict[str, Any] = {
        "label": dataset_label,
        "data": values,
    }

    if chart_type in ("pie", "doughnut"):
        dataset["backgroundColor"] = CHART_PALETTE[: len(labels)]
        dataset["borderWidth"] = 1
    elif chart_type == "line":
        dataset["borderColor"] = "rgba(79, 140, 255, 1)"
        dataset["backgroundColor"] = "rgba(79, 140, 255, 0.15)"
        dataset["fill"] = True
        dataset["tension"] = 0.3
    else:
        dataset["backgroundColor"] = CHART_PALETTE[: len(labels)]
        dataset["borderRadius"] = 6

    options: dict[str, Any] = {
        "responsive": True,
        "plugins": {"legend": {"display": chart_type in ("pie", "doughnut", "line")}},
    }
    if chart_type in ("pie", "doughnut"):
        options["plugins"]["legend"]["position"] = "right"

    return {
        "type": chart_type,
        "data": {"labels": labels, "datasets": [dataset]},
        "options": options,
    }


async def _generate_summary(
    user_query: str,
    agent_response: str,
    sql_data: str,
    llm: LLM,
) -> tuple[str, float, dict[str, int]]:
    try:
        result: SummaryResult = await llm.astructured_predict(
            SummaryResult,
            SUMMARY_PROMPT,
            user_query=user_query,
            agent_response=agent_response,
            sql_data=sql_data[:1500],
        )
        return result.summary.strip(), 0.0, _empty_usage()
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        response = await llm.acomplete(
            SUMMARY_PROMPT.format(
                user_query=user_query,
                agent_response=agent_response,
                sql_data=sql_data[:1500],
            )
        )
        cost, usage = _usage_from_llm_response(response)
        return str(response).strip(), cost, usage


async def build_visualization(
    user_query: str,
    agent_response: str,
    steps: list[dict],
    llm: LLM,
    sql_results: list[str] | None = None,
) -> tuple[dict, float, dict[str, int]]:
    intent = _parse_visual_intent(user_query)
    viz_cost = 0.0
    viz_tokens = _empty_usage()

    if sql_results is None:
        sql_results = extract_sql_results(steps)

    rows = parse_sql_rows(sql_results[-1]) if sql_results else None
    table = rows_to_table(rows)
    chart = build_chart_from_rows(rows, user_query, intent=intent)

    if chart is None and rows and len(rows) >= 2 and intent.get("chart_type"):
        chart = build_chart_from_rows(rows, user_query + " " + intent["chart_type"], intent=intent)

    if not sql_results:
        return {
            "summary": agent_response.strip(),
            "chart": chart,
            "table": table,
        }, viz_cost, viz_tokens

    if _is_good_agent_summary(agent_response):
        summary = agent_response.strip()
    else:
        quick = _summary_from_rows(rows, user_query)
        if quick:
            summary = quick
        else:
            sql_data = sql_results[-1]
            if len(sql_data) > 1500:
                sql_data = sql_data[:1500] + "\n... (truncated)"
            summary, viz_cost, viz_tokens = await _generate_summary(
                user_query, agent_response, sql_data, llm
            )

    return {
        "summary": summary or agent_response.strip(),
        "chart": chart,
        "table": table,
    }, viz_cost, viz_tokens
