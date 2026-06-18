"""ClickHouse native protocol client (port 9000) with clickhouse-connect-compatible API."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from clickhouse_driver import Client

CH_HOST = "localhost"
CH_PORT = 9000
CH_USER = "admin"
CH_PASSWORD = "hiffiofsuperlabs"

_TUPLE_FIELD_RE = re.compile(r"(\w+)\s+(?:Nullable\()?")


def _coerce_insert_value(value: Any, type_name: str | None) -> Any:
    """Normalize values for clickhouse-driver inserts."""
    if not type_name:
        return value
    if type_name.startswith("Tuple(") and isinstance(value, dict):
        inner = type_name[type_name.index("(") + 1 : type_name.rindex(")")]
        names = _TUPLE_FIELD_RE.findall(inner)
        return tuple(value.get(name) for name in names)
    return value


def _coerce_insert_row(
    row: list[Any],
    column_type_names: list[str] | None,
) -> list[Any]:
    if not column_type_names:
        return row
    return [
        _coerce_insert_value(value, type_name)
        for value, type_name in zip(row, column_type_names)
    ]


@dataclass
class QueryResult:
    column_names: list[str]
    result_rows: list[tuple[Any, ...]]


class NativeClickHouseClient:
    """Wraps clickhouse-driver with the subset of clickhouse-connect API we use."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def command(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self._client.execute(sql, params=parameters or {}, settings=settings or {})

    def query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> QueryResult:
        rows, column_types = self._client.execute(
            sql,
            params=parameters or {},
            settings=settings or {},
            with_column_types=True,
        )
        return QueryResult(
            column_names=[name for name, _type in column_types],
            result_rows=rows,
        )

    def insert(
        self,
        table: str,
        data: list[list[Any]],
        column_names: list[str] | None = None,
        column_type_names: list[str] | None = None,
    ) -> None:
        if column_type_names:
            data = [_coerce_insert_row(row, column_type_names) for row in data]
        if column_names:
            cols = ", ".join(column_names)
            sql = f"INSERT INTO {table} ({cols}) VALUES"
        else:
            sql = f"INSERT INTO {table} VALUES"
        self._client.execute(sql, data)


def _driver_client(database: str | None = None) -> Client:
    kwargs: dict[str, Any] = {
        "host": CH_HOST,
        "port": CH_PORT,
        "user": CH_USER,
        "password": CH_PASSWORD,
    }
    if database:
        kwargs["database"] = database
    return Client(**kwargs)


def get_ch_client(database: str | None = None) -> NativeClickHouseClient:
    return NativeClickHouseClient(_driver_client(database))
