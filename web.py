#!/usr/bin/env python3
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from test_query import LLM_MODEL, run_query_stream

HTML_PATH = Path(__file__).parent / "index.html"
PORT = 8000


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/query/stream":
            self._handle_query_stream()
            return
        if self.path == "/api/query":
            self._handle_query_json()
            return
        self.send_error(404)

    def _read_query(self) -> str | None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        query = str(body.get("query", "")).strip()
        return query or None

    def _handle_query_json(self) -> None:
        try:
            query = self._read_query()
            if not query:
                self._send_json(400, {"error": "Query is required"})
                return

            from test_query import run_query

            result = asyncio.run(run_query(query))
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_query_stream(self) -> None:
        try:
            query = self._read_query()
            if not query:
                self._send_json(400, {"error": "Query is required"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            async def stream() -> None:
                async for event in run_query_stream(query):
                    if event["kind"] == "step":
                        payload = json.dumps(event["step"])
                        self._write_sse("step", payload)
                    elif event["kind"] == "status":
                        payload = json.dumps({"text": event["text"]})
                        self._write_sse("status", payload)
                    else:
                        payload = json.dumps(
                            {
                                "response": event["response"],
                                "cost": event["cost"],
                                "tokens": event.get("tokens"),
                                "llm_calls": event.get("llm_calls"),
                                "query_path": event.get("query_path"),
                                "visualization": event.get("visualization"),
                            }
                        )
                        self._write_sse("done", payload)

            asyncio.run(stream())
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._write_sse("error", json.dumps({"error": str(exc)}))
            except Exception:
                self._send_json(500, {"error": str(exc)})

    def _write_sse(self, event: str, data: str) -> None:
        message = f"event: {event}\ndata: {data}\n\n"
        self.wfile.write(message.encode("utf-8"))
        self.wfile.flush()

    def _send(self, code: int, data: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"ClickHouse NL2SQL web UI: http://localhost:{PORT}")
    print(f"Model: {LLM_MODEL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
