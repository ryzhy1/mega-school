import json
import subprocess
import sys
import threading
from typing import Any, Optional

from helpers import safe_print


class MCPServerClient:
    """Client for local MCP server via stdio (JSON-RPC)."""
    def __init__(self, server_script: str = "server.py"):
        self.server_script = server_script
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start_server()

    def _start_server(self):
        safe_print(f"🔌 [MCP] Starting server: {self.server_script}")
        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent", "version": "1.0"},
            },
            msg_id=0,
        )
        _ = self._read_response()
        self._send_request("notifications/initialized", {}, msg_id=None)
        safe_print("✅ [MCP] Ready.")

    def _send_request(self, method: str, params: dict, msg_id: Optional[int] = 1):
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP process not started")
        req = {"jsonrpc": "2.0", "method": method, "params": params}
        if msg_id is not None:
            req["id"] = msg_id

        json_str = json.dumps(req, ensure_ascii=False)
        self.process.stdin.write(json_str + "\n")
        self.process.stdin.flush()

    def _read_response(self) -> Optional[dict]:
        if not self.process or not self.process.stdout:
            return None
        while True:
            line = self.process.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                continue

    def call_tool(self, name: str, arguments: dict) -> Any:
        with self._lock:
            self._send_request("tools/call", {"name": name, "arguments": arguments}, msg_id=1)
            resp = self._read_response()

        if resp and "result" in resp:
            content = resp["result"]["content"][0]["text"]
            try:
                return json.loads(content)
            except Exception:
                return content
        safe_print(f"❌ [MCP] Error/empty: {resp}")
        return None

    def close(self):
        try:
            if self.process:
                self.process.terminate()
        except Exception:
            pass
