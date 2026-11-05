"""Throwaway mock of a vLLM /v1 server for testing failure modes.

usage: python mock_llm_server.py PORT BEHAVIOR [BEHAVIOR ...]
Behaviors are consumed one per POST, then every later POST is 'ok'.
  ok | slow:<seconds> | drop | 503 | 400
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
BEHAVIORS = list(sys.argv[2:])
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": "Llama-3.3-70B-Instruct"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        with LOCK:
            behavior = BEHAVIORS.pop(0) if BEHAVIORS else "ok"
        print(f"[mock] POST -> {behavior}", flush=True)
        if behavior.startswith("slow:"):
            time.sleep(float(behavior.split(":")[1]))
        elif behavior == "drop":
            self.connection.close()
            return
        elif behavior == "503":
            return self._send(503, {"error": "overloaded"})
        elif behavior == "400":
            return self._send(400, {"error": "context length exceeded"})
        user = req["messages"][-1]["content"]
        self._send(200, {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": f"MOCK SUMMARY ({len(user)} prompt chars)."}}]})


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
