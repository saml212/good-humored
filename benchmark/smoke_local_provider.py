#!/usr/bin/env python3
"""Plumbing smoke for the `api:local` provider entry (Phase-B screen).

Stands up a stdlib mock of an OpenAI-compatible /chat/completions
endpoint on an ephemeral localhost port, points local.env at it, and
drives `make_openai_compat("local")` end-to-end — verifying the
registry entry, the model_var read-from-env path, auth header shape,
temperature field, and response parsing, all WITHOUT a GPU or vLLM.
On the node, local.env instead carries the real vLLM server's URL and
served-model id; this smoke is the proof the harness side works before
any GPU hour is spent.

Secrets hygiene: local.env is written under the fleet-secrets dir the
provider reads from. An EXISTING local.env is backed up and restored
byte-for-byte (try/finally) — this script never clobbers real node
config.

Run: python3 -m benchmark.smoke_local_provider
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.providers import _FLEET_SECRETS_DIR, make_openai_compat  # noqa: E402

CANNED_REPLY = "mock reply: the local provider plumbing works."
EXPECTED_MODEL = "mock-screen-candidate"
EXPECTED_KEY = "local-dummy-key"

seen_requests = []


class MockOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen_requests.append({
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": body,
        })
        resp = {"choices": [{"message": {"role": "assistant",
                                         "content": CANNED_REPLY},
                             "finish_reason": "stop"}],
                "model": body.get("model")}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep the smoke output clean
        pass


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), MockOpenAI)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    secrets_dir = Path(_FLEET_SECRETS_DIR)
    secrets_dir.mkdir(parents=True, exist_ok=True)
    env_path = secrets_dir / "local.env"
    backup = env_path.read_bytes() if env_path.exists() else None
    try:
        env_path.write_text(
            "LOCAL_API_KEY=%s\nLOCAL_BASE_URL=http://127.0.0.1:%d/v1\n"
            "LOCAL_MODEL=%s\n" % (EXPECTED_KEY, port, EXPECTED_MODEL))

        complete = make_openai_compat("local", temperature=0.9)
        reply = complete("say something")

        checks = [
            ("reply round-trips", reply == CANNED_REPLY),
            ("path is /v1/chat/completions",
             seen_requests[0]["path"] == "/v1/chat/completions"),
            ("bearer auth sent",
             seen_requests[0]["auth"] == "Bearer " + EXPECTED_KEY),
            ("model from LOCAL_MODEL",
             seen_requests[0]["body"]["model"] == EXPECTED_MODEL),
            ("temperature honored",
             seen_requests[0]["body"].get("temperature") == 0.9),
        ]
        failed = [name for name, ok in checks if not ok]
        for name, ok in checks:
            print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        if failed:
            print("SMOKE FAILED: %s" % ", ".join(failed))
            return 1
        print("SMOKE PASSED: api:local end-to-end against mock server "
              "(port %d)" % port)
        return 0
    finally:
        if backup is not None:
            env_path.write_bytes(backup)
        else:
            env_path.unlink(missing_ok=True)
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
