"""Manual live-provider smoke test — NOT part of the unit suite.

Hits real CLIs/APIs (some paid), so it lives outside benchmark/tests/
and unittest discovery never picks it up. Run directly:

  python3 -m benchmark.smoke_providers \
      --providers claude:haiku,codex:mini,api:deepseek

Prints one line per spec: latency and a short reply preview. Never
prints key material — get_provider() only ever returns a callable, keys
stay inside the provider closures.
"""

import argparse
import time

from .providers import get_provider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", required=True,
                    help="comma-separated provider specs, e.g. "
                         "claude:haiku,codex:mini,api:deepseek")
    ap.add_argument("--prompt", default="Say OK and nothing else")
    args = ap.parse_args()

    specs = [s.strip() for s in args.providers.split(",") if s.strip()]
    for spec in specs:
        t0 = time.time()
        try:
            complete = get_provider(spec)
            reply = complete(args.prompt)
            dt = time.time() - t0
            print("%-16s OK   %6.1fs  %r" % (spec, dt, reply[:80]))
        except Exception as e:  # smoke test: report and move to the next spec
            dt = time.time() - t0
            print("%-16s FAIL %6.1fs  %r" % (spec, dt, e))


if __name__ == "__main__":
    main()
