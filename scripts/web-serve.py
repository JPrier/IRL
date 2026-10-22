#!/usr/bin/env python3
"""Serve the built browser client locally.

    python3 scripts/web-serve.py [--dir dist/web] [--port 8080] [--base /IRL/]

Serves `.wasm` as `application/wasm`, disables caching (so rebuilds show up
immediately) and can mount the site under a subpath to reproduce a GitHub
Pages project URL. Binds to 127.0.0.1 only.
"""
import argparse
import functools
import http.server
import mimetypes
import os
import sys

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")


class Handler(http.server.SimpleHTTPRequestHandler):
    base = "/"

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def translate_path(self, path):
        if self.base != "/" and path.startswith(self.base):
            path = "/" + path[len(self.base):]
        elif self.base != "/" and path.rstrip("/") + "/" == self.base:
            path = "/"
        return super().translate_path(path)

    def do_GET(self):
        if self.base != "/" and self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", self.base)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        sys.stderr.write("[web] " + (fmt % args) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="dist/web")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--base", default="/", help="mount path, e.g. /IRL/")
    a = p.parse_args()
    base = "/" + a.base.strip("/") + "/" if a.base.strip("/") else "/"
    if not os.path.isfile(os.path.join(a.dir, "index.html")):
        sys.exit(f"{a.dir}/index.html not found — run ./scripts/web-build.sh first")
    Handler.base = base
    handler = functools.partial(Handler, directory=a.dir)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    print(f"IRL browser build: http://127.0.0.1:{a.port}{base}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
