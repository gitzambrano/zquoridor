# dev_server.py -- local HTTP server for developing gui_web/ without
# fighting the browser cache. Adds "Cache-Control: no-store" to every
# response so each reload always sees the current files.
# Usage: python dev_server.py [port]   (from gui_web/, default port 8123)
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123

class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

HTTPServer(("127.0.0.1", PORT), NoCacheHandler).serve_forever()
