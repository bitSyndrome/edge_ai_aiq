import os
import glob
import json
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import netron

NETRON_PORT = 8081
netron_lock = threading.Lock()


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Model Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column; }
  .toolbar { display: flex; align-items: center; gap: 12px; padding: 10px 20px; background: #16213e; border-bottom: 1px solid #0f3460; }
  .toolbar h3 { font-size: 14px; color: #aaa; }
  select { padding: 6px 12px; border-radius: 4px; border: 1px solid #0f3460; background: #1a1a2e; color: #eee; font-size: 14px; cursor: pointer; }
  select:hover { border-color: #e94560; }
  .info { font-size: 12px; color: #888; margin-left: auto; }
  iframe { flex: 1; border: none; background: #fff; }
  .empty { flex: 1; display: flex; align-items: center; justify-content: center; color: #555; font-size: 18px; }
</style>
</head>
<body>
<div class="toolbar">
  <h3>Model Viewer</h3>
  <select id="modelSelect" onchange="loadModel()">
    <option value="">-- Select Model --</option>
    {{OPTIONS}}
  </select>
  <span class="info" id="info"></span>
</div>
<iframe id="viewer" class="empty"></iframe>
<script>
function loadModel() {
  const sel = document.getElementById('modelSelect');
  const file = sel.value;
  const info = document.getElementById('info');
  const viewer = document.getElementById('viewer');
  if (!file) { viewer.src = ''; info.textContent = ''; return; }
  info.textContent = 'Loading...';
  fetch('/api/load?file=' + encodeURIComponent(file))
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        viewer.src = 'http://' + location.hostname + ':' + d.port;
        info.textContent = d.file + ' (' + d.size + ')';
      } else {
        info.textContent = 'Error: ' + d.error;
      }
    });
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    model_dir = "models"

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_index()
        elif parsed.path == "/api/load":
            params = parse_qs(parsed.query)
            self.load_model(params.get("file", [""])[0])
        else:
            self.send_error(404)

    def serve_index(self):
        files = sorted(glob.glob(os.path.join(self.model_dir, "*.onnx")))
        options = ""
        for f in files:
            name = os.path.basename(f)
            size_kb = os.path.getsize(f) / 1024
            options += f'<option value="{name}">{name} ({size_kb:.1f} KB)</option>\n'

        html = HTML_TEMPLATE.replace("{{OPTIONS}}", options)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def load_model(self, filename):
        if not filename:
            self.json_response({"ok": False, "error": "No file specified"})
            return

        filepath = os.path.join(self.model_dir, filename)
        if not os.path.isfile(filepath):
            self.json_response({"ok": False, "error": "File not found"})
            return

        with netron_lock:
            try:
                netron.stop()
            except Exception:
                pass
            netron.start(filepath, address=("0.0.0.0", NETRON_PORT), browse=False)

        size_kb = os.path.getsize(filepath) / 1024
        self.json_response({
            "ok": True,
            "file": filename,
            "size": f"{size_kb:.1f} KB",
            "port": NETRON_PORT,
        })

    def json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Model Viewer Web UI")
    parser.add_argument("--model-dir", default="models", help="Model directory (default: models)")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port (default: 8080)")
    args = parser.parse_args()

    Handler.model_dir = args.model_dir

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Model Viewer running at http://localhost:{args.port}")
    print(f"Netron backend on port {NETRON_PORT}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        try:
            netron.stop()
        except Exception:
            pass
        server.server_close()


if __name__ == "__main__":
    main()
