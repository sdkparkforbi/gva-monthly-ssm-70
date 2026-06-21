# -*- coding: utf-8 -*-
"""index.html → Playwright(chromium) → PDF(페이지번호). 로컬 HTTP 서버 자동 기동."""
import http.server, socketserver, threading, os, functools, sys
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8733
URL = f"http://localhost:{PORT}/index.html"
OUT = os.path.join(ROOT, "gva-monthly-ssm-70-paper.pdf")
FOOTER = ('<div style="font-size:9px;width:100%;text-align:center;color:#888;'
          'padding-top:2px;">- <span class="pageNumber"></span> / '
          '<span class="totalPages"></span> -</div>')


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
    httpd = socketserver.TCPServer(("", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    httpd = serve()
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 900, "height": 1200})
            pg.goto(URL, wait_until="networkidle", timeout=60000)
            pg.wait_for_function("window.__chartsReady===true", timeout=30000)
            try:
                pg.wait_for_function(
                    "document.querySelectorAll('mjx-container').length > 8", timeout=20000)
            except Exception:
                pass
            pg.wait_for_timeout(2500)
            pg.evaluate("try{Object.values(Chart.instances).forEach(c=>c.resize());}catch(e){}")
            pg.wait_for_timeout(800)
            pg.pdf(path=OUT, format="A4", print_background=True,
                   margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
                   display_header_footer=True, header_template="<div></div>",
                   footer_template=FOOTER)
            b.close()
        print("PDF 생성:", OUT, "|", os.path.getsize(OUT), "bytes")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
