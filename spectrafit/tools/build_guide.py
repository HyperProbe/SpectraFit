"""Build README.html, the user guide that opens with a double-click, from README.md.

    python tools/build_guide.py          # writes README.html next to README.md

README.md stays the only file to edit. The HTML is self-contained (images are embedded), so it
works when copied anywhere. tests/test_guide.py fails when README.html is out of date.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "README.md"
TARGET = ROOT / "README.html"

CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --code: #f6f8fa; --link: #0969da; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --code: #151b23; --link: #4493f8; }
}
html { background: var(--bg); }
body { color: var(--fg); background: var(--bg); max-width: 920px; margin: 0 auto; padding: 32px 20px 64px;
       font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; }
h1, h2 { border-bottom: 1px solid var(--line); padding-bottom: .3em; }
h1 { font-size: 2em; } h2 { font-size: 1.5em; margin-top: 1.8em; } h3 { font-size: 1.2em; margin-top: 1.5em; }
a { color: var(--link); }
img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }
code { background: var(--code); padding: .15em .35em; border-radius: 4px; font-size: 90%;
       font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: var(--code); padding: 14px 16px; border-radius: 6px; overflow-x: auto; line-height: 1.45; }
pre code { background: none; padding: 0; font-size: 85%; }
table { border-collapse: collapse; display: block; overflow-x: auto; margin: 1em 0; }
th, td { border: 1px solid var(--line); padding: 6px 13px; vertical-align: top; }
th { background: var(--code); text-align: left; }
blockquote { margin: 1em 0; padding: 0 1em; color: var(--muted); border-left: .25em solid var(--line); }
hr { border: none; border-top: 1px solid var(--line); margin: 2em 0; }
.version { color: var(--muted); font-size: 90%; }
@media print { body { max-width: none; } a { color: inherit; } }
"""


def _embed_images(html: str) -> str:
    """Replace src="relative.png" by the image itself, so the file works on its own."""
    def repl(m):
        src = m.group(1)
        path = ROOT / src
        if src.startswith(("http:", "https:", "data:")) or not path.is_file():
            return m.group(0)
        kind = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "svg": "svg+xml", "gif": "gif"}[path.suffix[1:].lower()]
        return f'src="data:image/{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def build() -> str:
    import markdown

    sys.path.insert(0, str(ROOT / "src"))
    from spectrafit import __version__

    text = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"],
                             output_format="html")
    body = _embed_images(body)
    # the version under the title, so people can tell which guide they have
    body = body.replace("</h1>", f'</h1>\n<p class="version">User guide for SpectraFit {__version__}</p>', 1)
    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>SpectraFit {__version__} – user guide</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
            f"<!-- Generated from README.md by tools/build_guide.py. Edit README.md, not this file. -->\n"
            f"{body}\n</body>\n</html>\n")


if __name__ == "__main__":
    TARGET.write_text(build(), encoding="utf-8")
    print(f"Wrote {TARGET} ({TARGET.stat().st_size // 1024} KB)")
