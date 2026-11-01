"""Inline vendor scripts must not receive dashboard helper injection."""

from html.parser import HTMLParser

from petri.dashboard.frontend import build_frontend_html


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.vendor = None
        self.scripts = {}

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.vendor = dict(attrs).get("data-vendor")

    def handle_data(self, data):
        if self.vendor:
            self.scripts[self.vendor] = self.scripts.get(self.vendor, "") + data

    def handle_endtag(self, tag):
        if tag == "script":
            self.vendor = None


def test_trace_helpers_do_not_break_the_vendor_head_literal():
    html = build_frontend_html()
    parsed = Scripts()
    parsed.feed(html)
    vendor = parsed.scripts["purify.js"]
    assert "</head><body>" in vendor
    assert "return ne}" in vendor
    assert "buildTraceTree" not in vendor
    assert html.count("function buildTraceTree(") == 1
    assert html.count("function renderTraceWaterfall(") == 1
