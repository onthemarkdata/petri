"""The self-contained trace view renders its API fixture headlessly."""

import json
import re
import shutil
import subprocess
import tomllib
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path

import pytest

from petri import __version__
from petri.dashboard.frontend import build_frontend_html

ROOT = Path(__file__).parents[2]


class Markup(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def render_fixture():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for the headless frontend contract.')
    html = build_frontend_html()
    code = re.search(r'<script id="trace-view-code">(.*?)</script>', html, re.S)[1]
    spans = json.loads((ROOT / 'tests/fixtures/trace.json').read_text())
    script = code + '\nconst spans = ' + json.dumps(spans) + ';\nconsole.log(JSON.stringify({tree:buildTraceTree(spans),markup:renderTraceWaterfall(spans)}));'
    result = subprocess.run([node, '-e', script], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_tree_builder_nests_step_agent_and_model():
    result = render_fixture()
    node = result['tree'][0]
    for depth, name in enumerate(['petri grow', 'dbos.step', 'agent.run', 'model.request']):
        assert node['depth'] == depth
        assert node['span']['name'] == name
        if node['children']:
            node = node['children'][0]
    assert node['duration_ns'] == 30_000_000


def test_waterfall_has_proportional_bars_and_indentation():
    markup = Markup(render_fixture()['markup'])
    bars = [attrs for tag, attrs in markup.tags if 'trace-bar' in attrs.get('class', '').split()]
    rows = [attrs for tag, attrs in markup.tags if attrs.get('class') == 'trace-row']
    assert [row['data-depth'] for row in rows] == ['0', '1', '2', '3']
    assert [re.search(r'width:([\d.]+)%', bar['style'])[1] for bar in bars] == ['100', '80', '60', '30']
    assert [bar['class'] for bar in bars] == ['trace-bar trace-workflow', 'trace-bar trace-step', 'trace-bar trace-agent', 'trace-bar trace-model']


def test_frontend_has_trace_panel_drilldown_empty_state_and_current_version():
    html = build_frontend_html()
    assert 'id="traces-panel"' in html
    assert 'View trace' in html
    assert '/api/traces' in html
    assert 'Enable telemetry.enabled and telemetry.local.enabled' in html
    assert 'traces-refresh' in html
    assert 'v' + __version__ in html
    assert __version__ == version('petri-grow')
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert config['tool']['setuptools_scm']['fallback_version'] == '0.0.0'
    assert '"0.0.0"' in (ROOT / 'petri/__init__.py').read_text()


def test_frontend_loads_no_remote_scripts_fonts_or_images():
    html = build_frontend_html()
    for tag, attrs in Markup(html).tags:
        for name in ['src', 'href', 'poster']:
            value = attrs.get(name, '')
            assert not re.match(r'^(https?:)?//', value), (tag, name, value)
    for url in re.findall(r'url\([\'\"]?([^\)\'\"]+)', html):
        assert not re.match(r'^(https?:)?//', url)
    assert all('data-vendor="' + filename + '"' in html for filename in ['xterm.css', 'xterm.js', 'fit.js', 'marked.js', 'purify.js'])


def test_colony_labels_use_the_complete_dish_prefix():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for the frontend label contract.')
    html = build_frontend_html()
    code = re.search(r'function colonyDisplayName\(.*?\n\}',html,re.S)[0]
    cases = [
        ['grow-live-isolated',{'dish_id':'grow-live'}],
        ['grow-live-long-colony-name',{'dish_id':'grow-live'}],
        ['dish-alpha',{'dish_id':'dish'}],
        ['other-colony',{'dish_id':'grow-live'}],
        ['unknown-prefix-colony',None],
    ]
    script = code+'\nconsole.log(JSON.stringify('+json.dumps(cases)+'.map(args=>colonyDisplayName(...args))));'
    result = subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout) == ['isolated','long-colony-name','alpha','other-colony','unknown-prefix-colony']
