"""Late span batches refresh visible telemetry without an idle polling loop."""

import json
import re
import shutil
import subprocess

import pytest

from petri.dashboard.frontend import build_frontend_html


CLOCK = r'''
let now = 0, nextId = 0;
const tasks = new Map();
function schedule(callback, delay) {
  const id = ++nextId;
  tasks.set(id, {at: now + delay, callback});
  return id;
}
function cancel(id) { tasks.delete(id); }
function advance(until) {
  while (true) {
    const due = [...tasks.entries()].filter(([id, task]) => task.at <= until)
      .sort((a, b) => a[1].at - b[1].at || a[0] - b[0]);
    if (!due.length) break;
    const [id, task] = due[0];
    now = task.at;
    tasks.delete(id);
    task.callback();
  }
  now = until;
}
'''


def run_fixture(script):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for the frontend refresh contract.')
    html = build_frontend_html()
    code = re.search(r'function createTelemetryRefresh\(.*?\n\}', html, re.S)[0]
    result = subprocess.run([node, '-e', code + CLOCK + script], check=True,
                            text=True, capture_output=True)
    return json.loads(result.stdout)


def test_span_export_after_domain_event_updates_usage_then_stops():
    result = run_fixture(r'''
let exportedUsage = 100;
const reads = [];
const refresh = createTelemetryRefresh(() => reads.push({at: now, usage: exportedUsage}),
  () => true, schedule, cancel);
refresh.request();
advance(4000);
exportedUsage = 200;
advance(15000);
const readsAfterExport = reads.length;
advance(60000);
console.log(JSON.stringify({reads, readsAfterExport, pending: tasks.size}));
''')
    assert result == {
        'reads': [
            {'at': 0, 'usage': 100}, {'at': 1500, 'usage': 100},
            {'at': 6000, 'usage': 200}, {'at': 15000, 'usage': 200},
        ],
        'readsAfterExport': 4,
        'pending': 0,
    }


def test_event_bursts_coalesce_and_navigation_cancels_followups():
    result = run_fixture(r'''
let active = true, reads = 0;
const refresh = createTelemetryRefresh(() => reads++, () => active, schedule, cancel);
refresh.request();
advance(500);
refresh.request();
const pendingAfterBurst = tasks.size;
active = false;
refresh.cancel();
advance(60000);
refresh.request();
console.log(JSON.stringify({reads, pendingAfterBurst, pending: tasks.size}));
''')
    assert result == {'reads': 2, 'pendingAfterBurst': 3, 'pending': 0}
