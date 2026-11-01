"""Offline CLI integration across backend selection, typed readers, and cleanup."""

from typer.testing import CliRunner

from petri.cli import app
from petri.harness.legacy_bridge import HarnessInferenceProvider
from tests.harness_utils import make_scripted_output_model
from tests.integration.test_bridge_seed import prepare_dish, seed_and_dry_run


def test_harness_block_seed_and_dry_run(monkeypatch, tmp_path):
    from petri.harness import resolve
    from petri.engine import preflight

    monkeypatch.chdir(tmp_path)
    dish = prepare_dish(tmp_path, {"backend": "provider", "model": "test:offline"})
    model = make_scripted_output_model([{"nodes": [{"claim_text": "Temperature is measured in Celsius."}]}])
    monkeypatch.setattr(resolve, "resolve_model", lambda config: model)
    monkeypatch.setattr(preflight, "check_claude_cli", lambda: (_ for _ in ()).throw(AssertionError("Wrong backend")))
    seed_and_dry_run(dish)


def test_seed_exception_closes_worker(monkeypatch, tmp_path):
    from petri.harness import resolve
    from petri.harness import legacy_bridge

    monkeypatch.chdir(tmp_path)
    prepare_dish(tmp_path, {"backend": "provider", "model": "test:offline"})
    model = make_scripted_output_model([{}])
    monkeypatch.setattr(resolve, "resolve_model", lambda config: model)
    created = []

    class RecordingProvider(HarnessInferenceProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(legacy_bridge, "HarnessInferenceProvider", RecordingProvider)
    result = CliRunner().invoke(app, ["seed", "A substantive claim", "--no-questions"])
    assert result.exit_code != 0
    assert created[0]._worker.closed
    assert not created[0]._worker.thread.is_alive()
