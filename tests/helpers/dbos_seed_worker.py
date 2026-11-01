"""Run the seed CLI with a local blocking typed runner for interruption tests."""

import json
import sqlite3
import sys
import threading
from pathlib import Path

from typer.testing import CliRunner

from petri.cli import app
from petri.cli import seed as seed_module
from petri.reasoning.decomposition_schemas import InitialDecomposition, WhyExpansion


def main():
    root = Path(sys.argv[1])
    mode = sys.argv[2]
    root.mkdir(parents=True, exist_ok=True)
    from petri.cli.init import create_petri_dish
    create_petri_dish(root, dish_name="dish")
    (root / "defaults" / "petri.yaml").write_text(
        "name: dish\nmodel: test\nexecution:\n  backend: dbos\ntelemetry:\n  enabled: false\n"
    )
    audit = root / "audit.sqlite"
    with sqlite3.connect(audit) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS calls(name TEXT)")

    class Runner:
        def __init__(self):
            self.closed = False
            self.release = threading.Event()

        def record(self, name):
            with sqlite3.connect(audit) as conn:
                conn.execute("INSERT INTO calls VALUES (?)", (name,))

        def initial_decomposition(self, request, on_progress=None):
            self.record("initial")
            return InitialDecomposition.model_validate(
                {
                    "nodes": [
                        {"claim_text": "The meter is a defined unit"},
                        {"claim_text": "The second is a defined unit", "relation": "limits"},
                    ]
                }
            )

        def expand_why(self, request, on_progress=None):
            if not (root / "release").exists():
                (root / "in-expansion").write_text("ready")
                self.release.wait()
                if self.closed:
                    raise RuntimeError("Interrupted before a model response")
            self.record(request.claim)
            return WhyExpansion.model_validate(
                {"is_atomic": True, "bedrock_reason": "A unit definition", "sub_premises": []}
            )

        def close(self):
            self.closed = True
            self.release.set()

    provider = Runner()
    seed_module.find_petri_dir = lambda: root
    seed_module.get_dish_id = lambda _: "dish"
    seed_module.resolve_provider = lambda _: provider
    arguments = ["seed", "--colony", "units", "--no-questions"]
    if mode == "resume":
        arguments.append("--resume")
    else:
        arguments.append("Physical units")
    result = CliRunner().invoke(app, arguments)
    (root / "cli-result.json").write_text(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "output": result.output,
                "error": repr(result.exception),
            }
        )
    )
    provider.close()
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
