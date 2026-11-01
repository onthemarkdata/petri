"""Pin extracted prompts to text captured from the original provider."""

import json
from pathlib import Path

import pytest

from petri.reasoning import prompts

SNAPSHOTS = [case for case in json.loads(
    (Path(__file__).parents[1] / "fixtures" / "prompts" / "legacy.json").read_text()
) if case["method"] not in {"decompose_claim", "decompose_why"}]
# Decomposition intentionally uses quota-free prompts, covered by test_decomposer_caps.


@pytest.mark.parametrize("case", SNAPSHOTS, ids=lambda case: case["method"])
def test_builder_and_separate_suffix_match_original(case):
    method = case["method"]
    body = getattr(prompts, f"build_{method}_prompt")(*case["args"], **case.get("kwargs", {}))
    suffix = ""
    if method != "summarize_evidence":
        args = []
        if method in {"decompose_claim", "decompose_why"}:
            args = [case["args"][3]]
        elif method == "assess_cell":
            args = [case["kwargs"]["valid_verdicts"]]
        suffix = getattr(prompts, f"{method}_output_suffix")(*args)
        assert suffix
    assert body + suffix == case["prompt"]
    assert case["prompt"][:len(body)] == body
    assert "EMIT JSON" not in body
    assert "Return ONLY a JSON" not in body
