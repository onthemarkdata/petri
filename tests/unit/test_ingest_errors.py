"""Failed sources raise typed errors rather than becoming evidence text."""

import pickle

import httpx
import pytest

from petri.reasoning.ingest import IngestError, ingest


def test_url_failure_preserves_source_kind_and_cause(monkeypatch):
    def fail(*args, **options):
        raise httpx.ConnectError("Local test connection refused")

    monkeypatch.setattr(httpx, "get", fail)
    with pytest.raises(IngestError) as caught:
        ingest("https://example.invalid/evidence")
    error = pickle.loads(pickle.dumps(caught.value))
    assert error.source_kind == "url"
    assert error.cause == "Local test connection refused"
    assert "example.invalid" in str(error)


def test_missing_file_and_malformed_pdf_fail(tmp_path):
    with pytest.raises(IngestError) as caught:
        ingest(str(tmp_path / "missing.txt"))
    assert caught.value.source_kind == "file"
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a PDF")
    with pytest.raises(IngestError) as caught:
        ingest(str(pdf))
    assert caught.value.source_kind == "pdf"


def test_plain_text_is_preserved():
    text = "A measured result supports the premise."
    assert ingest(text).content == text
