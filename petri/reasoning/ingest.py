"""Content ingestion for Petri.

Accepts URLs, local files (text, PDF, markdown, etc.), and raw text,
extracting readable content for use as seed claims or evidence.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from petri.models import IngestionResult

# File extensions we handle directly as text
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp",
    ".log", ".cfg", ".ini", ".toml",
}


class IngestError(RuntimeError):
    """A source could not be fetched, read, or decoded into evidence."""

    def __init__(self, source_kind: str, source: str, cause: str | Exception):
        self.source_kind = source_kind
        self.source = source
        self.cause = str(cause)
        super().__init__(source_kind, source, self.cause)

    def __str__(self):
        return f"Could not ingest {self.source_kind} {self.source!r}: {self.cause}"


def ingest(source: str) -> IngestionResult:
    """Extract typed content, raising IngestError for a failed source."""
    # Check if it's a URL
    if _is_url(source) or source.startswith(("http://", "https://")):
        return _ingest_url(source)

    if is_file_source(source):
        path = Path(source).expanduser()
        if path.suffix.lower() == ".pdf":
            return _ingest_pdf(path)
        return _ingest_file(path)

    # Treat as raw text
    return IngestionResult(
        source_type="text",
        source=source,
        title=_extract_title_from_text(source),
        content=source,
        metadata={},
    )


def is_file_source(source: str) -> bool:
    """Recognize explicit paths without treating multiline prose as a filename."""
    if "\n" in source or len(source) > 4096:
        return False
    path = Path(source).expanduser()
    if source.startswith(("/", "./", "../", "~/")) or path.suffix.lower() in _TEXT_EXTENSIONS | {".pdf"}:
        return True
    try:
        return path.is_file()
    except OSError:
        return False


def _is_url(source: str) -> bool:
    """Check if source looks like a URL."""
    try:
        parsed = urlparse(source)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _ingest_url(url: str) -> IngestionResult:
    """Fetch and extract text content from a URL."""
    import httpx

    try:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "Petri/0.1 (research-tool)"},
        )
        response.raise_for_status()
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise IngestError("url", url, exc) from exc

    content_type = response.headers.get("content-type", "")

    # PDF served over HTTP
    if "application/pdf" in content_type:
        return _ingest_pdf_bytes(response.content, source=url)

    # HTML
    if "text/html" in content_type:
        html = response.text
        title = _extract_html_title(html)
        text = _html_to_text(html)
        return IngestionResult(
            source_type="url",
            source=url,
            title=title,
            content=text,
            metadata={"url": url, "content_type": content_type},
        )

    # Plain text or other text formats
    return IngestionResult(
        source_type="url",
        source=url,
        title="",
        content=response.text,
        metadata={"url": url, "content_type": content_type},
    )


def _ingest_file(path: Path) -> IngestionResult:
    """Read a local text file."""
    try:
        content = path.read_text(errors="replace")
        size = path.stat().st_size
    except Exception as exc:
        raise IngestError("file", str(path), exc) from exc

    # If HTML file, extract text
    if path.suffix.lower() in (".html", ".htm"):
        title = _extract_html_title(content)
        content = _html_to_text(content)
    else:
        title = path.stem.replace("-", " ").replace("_", " ").title()

    return IngestionResult(
        source_type="file",
        source=str(path),
        title=title,
        content=content,
        metadata={"path": str(path), "size": size},
    )


def _ingest_pdf(path: Path) -> IngestionResult:
    """Extract text from a local PDF file."""
    try:
        raw = path.read_bytes()
        return _ingest_pdf_bytes(raw, source=str(path))
    except Exception as exc:
        if isinstance(exc, IngestError):
            raise
        raise IngestError("pdf", str(path), exc) from exc


def _ingest_pdf_bytes(data: bytes, source: str = "") -> IngestionResult:
    """Extract PDF text, raising a typed error when decoding is unavailable."""
    title = ""
    pages: list[str] = []
    page_count = 0

    try:
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)

        # Extract title from metadata
        meta = reader.metadata
        if meta and meta.title:
            title = meta.title

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

    except ImportError as exc:
        raise IngestError("pdf", source, "Install petri[pdf] to read PDF evidence") from exc

    except Exception as exc:
        raise IngestError("pdf", source, exc) from exc

    content = "\n\n".join(pages)
    if not title and source:
        title = Path(source).stem.replace("-", " ").replace("_", " ").title()

    return IngestionResult(
        source_type="pdf",
        source=source,
        title=title,
        content=content,
        metadata={
            "source": source,
            "page_count": page_count,
            "extracted_pages": len(pages),
        },
    )


# ── HTML helpers ─────────────────────────────────────────────────────────


def _extract_html_title(html: str) -> str:
    """Extract <title> from HTML."""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        # Clean HTML entities
        title = re.sub(r"&amp;", "&", title)
        title = re.sub(r"&lt;", "<", title)
        title = re.sub(r"&gt;", ">", title)
        title = re.sub(r"&#\d+;", "", title)
        title = re.sub(r"&\w+;", "", title)
        return title
    return ""


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text.

    Strips tags, scripts, styles, and normalizes whitespace.
    Preserves paragraph breaks.
    """
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<noscript[^>]*>.*?</noscript>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Add newlines for block elements
    block_tags = r"</?(p|div|br|hr|h[1-6]|li|tr|blockquote|article|section|header|footer|main|nav|aside)[^>]*>"
    text = re.sub(block_tags, "\n", text, flags=re.IGNORECASE)

    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common entities
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"&\w+;", " ", text)

    # Normalize whitespace: collapse runs of spaces/tabs on each line
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            line = re.sub(r"[ \t]+", " ", line)
            cleaned.append(line)

    # Collapse multiple blank lines
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def _extract_title_from_text(text: str) -> str:
    """Extract a title from raw text (first meaningful line)."""
    for line in text.strip().splitlines():
        line = line.strip()
        if line and len(line) > 5:
            return line[:200]
    return text[:200].strip()
