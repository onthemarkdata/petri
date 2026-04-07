"""Content ingestion for Petri.

Accepts URLs, local files (text, PDF, markdown, etc.), and raw text,
extracting readable content for use as seed claims or evidence.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from petri.models import IngestionResult

logger = logging.getLogger(__name__)

# File extensions we handle directly as text
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp",
    ".log", ".cfg", ".ini", ".toml",
}


def ingest(source: str) -> IngestionResult:
    """Ingest content from a URL, file path, or raw text.

    Returns a dict with:
        - ``source_type``: "url", "file", "pdf", or "text"
        - ``source``: the original source string
        - ``title``: extracted title (if available)
        - ``content``: the extracted text content
        - ``metadata``: additional info (url, file path, page count, etc.)
    """
    # Check if it's a URL
    if _is_url(source):
        return _ingest_url(source)

    # Check if it's a file path
    path = Path(source).expanduser()
    if path.exists() and path.is_file():
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
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch URL %s: %s", url, exc)
        return IngestionResult(
            source_type="url",
            source=url,
            title="",
            content=f"[Failed to fetch: {exc}]",
            metadata={"url": url, "error": str(exc)},
        )

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
    except Exception as exc:
        logger.warning("Failed to read file %s: %s", path, exc)
        return IngestionResult(
            source_type="file",
            source=str(path),
            title=path.name,
            content=f"[Failed to read: {exc}]",
            metadata={"path": str(path), "error": str(exc)},
        )

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
        metadata={"path": str(path), "size": path.stat().st_size},
    )


def _ingest_pdf(path: Path) -> IngestionResult:
    """Extract text from a local PDF file."""
    try:
        raw = path.read_bytes()
        return _ingest_pdf_bytes(raw, source=str(path))
    except Exception as exc:
        logger.warning("Failed to read PDF %s: %s", path, exc)
        return IngestionResult(
            source_type="pdf",
            source=str(path),
            title=path.stem,
            content=f"[Failed to read PDF: {exc}]",
            metadata={"path": str(path), "error": str(exc)},
        )


def _ingest_pdf_bytes(data: bytes, source: str = "") -> IngestionResult:
    """Extract text from PDF bytes.

    Tries pypdf first, falls back to a basic binary text extraction.
    """
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

    except ImportError:
        logger.info("pypdf not installed, attempting basic PDF text extraction")
        # Basic fallback: extract readable strings from binary
        text = _extract_text_from_binary(data)
        if text:
            pages.append(text)

    except Exception as exc:
        logger.warning("PDF extraction failed: %s", exc)
        return IngestionResult(
            source_type="pdf",
            source=source,
            title=title or Path(source).stem if source else "",
            content=f"[PDF extraction failed: {exc}]",
            metadata={"source": source, "error": str(exc)},
        )

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


def _extract_text_from_binary(data: bytes) -> str:
    """Last-resort: pull printable ASCII/UTF-8 runs from binary data."""
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = data.decode("latin-1", errors="ignore")

    # Keep runs of printable characters (min 20 chars to filter noise)
    runs = re.findall(r"[\x20-\x7E]{20,}", text)
    return "\n".join(runs)


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
