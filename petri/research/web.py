"""Bounded public-web retrieval with an injectable HTTP transport."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

MAX_BYTES = 256 * 1024
MAX_TEXT = 12_000
MAX_SECONDS = 15
SEARCH_URL = "https://html.duckduckgo.com/html/"


class PublicWebError(ValueError):
    """Retrieval failed or the URL is outside the public HTTP boundary."""


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)
    url: str
    title: str
    snippet: str = ""


class RetrievedPage(BaseModel):
    model_config = ConfigDict(frozen=True)
    url: str
    title: str
    text: str
    retrieved_at: str
    content_sha256: str
    truncated: bool
    untrusted_evidence: bool = True


async def public_addresses(host: str) -> list[str]:
    rows = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({str(row[4][0]) for row in rows})


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.in_title = False
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.ignored += 1
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"}:
            self.ignored = max(0, self.ignored - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.ignored:
            return
        if self.in_title:
            self.title.append(data)
        else:
            self.parts.append(data)


class _Search(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hits: list[SearchHit] = []
        self.url: str | None = None
        self.title: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag != "a" or "result__a" not in attributes.get("class", "").split():
            return
        href = urljoin(SEARCH_URL, attributes.get("href", ""))
        decoded = parse_qs(urlsplit(href).query).get("uddg", [href])[0]
        if urlsplit(decoded).scheme not in {"http", "https"}:
            return
        self.url, self.title = decoded, []

    def handle_data(self, data):
        if self.url is not None:
            self.title.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.url is not None:
            title = re.sub(r"\s+", " ", " ".join(self.title)).strip()
            if title:
                self.hits.append(SearchHit(url=self.url, title=title[:300]))
            self.url = None


class PublicWebClient:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolve_host: Callable[[str], Awaitable[list[str]]] = public_addresses,
    ):
        self.transport = transport
        self.resolve_host = resolve_host

    async def _validate(self, url: str) -> None:
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as error:
            raise PublicWebError("The source URL is invalid.") from error
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise PublicWebError("Use a public HTTP or HTTPS URL without embedded credentials.")
        if port not in {None, 80, 443}:
            raise PublicWebError("Public retrieval supports ports 80 and 443.")
        try:
            addresses = await asyncio.wait_for(self.resolve_host(parsed.hostname), timeout=5)
        except (OSError, TimeoutError) as error:
            raise PublicWebError("The public source hostname could not be resolved.") from error
        if not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise PublicWebError("The URL must resolve only to public IP addresses.")

    async def _read(self, url: str) -> tuple[str, str, bytes, bool]:
        current = url
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=httpx.Timeout(10, connect=5),
                follow_redirects=False,
                headers={
                    "User-Agent": "PetriResearch/2",
                    "Accept": "text/html,text/plain,application/json",
                },
            ) as client:
                for redirect in range(4):
                    await self._validate(current)
                    async with client.stream("GET", current) as response:
                        if response.is_redirect:
                            if redirect == 3 or not response.headers.get("location"):
                                raise PublicWebError(
                                    "The public source exceeded the redirect limit."
                                )
                            current = urljoin(current, response.headers["location"])
                            continue
                        response.raise_for_status()
                        content_type = (
                            response.headers.get("content-type", "").split(";")[0].strip().lower()
                        )
                        if content_type not in {
                            "text/html",
                            "application/xhtml+xml",
                            "text/plain",
                            "application/json",
                        }:
                            raise PublicWebError(
                                "The source must return HTML, plain text, or JSON."
                            )
                        chunks: list[bytes] = []
                        size = 0
                        truncated = False
                        async for chunk in response.aiter_bytes():
                            remaining = MAX_BYTES - size
                            chunks.append(chunk[:remaining])
                            size += min(len(chunk), remaining)
                            if len(chunk) > remaining:
                                truncated = True
                                break
                        return str(response.url), content_type, b"".join(chunks), truncated
        except httpx.HTTPError as error:
            raise PublicWebError(f"Public retrieval failed: {type(error).__name__}.") from error
        raise PublicWebError("The public source could not be retrieved.")

    async def _retrieve(self, url: str) -> tuple[str, str, bytes, bool]:
        try:
            return await asyncio.wait_for(self._read(url), timeout=MAX_SECONDS)
        except TimeoutError as error:
            raise PublicWebError("Public retrieval exceeded its time limit.") from error

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        if not query.strip() or len(query) > 1000 or not 1 <= limit <= 5:
            raise PublicWebError("Provide a search query of 1–1000 characters and a limit of 1–5.")
        request = httpx.Request("GET", SEARCH_URL, params={"q": query})
        _, _, body, _ = await self._retrieve(str(request.url))
        parser = _Search()
        parser.feed(body.decode("utf-8", errors="replace"))
        distinct = {hit.url: hit for hit in parser.hits}
        return list(distinct.values())[:limit]

    async def fetch(self, url: str) -> RetrievedPage:
        final, kind, body, truncated = await self._retrieve(url)
        decoded = body.decode("utf-8", errors="replace")
        title = ""
        if kind in {"text/html", "application/xhtml+xml"}:
            parser = _Text()
            parser.feed(decoded)
            title = re.sub(r"\s+", " ", " ".join(parser.title)).strip()[:300]
            text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
        else:
            text = decoded.strip()
        return RetrievedPage(
            url=final,
            title=title,
            text=text[:MAX_TEXT],
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            content_sha256=hashlib.sha256(body).hexdigest(),
            truncated=truncated or len(text) > MAX_TEXT,
        )
