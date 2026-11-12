"""Scrub completed spans before any owned processor can queue or export them."""

from __future__ import annotations

import json
import logging
import re
import threading
from itertools import islice
from urllib.parse import quote, unquote

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, SynchronousMultiSpanProcessor
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, Status

from petri.telemetry.config import CaptureLimits

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"
MAX_TEXT = 65_536
MAX_ITEMS = 128
MAX_DEPTH = 32
_CREDENTIAL = (
    r"(?:[a-z0-9]+[-_])*"
    r"(?:authorization|api[-_]?key|password|passwd|token|secret|private[-_]key|credentials?|cookie|signature)"
)
_KEY = re.compile(rf"^{_CREDENTIAL}$", re.IGNORECASE)
_AUTH = re.compile(r"\b(Bearer|Basic)\s+[a-z0-9._~+/=%:-]+", re.IGNORECASE)
_ASSIGNMENT = re.compile(
    rf"""(?P<prefix>["']?\b{_CREDENTIAL}["']?\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s&,;}}\]]+)""",
    re.IGNORECASE,
)
_ESCAPED_ASSIGNMENT = re.compile(
    rf"""(?P<prefix>\\["']{_CREDENTIAL}\\["']\s*:\s*)[\s\S]*""",
    re.IGNORECASE,
)
_USERINFO = re.compile(r"(https?://)[^\s/@]+(?::[^\s/@]*)?@", re.IGNORECASE)


class CaptureBudgetExceeded(ValueError):
    """Drop an over-budget span before either sink receives it."""


class _Budget:
    def __init__(self, limits):
        self.limits = limits
        self.nodes = 0
        self.input_bytes = 0

    def visit(self):
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise CaptureBudgetExceeded("node budget")

    def text(self, value):
        # Count bytes in small chunks before parsing or copying large strings.
        if len(value) > self.limits.max_input_bytes - self.input_bytes:
            raise CaptureBudgetExceeded("input budget")
        for offset in range(0, len(value), 4096):
            self.input_bytes += len(value[offset:offset + 4096].encode("utf-8"))
            if self.input_bytes > self.limits.max_input_bytes:
                raise CaptureBudgetExceeded("input budget")

    def preflight_json(self, value, depth):
        # Reject broad/deep JSON before json.loads can allocate its object tree.
        quoted = escaped = primitive = False
        nesting = nodes = 0
        for char in value:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
                primitive = False
                nodes += 1
            elif char in "[{":
                nesting += 1
                nodes += 1
                primitive = False
                if depth + nesting > self.limits.max_depth:
                    raise CaptureBudgetExceeded("decoded depth budget")
            elif char in "]}":
                nesting -= 1
                primitive = False
            elif char in ",: \t\n\r":
                primitive = False
            elif not primitive:
                nodes += 1
                primitive = True
            if self.nodes + nodes > self.limits.max_nodes:
                raise CaptureBudgetExceeded("decoded node budget")

    def serialize(self, value, *, ensure_ascii=True):
        # ASCII JSON covers escaping overhead at the local JSON sink too.
        size = 0
        chunks = []
        for chunk in json.JSONEncoder(ensure_ascii=ensure_ascii).iterencode(value):
            size += len(chunk.encode("utf-8"))
            if size > self.limits.max_span_bytes:
                raise CaptureBudgetExceeded("serialized span budget")
            chunks.append(chunk)
        return "".join(chunks)

    def substitute(self, pattern, replacement, value):
        # Bound expansion before joining regex substitutions into a new string.
        chunks = []
        size = end = 0
        for match in pattern.finditer(value):
            prefix, changed = value[end:match.start()], replacement(match)
            size += len(prefix) + len(changed)
            if size > max(len(value), self.limits.max_span_bytes):
                raise CaptureBudgetExceeded("redaction expansion budget")
            chunks.extend((prefix, changed))
            end = match.end()
        if not chunks:
            return value
        size += len(value) - end
        if size > max(len(value), self.limits.max_span_bytes):
            raise CaptureBudgetExceeded("redaction expansion budget")
        chunks.append(value[end:])
        return "".join(chunks)


class Redactor:
    """Protect named credentials and explicitly supplied runtime values, then bound text."""

    def __init__(self, protected_values=(), *, limits: CaptureLimits | None = None):
        self.limits = limits
        self._lock = threading.Lock()
        self._protected: tuple[str, ...] = ()
        self.protect(protected_values)

    def protect(self, values):
        with self._lock:
            protected = set(self._protected)
            for value in values:
                if value:
                    protected.update((value, quote(value, safe=""), unquote(value)))
                    if re.match(r"^(Bearer|Basic)\s+\S", value, re.IGNORECASE):
                        protected.add(value.split(None, 1)[1])
            self._protected = tuple(sorted(protected, key=len, reverse=True))

    def clean(self, value, counts, *, depth=0, sdk_limit=None):
        budget = counts.get("_budget")
        if budget:
            budget.visit()
        max_depth = self.limits.max_depth if self.limits else MAX_DEPTH
        max_items = self.limits.max_collection_items if self.limits else MAX_ITEMS
        if depth > max_depth:
            counts["bounded"] += 1
            return "[TRUNCATED: nesting limit]"
        if isinstance(value, dict):
            result = {}
            for key, item in islice(value.items(), max_items):
                safe_key = self.text(str(key), counts, depth=depth + 1)
                if _KEY.fullmatch(str(key).rsplit(".", 1)[-1]):
                    counts["redacted"] += 1
                    result[safe_key] = REDACTED
                else:
                    result[safe_key] = self.clean(
                        item, counts, depth=depth + 1, sdk_limit=sdk_limit
                    )
            counts["bounded"] += max(0, len(value) - max_items)
            return result
        if isinstance(value, (tuple, list)):
            counts["bounded"] += max(0, len(value) - max_items)
            return [
                self.clean(item, counts, depth=depth + 1, sdk_limit=sdk_limit)
                for item in islice(value, max_items)
            ]
        if isinstance(value, str):
            if sdk_limit is not None and len(value) >= sdk_limit:
                # SDK truncation can split a protected value before we receive it.
                counts["bounded"] += 1
                return "[TRUNCATED: SDK length limit]"
            return self.text(value, counts, depth=depth, visited=True)
        if isinstance(value, bytes):
            counts["redacted"] += 1
            return "[BINARY OMITTED]"
        return value

    def text(self, value, counts, *, depth=0, visited=False):
        budget = counts.get("_budget")
        if budget:
            if not visited:
                budget.visit()
            budget.text(value)
        if depth > (self.limits.max_depth if self.limits else MAX_DEPTH):
            counts["bounded"] += 1
            return "[TRUNCATED: nesting limit]"
        if budget and value.lstrip().startswith(("{", "[", '"')):
            budget.preflight_json(value, depth)
        try:
            nested = json.loads(value) if value.lstrip().startswith(("{", "[", '"')) else None
        except (ValueError, RecursionError):
            nested = None
        if isinstance(nested, (dict, list, str)):
            cleaned = self.clean(nested, counts, depth=depth + 1)
            encoded = budget.serialize(cleaned, ensure_ascii=False) if budget else json.dumps(cleaned, ensure_ascii=False)
            if self.limits and len(encoded) > self.limits.max_text_chars:
                counts["bounded"] += 1
                return json.dumps("[TRUNCATED: text limit]")
            return self.bound(encoded, counts)
        original = value
        for secret in self._protected:
            if budget and len(value) + value.count(secret) * (len(REDACTED) - len(secret)) > max(len(value), self.limits.max_span_bytes):
                raise CaptureBudgetExceeded("redaction expansion budget")
            value = value.replace(secret, REDACTED)
        def substitute(pattern, replacement, text):
            return budget.substitute(pattern, replacement, text) if budget else pattern.sub(replacement, text)

        value = substitute(_AUTH, lambda match: match[1] + " " + REDACTED, value)

        def omit_escaped_tail(match):
            # An internal escaped quote can resemble the credential's closing quote.
            counts["bounded"] += 1
            return match["prefix"] + r"\"[REDACTED]\" [TRUNCATED: escaped credential tail]"

        value = substitute(_ESCAPED_ASSIGNMENT, omit_escaped_tail, value)
        value = substitute(_ASSIGNMENT, lambda match: match["prefix"] + REDACTED, value)
        value = substitute(_USERINFO, lambda match: match[1] + REDACTED + "@", value)
        counts["redacted"] += int(value != original)
        return self.bound(value, counts)

    def bound(self, value, counts):
        max_text = self.limits.max_text_chars if self.limits else MAX_TEXT
        if len(value) > max_text:
            counts["bounded"] += 1
            value = value[:max_text] + "[TRUNCATED]"
        return value

    def span(self, span, limits=None):
        counts = {"redacted": 0, "bounded": 0}
        if self.limits:
            counts["_budget"] = _Budget(self.limits)
        attributes = self.clean(
            dict(span.attributes or {}),
            counts,
            sdk_limit=limits.max_span_attribute_length if limits else None,
        )
        events = [
            Event(
                self.text(event.name, counts),
                self.clean(
                    dict(event.attributes or {}),
                    counts,
                    sdk_limit=limits.max_attribute_length if limits else None,
                ),
                event.timestamp,
            )
            for event in span.events
        ]
        links = [
            Link(
                link.context,
                self.clean(
                    dict(link.attributes or {}),
                    counts,
                    sdk_limit=limits.max_attribute_length if limits else None,
                ),
            )
            for link in span.links
        ]
        resource = Resource(
            self.clean(dict(span.resource.attributes), counts),
            self.text(span.resource.schema_url, counts),
        )
        scope = span.instrumentation_scope
        if scope is not None:
            scope = InstrumentationScope(
                self.text(scope.name, counts),
                self.text(scope.version, counts) if scope.version else None,
                self.text(scope.schema_url, counts) if scope.schema_url else None,
                self.clean(dict(scope.attributes or {}), counts),
            )
        description = (
            self.text(span.status.description, counts) if span.status.description else None
        )
        name = self.text(span.name, counts)
        # SDK count/length limits may have dropped content before this processor ran.
        for field in ("attributes", "events", "links"):
            dropped = getattr(span, "dropped_" + field, 0)
            if dropped:
                attributes["petri.telemetry.dropped_" + field] = dropped
        for index, event in enumerate(span.events):
            dropped = getattr(event, "dropped_attributes", 0)
            if dropped:
                attributes[f"petri.telemetry.event_{index}_dropped_attributes"] = dropped
        dropped_link_attributes = sum(getattr(link, "dropped_attributes", 0) for link in span.links)
        if dropped_link_attributes:
            attributes["petri.telemetry.dropped_link_attributes"] = dropped_link_attributes
        if counts["redacted"]:
            attributes["petri.telemetry.redacted"] = True
        if counts["bounded"]:
            attributes["petri.telemetry.bounded_values"] = counts["bounded"]
        sanitized = ReadableSpan(
            name=name,
            context=span.context,
            parent=span.parent,
            resource=resource,
            attributes=attributes,
            events=events,
            links=links,
            kind=span.kind,
            status=Status(span.status.status_code, description),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=scope,
        )
        if self.limits:
            counts["_budget"].serialize(self._payload(sanitized))
        return sanitized

    @staticmethod
    def _payload(span):
        # Include identity, framing and markers without changing either sink's schema.
        def context(value):
            if value is None:
                return None
            return {"trace_id": f"{value.trace_id:032x}", "span_id": f"{value.span_id:016x}",
                    "trace_flags": int(value.trace_flags), "trace_state": value.trace_state.to_header(),
                    "is_remote": value.is_remote}

        scope = span.instrumentation_scope
        return {
            "name": span.name, "context": context(span.context), "parent": context(span.parent),
            "kind": span.kind.name, "start_time_ns": span.start_time, "end_time_ns": span.end_time,
            "status": {"code": span.status.status_code.name, "description": span.status.description},
            "attributes": dict(span.attributes or {}),
            "events": [{"name": e.name, "timestamp": e.timestamp,
                        "attributes": dict(e.attributes or {})} for e in span.events],
            "links": [{"context": context(link.context), "attributes": dict(link.attributes or {})}
                      for link in span.links],
            "resource": {"attributes": dict(span.resource.attributes), "schema_url": span.resource.schema_url},
            "scope": None if scope is None else {"name": scope.name, "version": scope.version,
                      "schema_url": scope.schema_url, "attributes": dict(scope.attributes or {})},
        }



class RedactingSpanProcessor(SynchronousMultiSpanProcessor):
    """Give every downstream processor the same sanitized completed span."""

    def __init__(self, redactor, limits=None):
        super().__init__()
        self.redactor = redactor
        self.limits = limits
        self._warned = False

    def on_end(self, span: ReadableSpan):
        try:
            sanitized = self.redactor.span(span, self.limits)
        except Exception as error:  # noqa: BLE001 — never export a span that failed redaction
            if not self._warned:
                logger.warning(
                    "Telemetry span dropped during redaction (%s).", type(error).__name__
                )
                self._warned = True
            return
        super().on_end(sanitized)
