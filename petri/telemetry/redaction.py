"""Scrub completed spans before any owned processor can queue or export them."""

from __future__ import annotations

import json
import logging
import re
import threading
from urllib.parse import quote, unquote

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, SynchronousMultiSpanProcessor
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, Status

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


class Redactor:
    """Protect named credentials and explicitly supplied runtime values, then bound text."""

    def __init__(self, protected_values=()):
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
        if depth > MAX_DEPTH:
            counts["bounded"] += 1
            return "[TRUNCATED: nesting limit]"
        if isinstance(value, dict):
            result = {}
            for key, item in list(value.items())[:MAX_ITEMS]:
                safe_key = self.text(str(key), counts, depth=depth + 1)
                if _KEY.fullmatch(str(key).rsplit(".", 1)[-1]):
                    counts["redacted"] += 1
                    result[safe_key] = REDACTED
                else:
                    result[safe_key] = self.clean(
                        item, counts, depth=depth + 1, sdk_limit=sdk_limit
                    )
            counts["bounded"] += max(0, len(value) - MAX_ITEMS)
            return result
        if isinstance(value, (tuple, list)):
            counts["bounded"] += max(0, len(value) - MAX_ITEMS)
            return [
                self.clean(item, counts, depth=depth + 1, sdk_limit=sdk_limit)
                for item in value[:MAX_ITEMS]
            ]
        if isinstance(value, str):
            if sdk_limit is not None and len(value) >= sdk_limit:
                # SDK truncation can split a protected value before we receive it.
                counts["bounded"] += 1
                return "[TRUNCATED: SDK length limit]"
            return self.text(value, counts, depth=depth)
        if isinstance(value, bytes):
            counts["redacted"] += 1
            return "[BINARY OMITTED]"
        return value

    def text(self, value, counts, *, depth=0):
        if depth > MAX_DEPTH:
            counts["bounded"] += 1
            return "[TRUNCATED: nesting limit]"
        try:
            nested = json.loads(value) if value.lstrip().startswith(("{", "[", '"')) else None
        except (ValueError, RecursionError):
            nested = None
        if isinstance(nested, (dict, list, str)):
            return self.bound(
                json.dumps(self.clean(nested, counts, depth=depth + 1), ensure_ascii=False),
                counts,
            )
        original = value
        for secret in self._protected:
            value = value.replace(secret, REDACTED)
        value = _AUTH.sub(lambda match: match[1] + " " + REDACTED, value)

        def omit_escaped_tail(match):
            # An internal escaped quote can resemble the credential's closing quote.
            counts["bounded"] += 1
            return match["prefix"] + r"\"[REDACTED]\" [TRUNCATED: escaped credential tail]"

        value = _ESCAPED_ASSIGNMENT.sub(omit_escaped_tail, value)
        value = _ASSIGNMENT.sub(lambda match: match["prefix"] + REDACTED, value)
        value = _USERINFO.sub(lambda match: match[1] + REDACTED + "@", value)
        counts["redacted"] += int(value != original)
        return self.bound(value, counts)

    @staticmethod
    def bound(value, counts):
        if len(value) > MAX_TEXT:
            counts["bounded"] += 1
            value = value[:MAX_TEXT] + "[TRUNCATED]"
        return value

    def span(self, span, limits=None):
        counts = {"redacted": 0, "bounded": 0}
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
        return ReadableSpan(
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
