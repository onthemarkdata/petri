"""Loopback-only transport and public identities for opt-in local diagnostics."""

import base64
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import stat
import subprocess
import threading
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOCK = Path(__file__).resolve().parents[2] / "models/gemma.lock.json"
REPO = LOCK.parents[1]
CORRELATION_HEADERS = frozenset({
    "traceparent", "tracestate", "x-request-id", "x-petri-run-id",
    "x-petri-workflow-id", "x-petri-cell-id",
})


def correlation_headers(headers, additional=()):
    """Copy supplied correlation values, excluding authentication and cookies."""
    allowed = CORRELATION_HEADERS | {name.lower() for name in additional}
    selected = {}
    for name, value in headers.items():
        name = name.lower()
        if (name not in allowed or "authorization" in name or "cookie" in name
                or name in {"x-api-key", "api-key", "www-authenticate", "proxy-authenticate"}):
            continue
        if not isinstance(value, str) or len(value) > 4096 or any(
            ord(char) < 32 or ord(char) > 126 for char in value
        ):
            raise ValueError("Invalid caller correlation header.")
        selected[name] = value
    return selected


class ResponseRejected(RuntimeError):
    def __init__(self, message, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(message)


def loopback_url(value):
    """Validate without resolving a remote host or including unsafe input in errors."""
    parsed = None
    try:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        port = parsed.port
    except (ValueError, TypeError):
        local, port = False, None
    if (
        not local or parsed.scheme != "http" or parsed.username is not None
        or parsed.password is not None or parsed.query or parsed.fragment
        or (port is not None and port == 0)
        or any(c.isspace() or ord(c) < 32 for c in value)
        or "\\" in value
        or any(part in {".", ".."} for part in urllib.parse.unquote(parsed.path).split("/"))
        or "%" in parsed.netloc
    ):
        raise ValueError("Use a local HTTP loopback endpoint without credentials, query, or fragment.")
    return value.rstrip("/")


class EndpointHTTPError(RuntimeError):
    def __init__(self, status, body):
        self.status = status
        super().__init__(f"Local endpoint returned HTTP {status}: {body}")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Local endpoint redirects are disabled.")


def configuration_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def connection_arguments(parser):
    parser.add_argument("--adapter", choices=["dmr", "openai"])
    parser.add_argument("--url", help="Legacy DMR management URL; use --base-url for generic servers.")
    parser.add_argument("--base-url", help="Exact loopback OpenAI API base, including its path prefix.")
    parser.add_argument("--model", help="Model ID sent to the endpoint; this may be a gateway alias.")
    parser.add_argument("--provenance", type=Path, help="JSON manifest declaring artifact and revision.")
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--lock", type=Path, default=LOCK)
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--no-auth", action="store_true", help="Confirm this route requires no auth.")
    auth.add_argument("--api-key-env", help="Environment variable holding this route's bearer key.")
    auth.add_argument("--basic-auth-file", type=Path, help="Owned, private TOML username/password file.")


def endpoint_from_args(args):
    adapter = args.adapter or ("openai" if args.base_url or args.model or args.provenance else "dmr")
    if adapter == "openai":
        if args.url or not all((args.base_url, args.model, args.provenance, args.context_size)):
            raise ValueError("Generic endpoints require --base-url, --model, --provenance, --context-size.")
        manifest = json.loads(args.provenance.read_text())
        lock = None
        base = args.base_url
    else:
        lock = json.loads(args.lock.read_text())
        manifest = lock
        if args.model and args.model != lock["model"]:
            raise ValueError("The DMR model must match its lock.")
        base = args.base_url or (args.url or "http://127.0.0.1:12434") + "/engines/llama.cpp/v1"
    credential = args.basic_auth_file
    if not (args.no_auth or args.api_key_env or credential):
        if manifest.get("credential_file"):
            credential = Path(manifest["credential_file"])
        elif os.environ.get("OPENAI_API_KEY"):
            args.api_key_env = "OPENAI_API_KEY"
        elif adapter != "dmr":
            raise ValueError("Select --no-auth, --api-key-env, or --basic-auth-file for this route.")
    return LocalEndpoint(
        base, args.model or lock["model"], args.context_size or lock["context_size"],
        args.max_tokens, manifest, adapter=adapter, lock=lock,
        api_key_env=args.api_key_env, basic_auth_file=credential,
        no_auth=args.no_auth or (adapter == "dmr" and not (credential or args.api_key_env)),
        management_url=args.url,
    )


class LocalEndpoint:
    def __init__(self, base_url, model, context_size, max_tokens, manifest, *, adapter="openai",
                 lock=None, api_key_env=None, basic_auth_file=None, no_auth=False,
                 management_url=None):
        self.base_url = loopback_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A nonempty model ID is required.")
        if not 0 < max_tokens <= context_size:
            raise ValueError("Require 0 < max-tokens <= context-size.")
        if adapter not in {"dmr", "openai"}:
            raise ValueError("Choose the dmr or openai adapter.")
        self.model, self.context_size, self.max_tokens = model, context_size, max_tokens
        self.adapter, self.lock = adapter, lock
        self.management_url = loopback_url(management_url) if management_url else None
        if adapter == "dmr" and self.management_url is None:
            suffix = "/engines/llama.cpp/v1"
            if not self.base_url.endswith(suffix):
                raise ValueError("DMR requires its llama.cpp API base or --url management base.")
            self.management_url = self.base_url[:-len(suffix)]
        artifact = manifest.get("artifact") or manifest.get("source")
        revision = manifest.get("revision")
        if not artifact or not revision:
            raise ValueError("Provenance requires an artifact (or source) and revision.")
        self.provenance = {
            "artifact": artifact, "revision": revision,
            "declared_model": manifest.get("model_name", manifest.get("model", model)),
            "serving_backend": manifest.get("provider", adapter),
            "weight_verification": "operator declaration; not independently verified",
        }
        # Names and aliases are declarations, never evidence of a weight digest.
        for key in ("files", "dmr_id", "weights_sha256", "response_model"):
            if key in manifest:
                self.provenance[key] = manifest[key]
        self.response_model = manifest.get("response_model", model)
        context_declarations = {key: manifest[key] for key in ("max_model_len", "context_size")
                                if key in manifest}
        if context_declarations:
            if any(value != context_size for value in context_declarations.values()):
                raise ValueError("--context-size must match every supplied manifest context declaration.")
            self.provenance["configured_context_size"] = context_size
            self.provenance["context_declarations"] = context_declarations
        self.request_options = {}
        template_options = manifest.get("chat_template_kwargs")
        if template_options is None and adapter == "dmr":
            template_options = {"enable_thinking": False}
        if template_options is not None:
            if not isinstance(template_options, dict):
                raise ValueError("chat_template_kwargs must be a JSON object.")
            self.request_options["chat_template_kwargs"] = template_options
        self.api_key_env = api_key_env
        self.authorization = None
        self.secret_values = []
        self.username = None
        if sum(bool(x) for x in (api_key_env, basic_auth_file, no_auth)) != 1:
            raise ValueError("Choose exactly one authentication source or confirm --no-auth.")
        self.auth_kind = "none" if no_auth else "basic" if basic_auth_file else "bearer"
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ValueError("The selected API key environment variable is empty.")
            self.authorization = "Bearer " + key
            self.secret_values.append(key)
        if basic_auth_file:
            try:
                with Path(basic_auth_file).open("rb") as stream:
                    info = os.fstat(stream.fileno())
                    if (Path(basic_auth_file).is_symlink() or not stat.S_ISREG(info.st_mode)
                            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
                        raise ValueError("Basic auth file must be owned by this user with private permissions.")
                    credentials = tomllib.load(stream)
                username, password = credentials["username"], credentials["password"]
                if not username or not password or ":" in username:
                    raise ValueError
                token = base64.b64encode(f"{username}:{password}".encode()).decode()
            except (OSError, ValueError, KeyError, TypeError):
                raise ValueError("Cannot load an owned, private Basic auth TOML file.") from None
            self.authorization = "Basic " + token
            self.username = username
            self.secret_values.extend([password, token])
        if self.authorization:
            self.secret_values.append(self.authorization)
        self._request_lock = threading.Lock()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def _literal_redact(self, text):
        for value in sorted(self.secret_values, key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
        return text

    def diagnostic(self, value):
        """Copy decoded values, preserving keys and configured public model identifiers.

        A Basic username alone is not a secret substring: it can be a protocol word
        or ordinary prose. Its explicit username field is removed from diagnostics.
        Passwords, keys and authorization values remain protected in text values.
        """
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key.lower() == "username" and self.username is not None and item == self.username:
                    result[key] = "[REDACTED]"
                elif key in {"model", "response_model", "declared_model", "id", "name"} and (
                    isinstance(item, str) and item in {self.model, self.response_model}
                ):
                    result[key] = item
                else:
                    result[key] = self.diagnostic(item)
            return result
        if isinstance(value, (list, tuple)):
            return [self.diagnostic(item) for item in value]
        return self.redact(value) if isinstance(value, str) else value

    def redact(self, text):
        """Sanitize diagnostic text, decoding JSON and JSON embedded in text first."""
        decoder = json.JSONDecoder()
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            pass
        else:
            if isinstance(parsed, (dict, list, str)):
                safe = self.diagnostic(parsed)
                return json.dumps(safe, ensure_ascii=False) if safe != parsed else text
        parts, start, index = [], 0, 0
        while index < len(text):
            if text[index] not in '{["':
                index += 1
                continue
            try:
                parsed, end = decoder.raw_decode(text, index)
            except ValueError:
                index += 1
                continue
            safe = self.diagnostic(parsed)
            if isinstance(parsed, (dict, list)) or safe != parsed:
                parts.append(self._literal_redact(text[start:index]))
                parts.append(json.dumps(safe, ensure_ascii=False) if safe != parsed else text[index:end])
                start = index = end
            else:
                index += 1
        parts.append(self._literal_redact(text[start:]))
        return "".join(parts)

    def diagnostic_json(self, value, **options):
        return json.dumps(self.diagnostic(value), **options)

    def reject_credential_echo(self, value):
        safe = self.diagnostic(value)
        if safe != value:
            raise ResponseRejected(
                "Local endpoint echoed protected authentication data; response was rejected.", safe
            )

    def identity(self):
        return {
            "adapter": self.adapter, "model": self.model, "base_url": self.base_url,
            "context_size": self.context_size, "max_tokens": self.max_tokens,
            "authentication": self.auth_kind, "provenance": self.provenance,
            "request_options": self.request_options,
        }

    def raw_request(self, url, payload=None, *, headers=None, correlation_allowlist=()):
        url = loopback_url(url)
        headers = {"Content-Type": "application/json",
                   **correlation_headers(headers or {}, correlation_allowlist)}
        if self.authorization:
            headers["Authorization"] = self.authorization
        request = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with self._request_lock, self.opener.open(request, timeout=300) as response:
                if 300 <= response.status < 400:
                    raise RuntimeError("Local endpoint redirects are disabled.")
                body = response.read()
                return response.status, response.headers.get("Content-Type", "application/json"), body
        except urllib.error.HTTPError as error:
            if 300 <= error.code < 400:
                raise RuntimeError("Local endpoint redirects are disabled.") from None
            return error.code, error.headers.get("Content-Type", "application/json"), error.read()
        except urllib.error.URLError:
            raise RuntimeError("Local endpoint connection failed.") from None

    def request(self, path, payload=None, *, management=False, headers=None):
        base = self.management_url if management else self.base_url
        status, _, body = self.raw_request(
            base + path, None if payload is None else json.dumps(payload).encode(),
            **({"headers": headers} if headers else {}),
        )
        if status >= 400:
            raise EndpointHTTPError(status, self.redact(body.decode(errors="replace")))
        result = json.loads(body) if body else None
        self.reject_credential_echo(result)
        return result

    def pi_document(self, base_url=None, key_env="PETRI_LOCAL_API_KEY"):
        provider = {
            "baseUrl": loopback_url(base_url or self.base_url), "api": "openai-completions",
            "apiKey": "local-no-auth" if self.auth_kind == "none" else (self.api_key_env or key_env),
            "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                       "supportsReasoningEffort": False, "maxTokensField": "max_tokens"},
            "models": [{"id": self.model, "name": self.model, "reasoning": False,
                        "input": ["text"], "contextWindow": self.context_size,
                        "maxTokens": self.max_tokens,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
        }
        if self.auth_kind == "basic":
            provider["authHeader"] = False
            provider["headers"] = {"Authorization": "PETRI_LOCAL_AUTHORIZATION"}
        return {"providers": {"petri-local": provider}}

    def pi_config(self, directory, **options):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "models.json").open("x") as stream:
            json.dump(self.pi_document(**options), stream, indent=2)
            stream.write("\n")

    def validate_response_identity(self, body, content_type):
        """Require original response identities and successful terminal completions."""
        stream_diagnostic = {"stream": "Raw SSE omitted because logical assembly did not complete."}
        try:
            self._validate_response(body, content_type, stream_diagnostic)
        except (ValueError, TypeError, KeyError) as error:
            diagnostic = (self.diagnostic_json(stream_diagnostic) if "text/event-stream" in content_type
                          else self.redact(body.decode(errors="replace")))
            raise ResponseRejected(str(error), diagnostic) from None

    def _stream_messages(self, responses):
        """Assemble client-visible delta strings for inspection, never for forwarding."""
        messages, tools, aliases = {}, {}, {}
        errors = []
        for response in responses:
            if not isinstance(response, dict):
                continue
            if response.get("error"):
                errors.append(response["error"])
            choices = response.get("choices", [])
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                index = choice.get("index", 0)
                if type(index) is not int or index < 0:
                    continue
                message = messages.setdefault(index, {"content": "", "tool_calls": []})
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise TypeError("Invalid streamed completion delta.")
                for key, value in delta.items():
                    if isinstance(value, str):
                        message[key] = message.get(key, "") + value
                tool_calls = delta.get("tool_calls", [])
                if not isinstance(tool_calls, list):
                    raise TypeError("Invalid streamed tool calls.")
                for tool in tool_calls:
                    if not isinstance(tool, dict) or not isinstance(tool.get("function", {}), dict):
                        raise TypeError("Invalid streamed tool call.")
                    tool_index, tool_id = tool.get("index"), tool.get("id")
                    if type(tool_index) is int and tool_index >= 0:
                        identity = (index, "index", tool_index)
                    elif isinstance(tool_id, str) and tool_id:
                        identity = aliases.get((index, tool_id), (index, "id", tool_id))
                    else:
                        raise ValueError("Streamed tool arguments lack a tool identity.")
                    if tool_id:
                        aliases[index, tool_id] = identity
                    if identity not in tools:
                        tools[identity] = {"function": {"name": "", "arguments": ""}}
                        message["tool_calls"].append(tools[identity])
                    assembled = tools[identity]
                    if tool_id:
                        assembled["id"] = tool_id
                    for key, value in tool.get("function", {}).items():
                        if isinstance(value, str):
                            assembled["function"][key] = assembled["function"].get(key, "") + value
        return {"choices": [{"index": index, "message": message}
                            for index, message in messages.items()], "errors": errors}

    def _validate_response(self, body, content_type, stream_diagnostic):
        streaming = "text/event-stream" in content_type
        done = False
        if streaming:
            responses, data_lines = [], []
            for line in body.decode("utf-8-sig").splitlines():
                if line == "":
                    if not data_lines:
                        continue
                    data = "\n".join(data_lines)
                    data_lines = []
                    if done:
                        raise ValueError("SSE data followed its terminal marker.")
                    if data == "[DONE]":
                        done = True
                    else:
                        responses.append(json.loads(data))
                elif line == "data" or line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" ") if line.startswith("data:") else "")
            logical = self._stream_messages(responses)
            self.reject_credential_echo(logical)
            stream_diagnostic.clear()
            stream_diagnostic.update(logical)
            if data_lines or not done:
                raise ValueError("SSE ended before its terminal marker.")
        else:
            responses = [json.loads(body)]
        seen, finished = set(), set()
        for response in responses:
            if not isinstance(response, dict) or response.get("error"):
                raise ValueError("Local endpoint returned a failed completion.")
            choices = response.get("choices")
            usage_only = streaming and choices == [] and isinstance(response.get("usage"), dict)
            if (response.get("model") != self.response_model
                    and not (usage_only and "model" not in response)):
                raise ValueError("Completion model identity is missing or differs from the declared response model.")
            self.reject_credential_echo(response)
            if usage_only:
                continue
            if not isinstance(choices, list) or not choices:
                raise ValueError("Completion choices are missing.")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise TypeError("Invalid completion choice.")
                index = choice.get("index", 0)
                if type(index) is not int or index < 0 or index in finished:
                    raise ValueError("Invalid or already completed response choice.")
                seen.add(index)
                reason = choice.get("finish_reason")
                if reason in {"stop", "tool_calls"}:
                    finished.add(index)
                elif reason is not None or not streaming:
                    raise ValueError("Local endpoint returned a failed or incomplete completion.")
        if not seen or seen != finished:
            raise ValueError("Completion ended before a successful terminal finish reason.")

    @contextlib.contextmanager
    def guarded_route(self, *, correlation_allowlist=()):
        """Serve only this model's completion path; upstream redirects never reach clients.

        The relay buffers responses, including SSE, and keeps upstream auth in memory.
        Its short-lived client key is passed through the environment, never an artifact.
        """
        endpoint, token = self, secrets.token_urlsafe(32)
        self.secret_values.extend([token, "Bearer " + token])
        expected_path = urllib.parse.urlsplit(self.base_url).path + "/chat/completions"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                    self.send_error(403)
                    return
                if self.path != expected_path:
                    self.send_error(404)
                    return
                try:
                    payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    data = json.loads(payload)
                    if data.get("model") != endpoint.model:
                        raise ValueError("Unexpected model ID.")
                    for field in ("max_tokens", "max_completion_tokens"):
                        if field in data and not 0 < data[field] <= endpoint.max_tokens:
                            raise ValueError("Response cap exceeds the selected limit.")
                    for field, value in endpoint.request_options.items():
                        if field in data and data[field] != value:
                            raise ValueError("Request options differ from the selected configuration.")
                        data[field] = value
                    if not any(field in data for field in ("max_tokens", "max_completion_tokens")):
                        data["max_tokens"] = endpoint.max_tokens
                    status, content_type, body = endpoint.raw_request(
                        endpoint.base_url + "/chat/completions", json.dumps(data).encode(),
                        headers=correlation_headers(self.headers, correlation_allowlist),
                        correlation_allowlist=correlation_allowlist,
                    )
                    if status < 400:
                        endpoint.validate_response_identity(body, content_type)
                    else:
                        body = endpoint.redact(body.decode(errors="replace")).encode()
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                    status, content_type = 502, "application/json"
                    diagnostic = getattr(error, "diagnostic", None)
                    body = endpoint.diagnostic_json({
                        "error": endpoint.redact(str(error)), "diagnostic_response": diagnostic,
                    }).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}" + urllib.parse.urlsplit(self.base_url).path
            yield base, token
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


@contextlib.contextmanager
def environment_values(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def save_identity(path, identity, *, resume):
    if resume:
        if not path.exists() or json.loads(path.read_text()) != identity:
            raise ValueError("Saved run identity differs; use the original model/backend/case/configuration.")
    else:
        with path.open("x") as stream:
            json.dump(identity, stream, indent=2)
            stream.write("\n")
