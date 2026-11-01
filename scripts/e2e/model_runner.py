"""Opt-in local Gemma readiness checks; these do not validate Petri workflows."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LOCK = Path(__file__).resolve().parents[2] / "models/gemma.lock.json"


class ModelRunner:
    def __init__(self, url, lock):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use a local HTTP Model Runner endpoint.")
        self.url = url.rstrip("/")
        self.lock = lock
        self.model = lock["model"]

    def request(self, path, payload=None):
        request = urllib.request.Request(
            self.url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()
            return json.loads(body) if body else None

    def verify(self):
        metadata = self.request("/models/" + self.model)
        if metadata["id"] != self.lock["dmr_id"]:
            raise RuntimeError(
                "Model digest differs from the lock; do not test a substituted model."
            )
        return metadata

    def configure(self):
        self.verify()
        return self.request(
            "/engines/llama.cpp/_configure",
            {
                "model": self.model,
                "context-size": self.lock["context_size"],
                "llamacpp": {"reasoning-budget": 0},
                "runtime-flags": ["--jinja", "--chat-template-kwargs", '{"enable_thinking":false}'],
            },
        )

    def complete(self, messages, **options):
        started = time.monotonic()
        result = self.request(
            "/engines/llama.cpp/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 256,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                **options,
            },
        )
        choice = result["choices"][0]
        if choice["finish_reason"] not in {"stop", "tool_calls"}:
            raise RuntimeError(f"Incomplete model response: {choice['finish_reason']}")
        return {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "message": choice["message"],
            "finish_reason": choice["finish_reason"],
            "usage": result.get("usage", {}),
        }

    def check(self, runs):
        self.verify()
        records = []
        for number in range(runs):
            record = self.complete([{"role": "user", "content": "Reply with exactly READY."}])
            if record["message"].get("content", "").strip() != "READY":
                raise RuntimeError("Readiness response must be exactly READY.")
            records.append({"check": "completion", "run": number + 1, **record})
        record = self.complete(
            [{"role": "user", "content": 'Return a JSON object with status "ready" and count 3.'}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "readiness",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["ready"]},
                            "count": {"type": "integer", "enum": [3]},
                        },
                        "required": ["status", "count"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        if json.loads(record["message"]["content"]) != {"status": "ready", "count": 3}:
            raise RuntimeError("Structured response did not match the expected result.")
        records.append({"check": "json_schema", **record})
        messages = [{"role": "user", "content": "Use lookup_source to look up source alpha."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_source",
                    "description": "Retrieve a public test source by identifier.",
                    "parameters": {
                        "type": "object",
                        "properties": {"source_id": {"type": "string"}},
                        "required": ["source_id"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        record = self.complete(messages, tools=tools, tool_choice="required")
        calls = record["message"].get("tool_calls", [])
        if len(calls) != 1 or calls[0]["function"]["name"] != "lookup_source":
            raise RuntimeError("Expected one lookup_source tool call.")
        if json.loads(calls[0]["function"]["arguments"]) != {"source_id": "alpha"}:
            raise RuntimeError("Tool arguments did not match the requested source.")
        records.append({"check": "tool_call", **record})
        messages += [
            record["message"],
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": "The verification code is PETRI42.",
            },
            {"role": "user", "content": "Reply with only the verification code."},
        ]
        record = self.complete(messages, tools=tools, tool_choice="none")
        if record["message"].get("content", "").strip() != "PETRI42":
            raise RuntimeError("Tool-result replay did not preserve the source content.")
        records.append({"check": "tool_result", **record})
        return records

    def pi_config(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "models.json"
        config = {
            "providers": {
                "petri-local": {
                    "baseUrl": self.url + "/engines/llama.cpp/v1",
                    "api": "openai-completions",
                    "apiKey": "local-no-auth",
                    "compat": {
                        "supportsStore": False,
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "maxTokensField": "max_tokens",
                    },
                    "models": [
                        {
                            "id": self.model,
                            "name": "Google Gemma 4 E2B QAT Q4_0",
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": self.lock["context_size"],
                            "maxTokens": 4096,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        }
                    ],
                }
            }
        }
        # Refuse to overwrite an existing personal or test configuration.
        with destination.open("x") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")

    def pi_check(self, executable):
        with tempfile.TemporaryDirectory(prefix="petri-model-check-") as directory:
            root = Path(directory)
            self.pi_config(root / "agent")
            (root / "source.txt").write_text("The verification code is PETRI42.\n")
            environment = {
                **os.environ,
                "PI_CODING_AGENT_DIR": str(root / "agent"),
                "PI_TELEMETRY": "0",
                "PI_OFFLINE": "1",
            }
            started = time.monotonic()
            run = subprocess.run(
                [
                    executable,
                    "--provider",
                    "petri-local",
                    "--model",
                    self.model,
                    "--no-session",
                    "--no-extensions",
                    "--no-skills",
                    "--no-prompt-templates",
                    "--no-themes",
                    "--no-context-files",
                    "--thinking",
                    "off",
                    "--tools",
                    "read",
                    "--mode",
                    "json",
                    "--system-prompt",
                    "Follow the user's instructions precisely.",
                    "-p",
                    "Read source.txt with the read tool. Reply with only its verification code.",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            events = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
            ends = [event for event in events if event.get("type") == "agent_end"]
            if not ends:
                raise RuntimeError("pi did not return an agent_end event.")
            messages = ends[-1]["messages"]
            calls = [
                block
                for msg in messages
                if msg.get("role") == "assistant"
                for block in msg.get("content", [])
                if block.get("type") == "toolCall"
            ]
            finals = [msg for msg in messages if msg.get("role") == "assistant"]
            answer = "".join(
                block.get("text", "")
                for block in finals[-1]["content"]
                if block.get("type") == "text"
            ).strip()
            if answer != "PETRI42" or not any(call.get("name") == "read" for call in calls):
                raise RuntimeError("pi must read the fixture and return its verification code.")
            return {
                "check": "pi_read_tool",
                "answer": answer,
                "tool_names": [call["name"] for call in calls],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "configure", "check", "pi-config", "pi-check"])
    parser.add_argument("--url", default="http://127.0.0.1:12434")
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pi", default="pi")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if not 1 <= args.runs <= 20:
        parser.error("--runs must be between 1 and 20")
    client = ModelRunner(args.url, json.loads(args.lock.read_text()))
    if args.action == "pi-config":
        if not args.directory:
            parser.error("pi-config requires --directory pointing to a new test configuration")
        client.pi_config(args.directory)
        return
    if args.action == "verify":
        result = client.verify()
    elif args.action == "configure":
        result = client.configure()
    elif args.action == "pi-check":
        client.verify()
        result = client.pi_check(args.pi)
    else:
        result = client.check(args.runs)
    evidence = {
        "scope": "model readiness, not Petri end-to-end validation",
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "model": client.model,
        "dmr_id": client.lock["dmr_id"],
        "result": result,
    }
    rendered = json.dumps(evidence, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Model check failed: {error}", file=sys.stderr)
        sys.exit(1)
