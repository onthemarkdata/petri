"""Opt-in local model readiness checks; these do not validate Petri workflows."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Direct script execution also works without an installed Petri package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.e2e.local_endpoint import (
    LocalEndpoint,
    configuration_hash,
    connection_arguments,
    endpoint_from_args,
    source_sha,
)


class ModelRunner:
    def __init__(self, url=None, lock=None, *, endpoint=None):
        if endpoint is None:
            # Retain the original Python constructor for the DMR regression tests.
            from scripts.e2e.local_endpoint import loopback_url

            loopback_url(url)
            manifest = {"artifact": lock.get("source", lock.get("model")),
                        "revision": lock.get("revision", "unspecified"), **lock}
            endpoint = LocalEndpoint(
                url.rstrip("/") + "/engines/llama.cpp/v1", lock["model"],
                lock["context_size"], 4096, manifest, adapter="dmr", lock=lock,
                no_auth=True, management_url=url,
            )
        self.endpoint = endpoint
        self.url, self.lock, self.model = endpoint.base_url, endpoint.lock, endpoint.model
        self.records = []
        self.last_response = None

    def request(self, path, payload=None):
        return self.endpoint.request(
            path, payload, management=path.startswith(("/models/", "/engines/"))
        )

    def verify(self):
        if self.endpoint.adapter != "dmr":
            return {"provenance": self.endpoint.provenance,
                    "verification": "declared identity; completion responses checked separately"}
        metadata = self.request("/models/" + self.model)
        if metadata["id"] != self.lock["dmr_id"]:
            raise RuntimeError(
                "Model digest differs from the lock; do not test a substituted model."
            )
        return metadata

    def configure(self):
        if self.endpoint.adapter != "dmr":
            raise ValueError("configure is available only for the DMR adapter.")
        self.verify()
        return self.request(
            "/engines/llama.cpp/_configure",
            {
                "model": self.model,
                "context-size": self.endpoint.context_size,
                "llamacpp": {"reasoning-budget": 0},
                "runtime-flags": ["--jinja", "--chat-template-kwargs", '{"enable_thinking":false}'],
            },
        )

    def complete(self, messages, **options):
        started = time.monotonic()
        result = self.request(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.endpoint.max_tokens,
                "stream": False,
                **self.endpoint.request_options,
                **options,
            },
        )
        self.last_response = result
        choice = result["choices"][0]
        if choice["finish_reason"] not in {"stop", "tool_calls"}:
            raise RuntimeError(f"Incomplete model response: {choice['finish_reason']}")
        if result.get("model") != self.endpoint.response_model:
            raise RuntimeError("Completion model identity is missing or differs from the declared response model.")
        return {
            "raw_response": result,
            "response_model": result["model"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "message": choice["message"],
            "finish_reason": choice["finish_reason"],
            "usage": result.get("usage", {}),
        }

    def check(self, runs):
        self.verify()
        self.records = []

        def probe(name, operation):
            self.last_response = None
            try:
                record = operation()
                self.records.append({"check": name, "outcome": "passed", **record})
                return record
            except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
                self.records.append({"check": name,
                                     "outcome": "unsupported" if getattr(error, "status", None) == 501 else "failed",
                                     "http_status": getattr(error, "status", None),
                                     "error": self.endpoint.redact(str(error)),
                                     "response": getattr(error, "diagnostic", self.last_response)})
                return None

        def plain():
            record = self.complete([{"role": "user", "content": "Reply with exactly READY."}])
            if record["message"].get("content", "").strip() != "READY":
                raise RuntimeError("Readiness response must be exactly READY.")
            return record

        for _ in range(runs):
            probe("completion", plain)

        def structured():
            record = self.complete(
                [{"role": "user", "content": 'Return a JSON object with status "ready" and count 3.'}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "readiness", "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"status": {"type": "string", "enum": ["ready"]},
                                           "count": {"type": "integer", "enum": [3]}},
                            "required": ["status", "count"], "additionalProperties": False,
                        },
                    },
                },
            )
            if json.loads(record["message"]["content"]) != {"status": "ready", "count": 3}:
                raise RuntimeError("Structured response did not match the expected result.")
            return record

        probe("json_schema", structured)
        messages = [{"role": "user", "content": "Use lookup_source to look up source alpha."}]
        tools = [{"type": "function", "function": {
            "name": "lookup_source", "description": "Retrieve a public test source by identifier.",
            "parameters": {"type": "object", "properties": {"source_id": {"type": "string"}},
                           "required": ["source_id"], "additionalProperties": False},
        }}]

        def tool_call():
            record = self.complete(messages, tools=tools, tool_choice="required")
            calls = record["message"].get("tool_calls", [])
            if (len(calls) != 1 or calls[0]["function"]["name"] != "lookup_source"
                    or not calls[0].get("id")):
                raise RuntimeError("Expected one lookup_source tool call with an ID.")
            if json.loads(calls[0]["function"]["arguments"]) != {"source_id": "alpha"}:
                raise RuntimeError("Tool arguments did not match the requested source.")
            return record

        call = probe("tool_call", tool_call)
        if call:
            messages += [call["message"], {
                "role": "tool", "tool_call_id": call["message"]["tool_calls"][0]["id"],
                "content": "The verification code is PETRI42.",
            }, {"role": "user", "content": "Reply with only the verification code."}]

            def replay():
                record = self.complete(messages, tools=tools, tool_choice="none")
                if record["message"].get("content", "").strip() != "PETRI42":
                    raise RuntimeError("Tool-result replay did not preserve the source content.")
                return record

            probe("tool_result", replay)
        else:
            self.records.append({"check": "tool_result", "outcome": "blocked",
                                 "error": "Tool call did not pass; replay was not attempted."})
        failures = [record for record in self.records if record["outcome"] != "passed"]
        if failures:
            raise RuntimeError("; ".join(record["error"] for record in failures))
        return self.records

    def pi_config(self, directory):
        self.endpoint.pi_config(directory)

    def pi_check(self, executable):
        with self.endpoint.guarded_route() as (base_url, token), tempfile.TemporaryDirectory(
            prefix="petri-model-check-"
        ) as directory:
            root = Path(directory)
            document = self.endpoint.pi_document(base_url, key_env="PETRI_LOCAL_RELAY_KEY")
            provider = document["providers"]["petri-local"]
            provider.update(apiKey="PETRI_LOCAL_RELAY_KEY", authHeader=True)
            provider.pop("headers", None)
            (root / "agent").mkdir()
            (root / "agent/models.json").write_text(json.dumps(document))
            (root / "source.txt").write_text("The verification code is PETRI42.\n")
            environment = {
                **os.environ,
                "PI_CODING_AGENT_DIR": str(root / "agent"),
                "PETRI_LOCAL_RELAY_KEY": token,
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
                check=False,
            )
            self.records = [{"check": "pi_read_tool", "exit_code": run.returncode,
                             "stdout": self.endpoint.redact(run.stdout),
                             "stderr": self.endpoint.redact(run.stderr)}]
            if run.returncode:
                raise RuntimeError(f"pi exited {run.returncode}: " + self.endpoint.redact(run.stderr))
            events = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
            ends = [event for event in events if event.get("type") == "agent_end"]
            if not ends:
                raise RuntimeError("pi did not return an agent_end event.")
            messages = ends[-1]["messages"]
            assistants = [message for message in messages if message.get("role") == "assistant"]
            if any(message.get("stopReason") in {"error", "length", "aborted"} for message in assistants):
                raise RuntimeError("pi returned a failed or incomplete assistant response.")
            if (not assistants or assistants[-1].get("stopReason") != "stop"
                    or any(message.get("stopReason") not in {"stop", "toolUse"} for message in assistants)):
                raise RuntimeError("pi did not prove a successful terminal assistant response.")
            if any(message.get("model") != self.model for message in assistants):
                raise RuntimeError("pi assistant model identity is missing or differs from the selected model.")
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
            reads = [call for call in calls if call.get("name") == "read"
                     and call.get("arguments", {}).get("path") == "source.txt"]
            replayed = any(
                message.get("role") == "toolResult" and not message.get("isError")
                and message.get("toolCallId") in {call.get("id") for call in reads}
                and "PETRI42" in json.dumps(message.get("content", []))
                for message in messages
            )
            if answer != "PETRI42" or not reads or not replayed:
                raise RuntimeError("pi must read the fixture and return its verification code.")
            return {
                "check": "pi_read_tool",
                "outcome": "passed",
                "exit_code": run.returncode,
                "diagnostics": self.records,
                "answer": answer,
                "tool_names": [call["name"] for call in calls],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "configure", "check", "pi-config", "pi-check"])
    connection_arguments(parser)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pi", default="pi")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if not 1 <= args.runs <= 20:
        parser.error("--runs must be between 1 and 20")
    endpoint = endpoint_from_args(args)
    client = ModelRunner(endpoint=endpoint)
    if args.action == "pi-config" and not args.directory:
        parser.error("pi-config requires --directory pointing to a new test configuration")
    evidence = {
        "scope": "model readiness, not Petri end-to-end validation",
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "identity": endpoint.identity(), "source_sha": source_sha(),
        "configuration_hash": configuration_hash(endpoint.identity()),
        "command": {"action": args.action, "runs": args.runs, "backend": "pi" if args.action == "pi-check" else "provider"},
    }
    try:
        if args.action == "pi-config":
            client.pi_config(args.directory)
            result = {"configuration": str(args.directory / "models.json"), "created": True}
        elif args.action == "verify":
            result = client.verify()
        elif args.action == "configure":
            result = client.configure()
        elif args.action == "pi-check":
            client.verify()
            result = client.pi_check(args.pi)
        else:
            result = client.check(args.runs)
        evidence.update(result=result, exit_code=0)
    except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError, subprocess.SubprocessError) as error:
        evidence.update(result=client.records, exit_code=1, error=endpoint.redact(str(error)))
    evidence["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    rendered = endpoint.diagnostic_json(evidence, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return evidence["exit_code"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
        print(f"Model check failed: {error}", file=sys.stderr)
        sys.exit(1)
