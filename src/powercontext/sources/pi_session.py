# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture one selected Pi session branch as immutable, cited event Sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    Source,
    SourceDefinition,
    TextEvidence,
)
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.models import SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"

_CHECKPOINT_VERSION = 1
_IDENTITY_PREFIX = "pi"


class PiSessionInput(BaseModel):
    """One complete offline Pi v3 JSONL session and an optional selected leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """The exact values needed to materialize one visible Pi event."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event identified by its session and event identifiers."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build a stable, delimiter-safe identity for one Pi event."""

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")

    readable = f"{_IDENTITY_PREFIX}:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(readable) <= MAX_SOURCE_ID_LENGTH:
        return readable
    digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode("utf-8")).hexdigest()
    return f"{_IDENTITY_PREFIX}:{digest}"


def _decode_records(content: bytes) -> list[Any]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be valid UTF-8") from error

    records: list[Any] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
    return records


def _load_session(content: bytes) -> tuple[str, dict[str, dict[str, JsonValue]]]:
    parsed = _decode_records(content)
    if not parsed:
        raise ValueError("Pi session must contain a session header")

    header = parsed[0]
    if not isinstance(header, dict):
        raise ValueError("Pi session header must be a JSON object")
    if header.get("type") != "session":
        raise ValueError("first Pi record must have type='session'")
    if header.get("version") != 3:
        raise ValueError("Pi session version must be 3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must have a non-empty string id")

    by_id: dict[str, dict[str, JsonValue]] = {}
    for entry in parsed[1:]:
        if not isinstance(entry, dict):
            raise ValueError("every Pi session record must be a JSON object")
        event_id = entry.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every Pi session record must have a non-empty string id")
        if event_id in by_id:
            raise ValueError("Pi session ids must be unique")
        if "parentId" not in entry:
            raise ValueError("every Pi session record must declare parentId")
        parent_id = entry["parentId"]
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError("Pi session parentId must be null or a non-empty string")
            if parent_id not in by_id:
                raise ValueError("Pi session parentId must reference an earlier record")
        by_id[event_id] = cast(dict[str, JsonValue], entry)

    return session_id, by_id


def _text_body(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) if parts else None
    return None


def _tool_call_body(block: dict[str, Any]) -> str:
    name = block.get("name")
    if not isinstance(name, str) or not name:
        name = "unknown"
    call_id = block.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = "unknown"
    arguments = block.get("arguments", {})
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {name} {call_id}] {encoded}"


def _message_body(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif block_type == "toolCall":
                parts.append(_tool_call_body(block))
        return "\n".join(parts) if parts else None
    return None


def _tool_result_body(message: dict[str, Any]) -> str:
    name = message.get("toolName")
    if not isinstance(name, str) or not name:
        name = "unknown"
    call_id = message.get("toolCallId")
    if not isinstance(call_id, str) or not call_id:
        call_id = "unknown"
    text = _text_body(message.get("content"))
    return f"[tool_result {name} {call_id}] {text or ''}"


def _bash_body(message: dict[str, Any]) -> str:
    command = message.get("command")
    if not isinstance(command, str):
        command = ""
    output = message.get("output")
    if not isinstance(output, str):
        output = ""
    return f"$ {command}\n{output}"


def _visible_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None

    role: str | None = None
    body: str | None = None

    if event_type == "message":
        message = record.get("message")
        if isinstance(message, dict):
            message_role = message.get("role")
            if isinstance(message_role, str) and message_role:
                role = message_role
                if message_role == "toolResult":
                    body = _tool_result_body(message)
                elif message_role == "bashExecution":
                    body = _bash_body(message)
                else:
                    body = _message_body(message.get("content"))
    elif event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if isinstance(summary, str):
            role = event_type
            body = summary
    elif event_type == "custom_message":
        body = _text_body(record.get("content"))
        role = "custom"

    if role is None or body is None or not body:
        return None

    return PiEventInput(
        session_id=session_id,
        event_id=cast(str, record["id"]),
        parent_id=cast(str | None, record.get("parentId")),
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _selected_chain(
    by_id: dict[str, dict[str, JsonValue]],
    leaf_id: str | None,
) -> tuple[dict[str, JsonValue], ...]:
    if leaf_id is None:
        if not by_id:
            raise ValueError("Pi session has no selectable leaf")
        selected = next(reversed(by_id))
    else:
        if leaf_id not in by_id:
            raise ValueError("selected Pi leaf does not exist")
        selected = leaf_id

    chain: list[dict[str, JsonValue]] = []
    current: str | None = selected
    while current is not None:
        record = by_id[current]
        chain.append(record)
        current = cast(str | None, record.get("parentId"))
    chain.reverse()
    return tuple(chain)


def _select_events(
    session_id: str,
    by_id: dict[str, dict[str, JsonValue]],
    leaf_id: str | None,
) -> tuple[PiEventInput, ...]:
    events: list[PiEventInput] = []
    for record in _selected_chain(by_id, leaf_id):
        event = _visible_event(session_id, record)
        if event is not None:
            events.append(event)
    return tuple(events)


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse and validate one Pi session before returning any visible event."""

    session_id, by_id = _load_session(value.content)
    return _select_events(session_id, by_id, value.leaf_id)


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != _CHECKPOINT_VERSION:
        raise ValueError("invalid Pi connector checkpoint")
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        raise ValueError("invalid Pi connector checkpoint")
    loaded: dict[str, str] = {}
    for identity, digest in events.items():
        if not isinstance(identity, str) or not isinstance(digest, str):
            raise ValueError("invalid Pi connector checkpoint entry")
        loaded[identity] = digest
    return loaded


class _PiEventAdapter:
    """Resolve a parsed event into a captured, identifiable Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    """Project one event into the canonical text-evidence representation."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    _PiEventAdapter(),
    version=PI_EVENT_SOURCE_VERSION,
    projections=(_PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Incrementally import visible events from one persisted Pi session file."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self.path.read_bytes()
        session_id, by_id = _load_session(content)
        events = _select_events(session_id, by_id, self.leaf_id)
        imported = _load_checkpoint(session.checkpoint)

        for event_id, record in by_id.items():
            identity = pi_event_source_id(session_id, event_id)
            previous = imported.get(identity)
            if previous is not None and previous != _record_digest(record):
                raise ValueError(f"imported Pi event {event_id!r} was rewritten")

        checkpoint = dict(imported)
        for event in events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            if identity in imported:
                continue
            outcome = await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                checkpoint[identity] = _record_digest(event.record)

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": _CHECKPOINT_VERSION, "events": checkpoint},
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
