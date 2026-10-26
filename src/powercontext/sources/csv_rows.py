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

"""Row-level CSV capture exposed through the standard text-evidence projection."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    Source,
    SourceDefinition,
    TextEvidence,
)
from powercontext.sources.models import SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"

_IDENTITY_PREFIX = "csv"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document to split into row Sources."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """The exact values needed to materialize one CSV row."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured CSV row identified by its document and primary-key value."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def _require_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without leading or trailing whitespace")


def _require_primary_key_value(primary_key_value: str) -> None:
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-empty, non-blank string")


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable identity shared by a row Source and its projection."""

    _require_document_id(document_id)
    _require_primary_key_value(primary_key_value)
    source_id = (
        f"{_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    )
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate one CSV document before returning any row."""

    _require_document_id(value.document_id)
    if not isinstance(value.primary_key, str) or not value.primary_key:
        raise ValueError("primary_key must be a non-empty string")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"invalid CSV document: {error}") from error

    if not records:
        raise ValueError("CSV document must contain a header")

    header = records[0]
    if not header:
        raise ValueError("CSV header must not be empty")
    if any(not column.strip() for column in header):
        raise ValueError("CSV header fields must be non-blank")
    if len(set(header)) != len(header):
        raise ValueError("CSV header fields must be unique")
    if value.primary_key not in header:
        raise ValueError("primary_key must match a CSV header column exactly")

    key_index = header.index(value.primary_key)
    columns = tuple(header)
    seen: set[str] = set()
    rows: list[CSVRowInput] = []

    for record in records[1:]:
        if not record:
            # A completely empty physical line carries no record.
            continue
        if len(record) != len(columns):
            raise ValueError("CSV record width must match the header width")

        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("primary_key_value must not be empty or blank")
        if primary_key_value in seen:
            raise ValueError("primary_key_value must be unique within a document")
        seen.add(primary_key_value)

        csv_row_source_id(value.document_id, primary_key_value)

        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )

    return tuple(rows)


class _CSVRowAdapter:
    """Resolve a validated row input into a captured row Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextEvidenceProjection:
    """Project one row into the canonical text-evidence representation."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    _CSVRowAdapter(),
    version=CSV_ROW_SOURCE_VERSION,
    projections=(_CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
