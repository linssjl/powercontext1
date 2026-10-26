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

"""Capture every physical PDF page and retrieve evidence pages by question."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.artifacts.models import MemoryCitation
from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    Source,
    SourceDefinition,
    TextEvidence,
)
from powercontext.sources.models import SourceMaterialization

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"

_TOKEN = re.compile(r"[a-z]+|\d+")

_QUESTION_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "according",
        "all",
        "also",
        "an",
        "and",
        "answer",
        "any",
        "are",
        "as",
        "assume",
        "at",
        "available",
        "be",
        "been",
        "being",
        "between",
        "billion",
        "by",
        "calculate",
        "calculated",
        "calculation",
        "can",
        "companies",
        "company",
        "could",
        "define",
        "defined",
        "detail",
        "details",
        "did",
        "do",
        "does",
        "during",
        "each",
        "fiscal",
        "for",
        "from",
        "fy",
        "give",
        "given",
        "had",
        "has",
        "have",
        "if",
        "in",
        "information",
        "into",
        "is",
        "it",
        "its",
        "less",
        "may",
        "me",
        "might",
        "million",
        "millions",
        "more",
        "most",
        "must",
        "no",
        "not",
        "number",
        "numbers",
        "of",
        "on",
        "only",
        "or",
        "other",
        "our",
        "outlined",
        "over",
        "percentage",
        "perspective",
        "please",
        "provide",
        "provided",
        "question",
        "regarding",
        "related",
        "report",
        "reported",
        "reports",
        "respect",
        "respond",
        "response",
        "shall",
        "show",
        "shown",
        "state",
        "stated",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "these",
        "they",
        "this",
        "those",
        "to",
        "total",
        "totals",
        "under",
        "usd",
        "use",
        "used",
        "using",
        "value",
        "values",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "within",
        "would",
        "year",
        "years",
        "you",
    }
)

_FINANCE_PHRASES = (
    "legal proceedings",
    "legal matters",
    "legal battles",
    "cash flow",
    "cash flows",
    "operating income",
    "operating profit",
    "depreciation amortization",
    "capital expenditures",
    "capital expenditure",
    "restructuring costs",
    "restructuring charges",
    "income statement",
    "income statements",
    "balance sheet",
    "geographic regions",
    "geographies",
    "ebitda",
    "gross margin",
    "net sales",
    "net revenue",
)

_SYNONYMS: dict[str, tuple[tuple[str, float], ...]] = {
    "capex": (("capital expenditures", 0.4),),
    "legal": (("litigation", 0.3), ("proceedings", 0.3)),
    "battles": (("proceedings", 0.2), ("litigation", 0.2)),
    "geographies": (("regions", 0.3), ("markets", 0.3)),
    "geography": (("regions", 0.3), ("markets", 0.3)),
    "restructuring": (("severance", 0.25),),
}

_FINANCIAL_TERMS = frozenset(
    {
        "amortization",
        "capex",
        "cash",
        "cost",
        "costs",
        "depreciation",
        "ebitda",
        "expenditure",
        "expenditures",
        "expense",
        "expenses",
        "flow",
        "income",
        "margin",
        "profit",
        "revenue",
        "revenues",
        "sale",
        "sales",
        "spending",
    }
)


class PDFDocumentInput(BaseModel):
    """One complete original PDF and its stable caller-supplied document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """The exact values needed to materialize one physical PDF page."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A captured physical page identified by document, digest and page number."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build a stable, bounded identity for one physical PDF page."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(document_sha256, str) or not document_sha256:
        raise ValueError("document_sha256 must be a non-empty string")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive integer")

    readable = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(readable) <= MAX_SOURCE_ID_LENGTH:
        return readable
    digest = hashlib.sha256(
        f"{document_id}\x00{document_sha256}\x00{page_number}".encode("utf-8")
    ).hexdigest()
    return f"pdf:{digest}:{page_number}"


def _read_pdf_texts(content: bytes, *, strict: bool) -> tuple[str, ...]:
    if not content:
        raise ValueError("PDF content must not be empty")

    try:
        reader = PdfReader(io.BytesIO(content), strict=strict)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a user password")
        pages = reader.pages
        if len(pages) < 1:
            raise ValueError("PDF must contain at least one page")
        texts: list[str] = []
        for page in pages:
            text = page.extract_text()
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise ValueError("PDF page text must be text")
            texts.append(text.strip())
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("invalid or unreadable PDF") from error
    return tuple(texts)


def _structurally_repairable(content: bytes) -> bool:
    head = content[:1024]
    tail = content[-2048:]
    return b"%PDF-" in head and b"%%EOF" in tail and b"startxref" in tail


def _parse_pdf_document_with_repair(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Batch-only strict-then-guarded-lenient entry; judging uses strict parsing."""

    if not isinstance(value.document_id, str) or not value.document_id.strip():
        raise ValueError("document_id must not be empty or whitespace")

    try:
        return parse_pdf_document(value)
    except ValueError:
        if not _structurally_repairable(value.content):
            raise
        texts = _read_pdf_texts(value.content, strict=False)
        document_sha256 = hashlib.sha256(value.content).hexdigest()
        return tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                text=text,
            )
            for page_number, text in enumerate(texts, start=1)
        )


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse and validate every physical page before returning any page."""

    if not isinstance(value.document_id, str) or not value.document_id.strip():
        raise ValueError("document_id must not be empty or whitespace")

    document_sha256 = hashlib.sha256(value.content).hexdigest()
    texts = _read_pdf_texts(value.content, strict=True)
    return tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )
        for page_number, text in enumerate(texts, start=1)
    )


class _PDFPageAdapter:
    """Resolve a parsed page into a captured, identifiable Source."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class _PDFPageTextEvidenceProjection:
    """Project one page into the canonical text-evidence representation."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = (
    AdapterSourceDefinition(
        _PDFPageAdapter(),
        version=PDF_PAGE_SOURCE_VERSION,
        projections=(_PDFPageTextEvidenceProjection(),),
    )
)


@dataclass
class _PageCandidate:
    citation: MemoryCitation
    source_id: str
    document_id: str
    page_number: int
    text: str
    score: float


def _core_terms(question: str) -> tuple[str, ...]:
    terms: list[str] = []
    for token in _TOKEN.findall(question.lower()):
        if token in _QUESTION_STOPWORDS:
            continue
        if token.isalpha() and len(token) == 1:
            continue
        if token not in terms:
            terms.append(token)
    return tuple(terms)


def _question_queries(question: str) -> tuple[tuple[str, float], ...]:
    lower = question.lower()
    core = _core_terms(question)
    queries: list[tuple[str, float]] = [(question, 0.5)]
    if core:
        queries.append((" ".join(core[:10]), 1.0))

    for phrase in _FINANCE_PHRASES:
        words = phrase.split()
        if phrase in lower or (words and all(word in core for word in words)):
            queries.append((phrase, 1.3))

    if "d&a" in lower:
        queries.append(("depreciation amortization", 0.6))
    if "p&l" in lower:
        queries.append(("income statement", 0.3))

    specific = [term for term in core if not term.isdigit() and len(term) >= 4]
    for index in range(0, min(len(specific), 8), 2):
        group = specific[index : index + 2]
        if len(group) == 2:
            queries.append((" ".join(group), 0.8))

    if set(core) & _FINANCIAL_TERMS:
        queries.extend(
            (
                ("operating profit", 0.9),
                ("gross profit", 0.8),
                ("net income", 0.8),
                ("cost of sales", 0.7),
            )
        )
        if "cash flow" in lower:
            queries.append(("consolidated statement of cash flows", 0.9))

    years = [term for term in core if len(term) == 4 and term.isdigit()]
    if years:
        queries.append((" ".join(years[:3]), 0.3))

    synonym_count = 0
    for term in core:
        if synonym_count >= 3:
            break
        for synonym, weight in _SYNONYMS.get(term, ()):
            if synonym_count >= 3:
                break
            queries.append((synonym, weight))
            synonym_count += 1

    best: dict[str, float] = {}
    for query, weight in queries:
        normalized = " ".join(query.split())
        if normalized:
            best[normalized] = max(best.get(normalized, 0.0), weight)
    return tuple(sorted(best.items(), key=lambda item: (-item[1], item[0]))[:16])


def _page_number_from_source_id(source_id: str) -> int | None:
    _, separator, tail = source_id.rpartition(":")
    if separator and tail.isdigit():
        page_number = int(tail)
        return page_number if page_number >= 1 else None
    return None


async def search_pdf_pages(
    memory: Any,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search a scoped Memory head and return distinct, exact PDF page citations."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return ()
    if not isinstance(question, str) or not question.strip():
        return ()

    core = set(_core_terms(question))
    question_phrases = tuple(phrase for phrase in _FINANCE_PHRASES if phrase in question.lower())
    search_limit = max(30, min(limit * 3, 100))

    hit_scores: dict[tuple[str, int, str, str], float] = {}
    hit_meta: dict[tuple[str, int, str, str], tuple[MemoryCitation, Any]] = {}

    for query, weight in _question_queries(question):
        try:
            page = await memory.search(
                SearchMemoryRequest(query=query, limit=search_limit, mode="fts")
            )
        except ValueError:
            continue
        for rank, hit in enumerate(page.hits, start=1):
            if hit.memory_ref is None:
                continue
            key = (
                hit.memory_ref.artifact_id,
                hit.memory_ref.revision,
                hit.entry_id,
                hit.entry_version_id,
            )
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            hit_scores[key] = hit_scores.get(key, 0.0) + weight / (60.0 + rank)
            hit_meta[key] = (citation, hit)

    candidates: dict[str, _PageCandidate] = {}
    for key, score in hit_scores.items():
        citation, _hit = hit_meta[key]
        try:
            entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        except Exception:
            continue
        pdf_sources = [source for source in entry.sources if source.source_type == PDF_PAGE_SOURCE_NAME]
        if len(pdf_sources) != 1:
            continue
        source_id = pdf_sources[0].source_id
        page_number = _page_number_from_source_id(source_id)
        if page_number is None:
            continue
        text = entry.text
        if "\n" in text:
            document_id, page_text = text.split("\n", 1)
        else:
            document_id, page_text = "", text
        if not page_text.strip():
            continue
        existing = candidates.get(source_id)
        if existing is None:
            candidates[source_id] = _PageCandidate(
                citation=citation,
                source_id=source_id,
                document_id=document_id,
                page_number=page_number,
                text=text,
                score=score,
            )
        else:
            existing.score += score

    if not candidates:
        return ()

    financial = bool(core & _FINANCIAL_TERMS)
    for candidate in candidates.values():
        text_lower = candidate.text.lower()
        normalized = " ".join(text_lower.split())
        tokens = _TOKEN.findall(text_lower)
        terms = set(tokens)
        coverage = len(core & terms) / len(core) if core else 0.0
        phrase_hits = min(3, sum(1 for phrase in question_phrases if phrase in normalized))
        document_match = (
            1.0
            if candidate.document_id
            and any(term in candidate.document_id.lower() for term in core if len(term) >= 4)
            else 0.0
        )
        numeric_ratio = sum(token.isdigit() for token in tokens) / len(tokens) if tokens else 0.0
        table_bonus = min(candidate.text.count("$"), 20) if financial and "gross profit" in normalized else 0
        candidate.score = (
            candidate.score * 20.0
            + 1.5 * coverage
            + 0.8 * phrase_hits
            + 0.5 * document_match
            + (0.8 * numeric_ratio if financial else 0.0)
            + (0.05 * table_bonus if financial else 0.0)
            + (1.0 if financial and "gross profit" in normalized else 0.0)
        )

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.score, candidate.document_id, candidate.page_number),
    )
    top = ordered[:5]
    for candidate in ordered:
        for anchor in top:
            if candidate is anchor:
                continue
            if (
                candidate.document_id
                and candidate.document_id == anchor.document_id
                and abs(candidate.page_number - anchor.page_number) == 1
            ):
                candidate.score += 0.3
                break
    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.score, candidate.document_id, candidate.page_number),
    )
    return tuple(candidate.citation for candidate in ordered[:limit])


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
