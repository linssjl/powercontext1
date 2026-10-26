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

"""HTTP cross-encoder scoring for Memory reranking."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANKER_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score a complete Memory candidate pool through one HTTP endpoint.

    ``strict_schema`` defaults to ``True`` because the published challenge
    contract requires a response object containing exactly ``scores`` and
    score records containing exactly ``rank`` and ``score``. Setting it to
    ``False`` ignores additional fields for deployments whose endpoint adds
    harmless metadata.

    ``fallback_on_invalid`` defaults to ``False`` because invalid responses
    must surface as ``ValueError`` (and HTTP failures as ``httpx`` errors) in
    the contract. Setting it to ``True`` returns the original candidate order
    as a last-resort decision instead of raising.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANKER_POLICY_ID,
        strict_schema: bool = True,
        fallback_on_invalid: bool = False,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self._strict_schema = strict_schema
        self._fallback_on_invalid = fallback_on_invalid
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score each candidate once and select the global top ``limit`` ranks."""

        candidate_tuple = tuple(candidates)
        candidate_count = len(candidate_tuple)
        if candidate_count < 1:
            raise ValueError("rerank requires at least one candidate")
        if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= candidate_count):
            raise ValueError("rerank limit must be between one and the candidate count")

        scores: dict[int, float] = {}
        requests = 0
        offset = 0
        try:
            while offset < candidate_count:
                batch = candidate_tuple[offset : offset + self._batch_size]
                ranks = tuple(range(offset + 1, offset + 1 + len(batch)))
                body = {
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text}
                        for rank, candidate in zip(ranks, batch, strict=True)
                    ],
                }
                requests += 1
                try:
                    response = await self._client.post(self._endpoint, json=body)
                except httpx.HTTPError:
                    # Preserve the concrete httpx subclass (ConnectError,
                    # TimeoutException, HTTPStatusError, ...) expected by callers.
                    raise
                except Exception as error:
                    raise httpx.RequestError(
                        f"rerank request failed for {self._endpoint}: {error}",
                        request=httpx.Request("POST", self._endpoint),
                    ) from error

                if not (200 <= response.status_code < 300):
                    raise _http_status_error(response, self._endpoint)

                try:
                    parsed = _parse_batch_scores(
                        response,
                        frozenset(ranks),
                        strict=self._strict_schema,
                    )
                except ValueError:
                    raise
                except Exception as error:
                    raise ValueError("invalid rerank response") from error

                scores.update(parsed)
                offset += len(batch)
        except (httpx.HTTPError, ValueError):
            if not self._fallback_on_invalid:
                raise
            return MemoryRerankDecision(
                selected_ranks=tuple(range(1, min(limit, candidate_count) + 1)),
                usage=InferenceUsage(requests=requests),
                used_fallback=True,
            )

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _score in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _http_status_error(response: httpx.Response, endpoint: str) -> httpx.HTTPStatusError:
    detail = ""
    try:
        detail = response.text.strip()[:300]
    except Exception:
        detail = ""
    message = f"rerank endpoint {endpoint} returned HTTP {response.status_code}"
    if detail:
        message = f"{message}: {detail}"
    try:
        request = response.request
    except Exception:
        request = httpx.Request("POST", endpoint)
    return httpx.HTTPStatusError(message, request=request, response=response)


def _parse_batch_scores(
    response: httpx.Response,
    expected: frozenset[int],
    *,
    strict: bool,
) -> dict[int, float]:
    """Validate one response and return scores keyed by original global rank."""

    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("rerank response must be valid JSON") from error
    except Exception as error:
        raise ValueError("rerank response could not be decoded") from error

    if not isinstance(payload, dict) or "scores" not in payload:
        raise ValueError("rerank response must contain a scores field")
    if strict and set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly the scores field")

    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list) or len(raw_scores) != len(expected):
        raise ValueError("rerank response must cover the batch exactly once")

    parsed: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            raise ValueError("each rerank score must be an object")
        if "rank" not in item or "score" not in item:
            raise ValueError("each rerank score must contain rank and score")
        if strict and set(item) != {"rank", "score"}:
            raise ValueError("each rerank score must contain exactly rank and score")

        rank = item["rank"]
        score = item["score"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank not in expected:
            raise ValueError("rerank score rank is not part of this batch")
        if rank in parsed:
            raise ValueError("rerank score rank is duplicated")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("rerank score must be a finite number")
        try:
            numeric = float(score)
        except (OverflowError, ValueError) as error:
            raise ValueError("rerank score must be a finite number") from error
        if not math.isfinite(numeric):
            raise ValueError("rerank score must be a finite number")
        parsed[rank] = numeric

    if set(parsed) != expected:
        raise ValueError("rerank response must cover the batch exactly once")
    return parsed


__all__ = ["DEFAULT_HTTP_RERANKER_POLICY_ID", "HTTPMemoryReranker"]
