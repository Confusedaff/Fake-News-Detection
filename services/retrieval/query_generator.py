"""
Query Generator (Section 4).

Responsibility: turn a claim (+ prior weak-retrieval feedback) into a search
query. Does NOT touch the index directly — it only ever hands a query string
back to the orchestrator, which passes it to the Retriever.

MVP behavior (Section 3, step 5): the first query is just the claim text
verbatim. On retry (weak retrieval, attempts < MAX_RETRIEVAL_ATTEMPTS), an
LLM-assisted reformulation is used — bounded by MAX_LLM_CALLS (Section 7).
This is one of exactly two narrow, bounded LLM sub-tasks in the whole system
(the other being optional final-answer verification) — the LLM never drives
control flow itself.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "that", "this",
}


class QueryGenerator(ABC):
    @abstractmethod
    def initial_query(self, claim: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def reformulate(self, claim: str, previous_query: str, attempt_number: int) -> str:
        raise NotImplementedError


class HeuristicQueryGenerator(QueryGenerator):
    """
    Dependency-free MVP default. Initial query is the claim verbatim
    (per Section 3, step 5). Reformulation strips stopwords and short tokens
    to sharpen toward entities/keywords — a reasonable non-LLM fallback and
    what MockLLMQueryGenerator degrades to if the LLM call fails
    (Section 15: "skip reformulation and use original query if reformulation
    LLM fails").
    """

    def initial_query(self, claim: str) -> str:
        return claim.strip()

    def reformulate(self, claim: str, previous_query: str, attempt_number: int) -> str:
        tokens = re.findall(r"[A-Za-z0-9']+", claim)
        keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
        if not keywords:
            return claim.strip()
        return " ".join(keywords)


class LLMQueryGenerator(QueryGenerator):
    """
    LLM-assisted reformulation (Section 3 step 5, Section 7). Counts against
    MAX_LLM_CALLS — callers (the orchestrator) are responsible for enforcing
    that bound; this class just executes one call per invocation.

    `llm_call_fn` is injected so this class has no hard dependency on any
    specific SDK — pass a callable `(prompt: str) -> str`.

    Falls back to the heuristic generator on any error, matching the Section
    15 failure-handling table for "LLM unavailable".
    """

    def __init__(self, llm_call_fn):
        self._llm_call_fn = llm_call_fn
        self._fallback = HeuristicQueryGenerator()

    def initial_query(self, claim: str) -> str:
        # Initial query is always the claim verbatim — no LLM call needed here.
        return self._fallback.initial_query(claim)

    def reformulate(self, claim: str, previous_query: str, attempt_number: int) -> str:
        prompt = (
            "Rewrite the following claim as a short, keyword-focused search "
            "query optimized for retrieving fact-checking evidence. "
            "Return ONLY the query text, nothing else.\n\n"
            f"Claim: {claim}\n"
            f"Previous (weak) query: {previous_query}\n"
            f"Attempt: {attempt_number}"
        )
        try:
            result = self._llm_call_fn(prompt)
            result = (result or "").strip()
            return result if result else self._fallback.reformulate(claim, previous_query, attempt_number)
        except Exception:
            # Never let a flaky LLM call break retrieval — degrade instead.
            return self._fallback.reformulate(claim, previous_query, attempt_number)
