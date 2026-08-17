# Fake News Verifier — `services/` Layer: Status & Handoff

**Scope of this doc:** everything under `packages/` and `services/` in the
repo — the internal module layer described in Sections 4, 6, 7, 7A, 11, and
22 of the architecture doc. It does not cover `apps/`, `ml/`, `data/`,
`infrastructure/`, which are owned by other parts of the team.

---

## 1. What's done

### 1.1 Contracts (`packages/schemas/contracts.py`)
Every inter-module data shape from the architecture doc is implemented as a
frozen Pydantic model: `ClassificationResult`, `RetrievalResult` /
`RetrievalChunk`, `VerificationResult` / `ChunkVerification`,
`ImageVerificationResult` / `ReverseSearchMatch`, `FusedResult`,
`DecisionResult` / `EvidenceItem`, `OrchestrationEvent`, plus the top-level
`ClaimRequest` / `ClaimResult` matching the `POST /v1/claims/analyze` API
shape from Section 8. This is the frozen contract layer — Section 22 calls
it "load-bearing, not decorative," and every module below imports from here
instead of defining its own shapes.

### 1.2 Config (`packages/config/__init__.py`)
Centralized, named, tunable constants for everything the doc specifies as a
bound or threshold:
- `OrchestratorBounds`: `MAX_RETRIEVAL_ATTEMPTS=3`, `MAX_EVIDENCE_CHUNKS=10`,
  `MAX_LLM_CALLS=2`, soft/hard latency budgets (Section 7).
- `RouterThresholds`: classifier confidence threshold (0.85), retrieval
  relevance floor (0.6) (Section 3/10).
- `AggregationWeights`: classifier voting weight, same-publisher decay,
  verdict-taxonomy margin cut points (Section 11).

### 1.3 Classifier (`services/classifier/`)
- `Classifier` — abstract interface, `predict(claim) -> ClassificationResult`.
- `MockClassifier` — deterministic (hash-based), zero heavy deps. Applies a
  toy temperature-scaling calibration so calibrated ≠ raw even in the mock,
  matching Section 6's "calibration is not optional" rule.
- `DistilBertClassifier` — real MVP shape (fine-tuned DistilBERT/RoBERTa +
  temperature scaling). Lazy-imports `torch`/`transformers`. **Not trained
  — this is wiring, not a trained model.**

### 1.4 Retrieval (`services/retrieval/`)
- `Retriever` interface, `MockRetriever` (tiny 3-doc fixture corpus,
  keyword-overlap scoring), `FaissRetriever` (real: sentence-transformers +
  FAISS `IndexFlatIP`, lazy-imported).
- `QueryGenerator`: `HeuristicQueryGenerator` (stopword-stripping, no LLM)
  and `LLMQueryGenerator` (injectable `llm_call_fn`, falls back to
  heuristic on any failure — Section 15's LLM-unavailable rule).

### 1.5 Verification (`services/verification/`)
- `Verifier` interface, `MockVerifier` (deterministic label/confidence per
  chunk), `LocalNLIVerifier` (real: `roberta-large-mnli`, lazy-imported —
  this is the **primary** verifier per Section 14's prompt-injection threat
  model, since it has no instruction-following capability to hijack).
- `LLMVerifier` — secondary/optional, implements Section 14's mitigation
  explicitly: evidence passed as a delimited data field, fixed system
  prompt never derived from document content, strict output parsing with a
  safe fallback (`UNRELATED`, confidence 0) on any parse failure.

### 1.6 Image (`services/image/`)
- `validate_image_bytes` — size cap (10MB), MIME allowlist, Pillow
  decompression-bomb guard, run **before** any model sees the bytes
  (Section 14).
- `ImageAnalyzer` interface, `MockImageAnalyzer` (deterministic,
  zero-heavy-dep, still runs real validation), `PretrainedImageAnalyzer`
  (real: composes three swappable sub-interfaces).
- Real sub-component stubs: `ELAForgeryDetector` (error-level analysis +
  hook for a forgery CNN), `GoogleVisionReverseSearch` (the one external
  network call in the system — allowlisted endpoint, image-bytes-only, no
  user-supplied URLs, per Section 14's SSRF mitigation),
  `ClipConsistencyScorer` (CLIP-based claim/image similarity).

### 1.7 Fusion (`services/fusion/`)
`Fusion.reconcile(text_result, image_result) -> FusedResult`. Never blends
text and image signals into one number (Section 7A's core rule) — keeps
`text_verification`, `image_verification`, `caption_image_consistency` as
distinct fields, and generates a template-based `note` (e.g. "image
predates the claimed event by ~1.1 years") from already-computed dates and
scores, not a model call.

### 1.8 Aggregation (`services/aggregation/`)
`Aggregator.aggregate(...) -> DecisionResult` implementing Section 11's full
algorithm:
- Per-chunk weight = `source_quality × retrieval_relevance × nli_confidence
  × recency_factor(publication_date)`.
- Support/contradict mass accumulation with an independent-source bonus
  (repeat publishers down-weighted).
- Image mass folded into the same accumulation as one more weighted vote
  (Section 7A), not a separately-averaged number.
- Classifier folded in as a weighted voter (higher weight only when
  evidence is weak/absent).
- Full verdict taxonomy including `INSUFFICIENT_EVIDENCE` (no evidence at
  all) and `CONFLICTING_EVIDENCE` (both masses high and balanced) as
  first-class outputs, never forced onto the True/False spectrum.
- `heuristic_score` and `calibrated_probability` kept as permanently
  separate fields — `calibrated_probability` is `None` at MVP by design
  (Section 11: needs a separately trained calibration model).

### 1.9 Orchestrator (`services/orchestrator/`)
The bounded state machine tying every module above together, matching
Section 3/7/7A step-by-step:
- Enforces all bounds from `packages/config` — retrieval retry loop, LLM
  call budget, soft/hard latency timeout.
- Runs the image branch on a background thread **in parallel** with text
  retrieval/verification (Section 7A: "pays the max, not the sum").
- Every state transition appends an `OrchestrationEvent` to an in-memory
  list — this list becomes `decision_trail` in the response, so the trail
  can never show a step that didn't actually run.
- Graceful degradation implemented for every failure mode in Section 15's
  table: classifier down, retrieval down, image invalid/unavailable,
  reverse-search down, LLM down — none of these crash the pipeline.

### 1.10 Wiring + tests
- `services/factory.py`: `build_mock_orchestrator()` (zero heavy deps) and
  `build_production_orchestrator()` (shows exactly which real classes to
  instantiate once real models/credentials exist).
- `demo.py`: runs the 4-scenario demo script from Section 25 (fast path,
  verification path, insufficient evidence, image + fabricated context).
- `tests/unit/`: 14 tests covering the orchestrator end-to-end and the
  aggregator's verdict logic specifically (support/contradict/conflicting,
  publisher down-weighting, heuristic-vs-calibrated separation).

**Verified working** (as of your last run): all 14 tests pass, `demo.py`
runs clean, inside `myenv` on your machine.

---

## 2. What's explicitly mocked / not real yet

Nothing above trains or calls a real ML model except through lazy-imported
classes that are wired but not exercised. Concretely, still fake/stubbed:

| Piece | Current state |
|---|---|
| Classifier | Hash-based mock. No LIAR training has happened. |
| Retrieval corpus | 3 hardcoded fixture documents. No real evidence corpus, no real embeddings, no real FAISS index built. |
| NLI verifier | Hash-based mock. `roberta-large-mnli` wiring exists but is untested against real evidence. |
| Image manipulation detection | Mock returns a hash-derived confidence. `ELAForgeryDetector`'s ELA computation is real, but there's no trained forgery CNN behind it yet — it's currently just the raw ELA signal. |
| Reverse image search | Mock. `GoogleVisionReverseSearch` needs a real API key and has a placeholder for `first_seen_date`/`context_similarity` enrichment that doesn't exist yet. |
| CLIP consistency scorer | Mock. `ClipConsistencyScorer` wiring is real but untested. |
| Postgres persistence | **Does not exist.** `OrchestrationEvent` rows live only in `Orchestrator.events`, in memory, per single `run()` call — lost the moment the process exits or the next claim starts. |
| Redis caching | **Does not exist.** No caching layer at all yet (Section 12). |
| API layer (`apps/api/`) | Not part of this handoff — whoever owns `apps/` needs to call into `services.factory` directly. |

---

## 3. What needs to change for production (in priority order)

### 3.1 Postgres persistence — *do this first*
Right now a claim's full decision trail disappears the instant `run()`
returns unless the caller manually holds onto the `ClaimResult` object.
Needed:
- Implement the tables from Section 9 (`claims`, `classification_results`,
  `retrieval_results`, `verification_results`, `orchestration_events`).
- Add a persistence hook in `Orchestrator._transition()` (or a wrapper
  around `run()`) that writes each `OrchestrationEvent` as it's created,
  not just at the end — this is what makes a crash mid-pipeline still
  leave a partial, inspectable trail (Section 3's "observable and
  resumable after a crash" claim, which is currently *not* true — a crash
  today loses the whole in-memory trail).
- Write the final `ClaimResult` to the `claims` table on `FINALIZED` /
  `ERROR` / `TIMED_OUT`.

### 3.2 Swap in real Classifier
- Sub-team A trains DistilBERT/RoBERTa on LIAR (Section 6).
- Fit a temperature-scaling (or isotonic) calibration map on a held-out
  split — **do not skip this**, it's the difference between the router
  threshold meaning something and being noise.
- Point `DistilBertClassifier(model_path=..., version=...)` at the trained
  checkpoint in `factory.py::build_production_orchestrator`.
- Track precision/recall/F1 per class + confusion matrix on the LIAR test
  split before considering it done (Section 6).

### 3.3 Build the real evidence corpus + retriever
- Data Engineer 2's pipeline: ingest → clean → dedup → chunk → embed with
  `all-MiniLM-L6-v2` → build a FAISS `IndexFlatIP`.
- Replace `MockRetriever`'s 3-doc fixture with `FaissRetriever(index_path=...,
  metadata=..., corpus_version=...)`.
- `metadata` must be index-aligned with the FAISS index — get this pipeline
  right once, since Section 5 makes `corpus_version` part of every cache
  key and every stored prediction.
- Watch the relevance floor (0.6) against real embedding scores once this
  is live — the mock's scores were hand-tuned to trip the floor easily and
  will not resemble real cosine-similarity score distributions.

### 3.4 Swap in real Verifier
- Wire `LocalNLIVerifier` against `roberta-large-mnli` and validate its
  label mapping (contradiction/neutral/entailment → CONTRADICTS/UNRELATED/
  SUPPORTS) against a few hand-checked examples — the order matters and is
  easy to get backwards silently.
- Only add `LLMVerifier` as a secondary signal once `LocalNLIVerifier` is
  solid; per Section 14 it should never be the *only* verifier given the
  prompt-injection risk.

### 3.5 Real image pipeline
- Get an actual trained forgery-detection checkpoint (MantraNet/CASIA-style)
  behind `ELAForgeryDetector._model` — right now it's ELA-only with a
  pass-through where the CNN score should be.
- Get a real reverse-search API key (Google Vision web-detection or TinEye)
  and wire `GoogleVisionReverseSearch`; build the `first_seen_date`
  enrichment step the placeholder currently skips (the raw API doesn't
  return this directly — needs a follow-up lookup per match).
- Confirm `ClipConsistencyScorer`'s sigmoid-normalized score distribution
  makes sense against real image/caption pairs — the 0.35 "low consistency"
  threshold in `Fusion` was a documented guess, not tuned against data.

### 3.6 Redis caching (Section 12)
Not started. Once real classifier/corpus versions exist, add:
- Claim result cache: key = `hash(normalized_claim + classifier_version +
  verifier_version + corpus_version)`.
- Retrieval cache: key = `hash(query_text + embedding_model_version +
  corpus_version + top_k + filters)`.
- Image analysis cache: key = `hash(image_bytes + forensics_model_version +
  consistency_model_version)`.
This is what makes "any model version bump auto-invalidates old cache
entries" true — the versioning has to actually be wired through from the
real model classes for this property to hold.

### 3.7 API layer (`apps/api/`)
Not part of this handoff, but the services layer is ready for it:
- Wrap `services.factory.build_mock_orchestrator()` (swap to
  `build_production_orchestrator()` later) behind `POST
  /v1/claims/analyze`.
- Implement request validation, auth, rate limiting, idempotency at the API
  layer — none of that exists in `services/` by design (Section 4: "API
  Layer... does not do business logic").
- Base64-decode the incoming `image.data` field into raw bytes before
  calling `Orchestrator.run(image_bytes=..., image_mime_type=...)`.

### 3.8 Calibrated probability (Should-Build, not MVP-blocking)
Section 11 explicitly scopes this out of MVP. Once there's a labeled eval
set: train a small logistic regression on `(heuristic_score,
classifier_probability, evidence_count, source_quality_mean) → outcome`
and populate `DecisionResult.calibrated_probability`, which is currently
always `None`.

### 3.9 Tuning pass (Day 5 in Section 24's plan)
Everything in `packages/config/__init__.py` was set from the doc's stated
defaults or reasonable placeholders, not tuned against real data:
- `AggregationWeights.STRONG_MARGIN` / `MODERATE_MARGIN` — verdict cut
  points, currently 0.5 / 0.15.
- `Fusion.LOW_CONSISTENCY_THRESHOLD` (0.35) and `STALE_IMAGE_DAYS` (180).
- `RouterThresholds.RETRIEVAL_RELEVANCE_FLOOR` (0.6) — will very likely
  need adjusting once real embedding scores replace the mock's inflated
  ones.
Run these against the golden eval set once one exists and adjust.

---

## 4. Suggested order of operations for your team

1. **Postgres persistence** (3.1) — do this before swapping in real models,
   so every subsequent test run leaves a debuggable trail instead of
   evaporating.
2. **Real corpus + retriever** (3.3) — everything downstream (verifier,
   aggregator tuning) needs real evidence to be meaningful.
3. **Real classifier** (3.2) and **real verifier** (3.4) — can happen in
   parallel once the corpus exists, by different people, since they're
   independent contracts.
4. **API layer** (3.7) — can start immediately in parallel with the above,
   since it only needs `build_mock_orchestrator()` until integration day.
5. **Real image pipeline** (3.5) — can also start in parallel; it's fully
   independent of the text-side work.
6. **Redis caching** (3.6) — add once real version strings exist to key on.
7. **Tuning pass** (3.9) — last, once there's a real eval set to tune
   against.
8. **Calibrated probability** (3.8) — optional, after everything else is
   stable.
