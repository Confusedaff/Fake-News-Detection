"""
Image Module (new, Section 4/6/7A).

Responsibility: run manipulation detection, reverse-image search, and
caption-image consistency scoring on an attached image; return an
ImageVerificationResult.

Does NOT touch text evidence, does not call the classifier or verifier, does
not produce a verdict — matches Fusion's job, not this module's.

Same abstraction discipline as Classifier/Retriever/Verifier (Section 6):
the orchestrator calls ImageAnalyzer.analyze(image, claim_text) and never
knows which forgery model, search API, or consistency model is behind it.

Three internal sub-components, each independently swappable and each
degrading gracefully on its own per Section 15's failure table:
- ManipulationDetector: ELA + a lightweight forgery-detection CNN.
- ReverseSearchClient: the ONE external network call in the whole pipeline.
  Timeout-bounded; failure never blocks the text path (Section 7A/14/15).
- ConsistencyScorer: CLIP-style claim-vs-image similarity.

Security (Section 14): input validation (size cap, MIME/type check, hardened
decode against decompression bombs) happens in `validate_image_bytes` and
must run BEFORE any model sees the bytes.
"""

from __future__ import annotations

import hashlib
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from packages.schemas import ImageVerificationResult, ReverseSearchMatch

# Section 14: strict cap before any decode is attempted.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation (Section 14/15).

    Callers (the orchestrator) must catch this and proceed with the
    text-only pipeline rather than rejecting the whole claim — per the
    Section 15 failure table entry for "Invalid/corrupt/oversized image".
    """


def validate_image_bytes(raw_bytes: bytes, mime_type: str) -> None:
    """
    Section 14 mitigation for "malicious/oversized image uploads": size cap
    and MIME/type validation BEFORE the image reaches any model, plus a
    hardened decode (Pillow with bomb-detection thresholds) rather than a
    raw unsanitized decode.
    """
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError(f"Unsupported mime_type: {mime_type}")
    if len(raw_bytes) == 0:
        raise ImageValidationError("Empty image payload")
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            f"Image exceeds max size of {MAX_IMAGE_BYTES} bytes"
        )

    try:
        from PIL import Image
        # Pillow's decompression-bomb guard — raises DecompressionBombError
        # if the decoded pixel count is absurd relative to file size.
        Image.MAX_IMAGE_PIXELS = 64_000_000  # ~64MP cap, tune per deployment
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()  # structural check, does not fully decode
    except ImportError:
        # Pillow not installed in this environment (e.g. minimal CI image) —
        # size/MIME checks above still applied; log-worthy but not fatal here.
        pass
    except Exception as e:
        raise ImageValidationError(f"Could not decode image: {e}") from e


def content_hash(raw_bytes: bytes) -> str:
    """Used for both object-storage keying (dedupe) and the image-analysis
    cache key (Section 5/12)."""
    return hashlib.sha256(raw_bytes).hexdigest()


# --------------------------------------------------------------------------
# Sub-component interfaces
# --------------------------------------------------------------------------

@dataclass
class ManipulationCheck:
    manipulation_detected: bool
    manipulation_confidence: float
    model_version: str


class ManipulationDetector(ABC):
    @abstractmethod
    def check(self, raw_bytes: bytes) -> ManipulationCheck:
        raise NotImplementedError


class ReverseSearchClient(ABC):
    """
    The one external network call in the pipeline (Section 7A/14). Must be
    timeout-bounded and its failure must degrade to "unavailable" rather
    than blocking or failing the claim (Section 15).
    """

    @abstractmethod
    def search(self, raw_bytes: bytes, timeout_s: float = 5.0) -> tuple[bool, list[ReverseSearchMatch]]:
        """Returns (available, matches). available=False means the call
        failed/timed out — matches will be empty and downstream must mark
        reverse search as skipped, never fabricate results."""
        raise NotImplementedError


class ConsistencyScorer(ABC):
    @abstractmethod
    def score(self, raw_bytes: bytes, claim_text: str) -> tuple[float, str]:
        """Returns (consistency_score, model_version)."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# ImageAnalyzer interface + implementations
# --------------------------------------------------------------------------

class ImageAnalyzer(ABC):
    @abstractmethod
    def analyze(self, raw_bytes: bytes, mime_type: str, claim_text: str) -> ImageVerificationResult:
        raise NotImplementedError


class MockImageAnalyzer(ImageAnalyzer):
    """
    Deterministic stand-in for local dev/tests and Day 1-3 integration
    (owned, per Section 23, by the Retrieval/NLI Engineer alongside the real
    image module — same "wrap a pretrained model behind a versioned
    interface" pattern).

    Still runs full input validation so callers exercise the real
    ImageValidationError path even against the mock.
    """

    def __init__(
        self,
        forensics_version: str = "mock-image-forensics-v0",
        consistency_version: str = "mock-clip-v0",
        simulate_reverse_search_down: bool = False,
    ):
        self.forensics_version = forensics_version
        self.consistency_version = consistency_version
        self.simulate_reverse_search_down = simulate_reverse_search_down

    def analyze(self, raw_bytes: bytes, mime_type: str, claim_text: str) -> ImageVerificationResult:
        validate_image_bytes(raw_bytes, mime_type)
        digest = hashlib.sha256(raw_bytes + claim_text.encode()).hexdigest()

        manipulation_confidence = (int(digest[:4], 16) % 3000) / 10000.0  # low by default: [0, 0.3)
        manipulation_detected = manipulation_confidence > 0.2

        if self.simulate_reverse_search_down:
            reverse_available = False
            matches: list[ReverseSearchMatch] = []
            earliest = None
        else:
            reverse_available = True
            n_matches = int(digest[4:5], 16) % 3
            matches = []
            for i in range(n_matches):
                days_ago = 100 + (int(digest[5 + i:6 + i], 16) * 10)
                matches.append(
                    ReverseSearchMatch(
                        url=f"example-news-{i}.com/story",
                        first_seen_date=date.today() - timedelta(days=days_ago),
                        context_similarity=round(0.6 + (int(digest[6 + i:8 + i], 16) % 40) / 100.0, 2),
                    )
                )
            earliest = min((m.first_seen_date for m in matches), default=None)

        consistency = (int(digest[8:10], 16) % 10000) / 10000.0

        return ImageVerificationResult(
            manipulation_detected=manipulation_detected,
            manipulation_confidence=round(manipulation_confidence, 4),
            reverse_search_matches=matches,
            reverse_search_available=reverse_available,
            earliest_known_date=earliest,
            caption_image_consistency=round(consistency, 4),
            forensics_model_version=self.forensics_version,
            consistency_model_version=self.consistency_version,
        )


class PretrainedImageAnalyzer(ImageAnalyzer):
    """
    Real MVP composition (Section 6): ELA + lightweight forgery CNN for
    manipulation detection, a third-party reverse-search API call, and a
    CLIP-style model for caption-image consistency.

    This class only orchestrates the three sub-components and enforces the
    Section 7A/15 degradation rule: reverse-search failure never blocks the
    other two signals or the text path.
    """

    def __init__(
        self,
        manipulation_detector: ManipulationDetector,
        reverse_search_client: ReverseSearchClient,
        consistency_scorer: ConsistencyScorer,
        reverse_search_timeout_s: float = 5.0,
    ):
        self.manipulation_detector = manipulation_detector
        self.reverse_search_client = reverse_search_client
        self.consistency_scorer = consistency_scorer
        self.reverse_search_timeout_s = reverse_search_timeout_s

    def analyze(self, raw_bytes: bytes, mime_type: str, claim_text: str) -> ImageVerificationResult:
        validate_image_bytes(raw_bytes, mime_type)

        manip = self.manipulation_detector.check(raw_bytes)

        try:
            reverse_available, matches = self.reverse_search_client.search(
                raw_bytes, timeout_s=self.reverse_search_timeout_s
            )
        except Exception:
            # Section 15: reverse-search failure degrades gracefully, never
            # blocks the text path or the rest of the image analysis.
            reverse_available, matches = False, []

        earliest = min(
            (m.first_seen_date for m in matches if m.first_seen_date is not None),
            default=None,
        )

        consistency_score, consistency_version = self.consistency_scorer.score(raw_bytes, claim_text)

        return ImageVerificationResult(
            manipulation_detected=manip.manipulation_detected,
            manipulation_confidence=manip.manipulation_confidence,
            reverse_search_matches=matches,
            reverse_search_available=reverse_available,
            earliest_known_date=earliest,
            caption_image_consistency=consistency_score,
            forensics_model_version=manip.model_version,
            consistency_model_version=consistency_version,
        )


# --------------------------------------------------------------------------
# Reference sub-component implementations (real MVP building blocks)
# --------------------------------------------------------------------------

class ELAForgeryDetector(ManipulationDetector):
    """
    Error-Level Analysis + a lightweight pretrained forgery CNN
    (e.g. MantraNet/CASIA-trained checkpoint), per Section 6. Runs fully
    locally — no external calls.

    Lazy-imports Pillow/numpy/torch.
    """

    def __init__(self, model_path: Optional[str] = None, version: str = "image-forensics-v0.1"):
        self.model_path = model_path
        self._version = version
        self._model = None  # loaded lazily on first check()

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ELAForgeryDetector requires `torch` for the forgery CNN. "
                "Install it or use MockImageAnalyzer for development."
            ) from e
        if self.model_path:
            import torch
            self._model = torch.load(self.model_path, map_location="cpu")
            self._model.eval()

    def check(self, raw_bytes: bytes) -> ManipulationCheck:
        try:
            from PIL import Image, ImageChops
        except ImportError as e:
            raise ImportError("ELAForgeryDetector requires Pillow.") from e

        self._load_model()

        # --- Error Level Analysis: re-compress at a known quality and diff. ---
        original = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=90)
        buf.seek(0)
        recompressed = Image.open(buf)
        diff = ImageChops.difference(original, recompressed)
        extrema = diff.getextrema()
        max_diff = max(channel[1] for channel in extrema)
        ela_signal = min(1.0, max_diff / 255.0)

        if self._model is not None:
            # Real forgery CNN scores the ELA map; placeholder wiring left
            # for the ML team to fill in with their actual checkpoint's
            # preprocessing/inference call.
            confidence = ela_signal  # replace with self._model(...) output
        else:
            confidence = ela_signal

        return ManipulationCheck(
            manipulation_detected=confidence > 0.35,
            manipulation_confidence=round(confidence, 4),
            model_version=self._version,
        )


class GoogleVisionReverseSearch(ReverseSearchClient):
    """
    Real MVP reverse-search client (Section 6): calls a third-party API
    (e.g. Google Vision API web-detection). This is deliberately the only
    module in the whole system that makes an outbound network call to a
    third party, and Section 14 requires it be called ONLY with the
    submitted image bytes — never a user-supplied URL — over a fixed
    allowlisted endpoint with a bounded timeout.

    `http_post_fn` is injected: callable `(url, files, timeout) -> dict`,
    so this class carries no hard dependency on `requests` vs `httpx`.
    """

    ALLOWLISTED_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

    def __init__(self, api_key: str, http_post_fn):
        self.api_key = api_key
        self._http_post_fn = http_post_fn

    def search(self, raw_bytes: bytes, timeout_s: float = 5.0) -> tuple[bool, list[ReverseSearchMatch]]:
        try:
            response = self._http_post_fn(
                f"{self.ALLOWLISTED_ENDPOINT}?key={self.api_key}",
                files={"image": raw_bytes},
                timeout=timeout_s,
            )
        except Exception:
            return False, []

        try:
            matches = []
            for page in response.get("webDetection", {}).get("pagesWithMatchingImages", []):
                matches.append(
                    ReverseSearchMatch(
                        url=page.get("url", ""),
                        first_seen_date=None,  # API does not directly provide this; needs enrichment
                        context_similarity=0.75,  # placeholder heuristic until enrichment pass is added
                    )
                )
            return True, matches
        except Exception:
            return False, []


class ClipConsistencyScorer(ConsistencyScorer):
    """
    Real MVP consistency scorer (Section 6): openai/clip-vit-base-patch32,
    scoring how well claim text matches image content. Fully local, cheap.

    Lazy-imports torch/transformers/Pillow.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", version: Optional[str] = None):
        self.model_name = model_name
        self._version = version or f"clip-{model_name.split('/')[-1]}"
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise ImportError(
                "ClipConsistencyScorer requires `torch` and `transformers`. "
                "Install them or use MockImageAnalyzer for development."
            ) from e
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.eval()

    def score(self, raw_bytes: bytes, claim_text: str) -> tuple[float, str]:
        import torch
        from PIL import Image

        self._load()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        inputs = self._processor(text=[claim_text], images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_image is CLIP's cosine-similarity-derived score,
            # scaled by a learned temperature; normalize to [0,1] via sigmoid
            # since raw logits aren't bounded probabilities.
            score = torch.sigmoid(outputs.logits_per_image[0][0]).item()

        return round(score, 4), self._version
