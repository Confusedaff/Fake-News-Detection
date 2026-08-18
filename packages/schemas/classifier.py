"""
packages/schemas/classifier.py

Shared Pydantic schemas for the Team 4 ML Classifier service.
Acts as the single source of truth for classifier request and response contracts across teams.
"""

from __future__ import annotations

from typing import Dict
from pydantic import BaseModel, Field


class ClassifierRequest(BaseModel):
    """Input payload to the sequence classifier.

    Primary input is solely the claim text.
    No live retrieval or evidence documents required at this layer.
    """
    text: str = Field(
        ...,
        description="The raw claim text to be classified for truthfulness.",
        examples=["The government announced a new healthcare policy yesterday."],
    )


class Prediction(BaseModel):
    """The argmax prediction outcome."""
    label: str = Field(
        ...,
        description="The predicted 6-way LIAR label (pants-fire, false, barely-true, half-true, mostly-true, true).",
    )
    label_id: int = Field(
        ...,
        description="Integer class index (0..5).",
    )
    confidence: float = Field(
        ...,
        description="Calibrated probability corresponding to the predicted class label.",
    )


class ClassifierResponse(BaseModel):
    """Output contract produced by Team 4 ML Classifier."""
    text: str = Field(
        ...,
        description="The claim text evaluated.",
    )
    prediction: Prediction = Field(
        ...,
        description="Top-1 prediction summary.",
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Calibrated probability distribution across all 6 LIAR classes.",
    )
    raw_logit_confidence: float = Field(
        ...,
        description="Uncalibrated maximum softmax confidence from the raw model logits.",
    )
    calibrated_probability: float = Field(
        ...,
        description="Calibrated maximum softmax probability after temperature scaling.",
    )
    model_version: str = Field(
        ...,
        description="Immutable version identifier of the deployed model artifact.",
        examples=["classifier-v3.0"],
    )
