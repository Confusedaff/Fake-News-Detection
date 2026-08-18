"""
ml_classifier/training/dataset.py

Thin PyTorch Dataset over the processed CSVs produced by
ml_classifier/datasets/prepare_dataset.py. Tokenization only happens here —
all cleaning/validation/splitting already happened upstream.

Fixes applied:
  - keep_default_na=False / na_filter=False on CSV read: prevents pandas
    from turning blank metadata cells (context/job_title/state_info) back
    into NaN -> str(NaN) == "nan" -> the literal word "nan" was getting
    injected into the model's input text for every row with missing
    metadata. This reads blanks as "" instead, which _build_text already
    handles correctly (skips empty fields).
  - Tokenize the whole split ONCE in __init__ instead of on every
    __getitem__ call. The text never changes between epochs, so repeated
    re-tokenization was pure wasted CPU work every single epoch.
  - padding=False here + a DataCollatorWithPadding at the Trainer/DataLoader
    level (see train.py / calibrate.py) pads each batch only to its own
    longest sequence instead of a fixed max_length=128 for every batch —
    LIAR statements are mostly much shorter than 128 tokens, so this cuts
    a lot of wasted compute on padding tokens.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# LIAR speaker-history count columns (processed CSV) -> readable label tokens.
HISTORY_COLUMNS: list[tuple[str, str]] = [
    ("pants_on_fire_counts", "pants-fire"),
    ("false_counts", "false"),
    ("barely_true_counts", "barely-true"),
    ("half_true_counts", "half-true"),
    ("mostly_true_counts", "mostly-true"),
]


class LiarDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str = "statement",
        label_column: str = "label_id",
        max_length: int = 128,
        use_metadata: bool = True,
    ):
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.use_metadata = use_metadata

        # keep_default_na=False + na_filter=False: blank cells stay as "",
        # never silently become NaN (which would otherwise get stringified
        # into the literal word "nan" inside the model's input text below).
        df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)

        texts = [self._build_text(row) for row in df.to_dict(orient="records")]

        # Tokenize the entire split once, up front. padding=False here —
        # padding is applied per-batch instead, via DataCollatorWithPadding
        # wherever this dataset is consumed.
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = df[label_column].astype(int).tolist()

    def _format_speaker_history(self, row: dict) -> str:
        """Format LIAR credibility-history counts as structured text."""
        parts: list[str] = []
        for column, label in HISTORY_COLUMNS:
            raw = str(row.get(column, "")).strip()
            if not raw:
                continue
            try:
                count = int(float(raw))
            except ValueError:
                continue
            parts.append(f"{label}={count}")
        if not parts:
            return ""
        return "speaker history " + " ".join(parts)

    def _build_text(self, row: dict) -> str:
        statement = str(row.get(self.text_column, "")).strip()

        if not self.use_metadata:
            return statement or "unknown statement"

        speaker = str(row.get("speaker", "")).strip()
        party = str(row.get("party_affiliation", "")).strip()
        subject = str(row.get("subject", "")).strip()
        context = str(row.get("context", "")).strip()

        prefix_parts: list[str] = []

        history = self._format_speaker_history(row)
        if history:
            prefix_parts.append(history)

        metadata_parts = []
        if speaker:
            # LIAR speaker values are slugs like "dwayne-bohac" — turn hyphens
            # into spaces so the tokenizer sees it closer to natural text
            # rather than a single unfamiliar hyphenated token.
            metadata_parts.append(speaker.replace("-", " "))
        if party:
            metadata_parts.append(f"({party})")
        if subject:
            metadata_parts.append(f"on {subject.replace('-', ' ').replace(',', ', ')}")
        if context:
            metadata_parts.append(f"in {context}")

        if metadata_parts:
            prefix_parts.append(" ".join(metadata_parts))

        prefix = ". ".join(prefix_parts)
        if prefix and statement:
            return f"{prefix}: {statement}"
        return statement or "unknown statement"

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }