from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TemperatureScaler:
    """Post-hoc temperature scaling for multiclass logits.

    T is fitted on a validation/calibration set only.
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def fit(self, logits, labels):
        import torch

        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        log_temperature = torch.tensor(
            np.log(self.temperature),
            dtype=torch.float32,
            requires_grad=True,
        )

        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=0.1,
            max_iter=100,
            line_search_fn="strong_wolfe",
        )
        criterion = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature).detach().item())
        return self

    def transform_logits(self, logits):
        return np.asarray(logits, dtype=float) / self.temperature

    def predict_proba(self, logits):
        x = self.transform_logits(logits)
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def save(self, path: str | Path):
        Path(path).write_text(
            json.dumps(
                {
                    "method": "temperature_scaling",
                    "temperature": self.temperature,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(data["temperature"]))
