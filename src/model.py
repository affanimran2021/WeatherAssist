"""LSTM forecasting architecture for Toronto electricity demand anomaly detection.

Matches Group 6 Proposal Appendix A (Fig. A.1):
- Window size: 24 hours
- Input features: 2 (baseline: hour, demand) or 4 (context: hour, temp, holiday, demand)
- Hidden size: 32
- LSTM layers: 1
- Output size: 1 (next-hour demand prediction)
- Parameter count: 4,641 (input_size=2) / 4,897 (input_size=4)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DemandLSTM(nn.Module):
    """Single-layer LSTM + linear projection for next-hour demand forecasting."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next-hour scaled demand from a (batch, window_size, input_size) sequence."""
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        if self.output_size == 1:
            return pred.squeeze(-1)
        return pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
