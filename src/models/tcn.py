"""Temporal Convolutional Network for AQRF.

~850K parameters. Fits in 4GB VRAM with batch_size=256.
Two output heads: direction (classification) + magnitude (regression).
"""
from typing import Optional

import torch
import torch.nn as nn
import structlog

from src.utils.config import TCNConfig

logger = structlog.get_logger(__name__)


class TCNBlock(nn.Module):
    """Dilated causal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Causal padding: pad left only so the conv never sees future timesteps
        self.causal_pad = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        # LayerNorm over the channel dim — more stable than BatchNorm for shuffled financial sequences
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, in_channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        residual = self.residual(x)

        # Apply causal (left-only) padding before conv
        out = torch.nn.functional.pad(x, (self.causal_pad, 0))
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)

        return out + residual


class TCN(nn.Module):
    """Temporal Convolutional Network for forex direction prediction."""

    def __init__(self, config: Optional[TCNConfig] = None):
        super().__init__()

        config = config or TCNConfig()
        self.config = config

        # Input: (batch, seq=60, features=45)
        # Conv1d expects (batch, channels, seq) so we permute

        channels = config.channels
        kernel_size = config.kernel_size
        dropout = config.dropout

        self.blocks = nn.ModuleList()
        in_ch = config.n_features  # feature dimension

        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            block = TCNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.blocks.append(block)
            in_ch = out_ch

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Shared dense layers
        self.dense = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Direction head: binary classification
        self.direction_head = nn.Sequential(
            nn.Linear(128, config.direction_classes),
        )

        # Magnitude head: regression
        self.magnitude_head = nn.Sequential(
            nn.Linear(128, 1),
        )

        self._init_weights()
        self._log_params()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _log_params(self) -> None:
        """Log model parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("tcn_initialized", total_params=total, trainable_params=trainable)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len=60, n_features=45)

        Returns:
            direction_logits: (batch, 2)
            magnitude: (batch, 1)
        """
        # Permute for Conv1d: (batch, features, seq)
        x = x.permute(0, 2, 1)

        # Apply TCN blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling: (batch, channels, 1) -> (batch, channels)
        x = self.global_pool(x).squeeze(-1)

        # Shared representation
        x = self.dense(x)

        # Output heads
        direction_logits = self.direction_head(x)
        magnitude = self.magnitude_head(x)

        return direction_logits, magnitude
