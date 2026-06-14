"""Transformer model for AQRF.

~1.2M parameters. CLS token for sequence classification.
Gradient checkpointing support for 4GB VRAM.
"""
from typing import Optional

import torch
import torch.nn as nn
import structlog

from src.utils.config import TransformerConfig

logger = structlog.get_logger(__name__)


class ForexTransformer(nn.Module):
    """Transformer encoder for forex prediction."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()

        config = config or TransformerConfig()
        self.config = config

        d_model = config.d_model
        seq_len = 60

        # Input projection
        self.input_proj = nn.Linear(config.n_features, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Embedding(seq_len, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # Shared dense
        self.dense = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Direction head
        self.direction_head = nn.Linear(64, config.direction_classes)

        # Magnitude head
        self.magnitude_head = nn.Linear(64, 1)

        self._init_weights()
        self._log_params()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.pos_encoding.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _log_params(self) -> None:
        """Log parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("transformer_initialized", total_params=total, trainable_params=trainable)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len=60, n_features=45)

        Returns:
            direction_logits: (batch, 2)
            magnitude: (batch, 1)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Input projection
        x = self.input_proj(x)  # (batch, seq, d_model)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_encoding(positions)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (batch, seq+1, d_model)

        # Transformer
        if self.config.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.transformer, x)
        else:
            x = self.transformer(x)

        # Extract CLS output
        cls_output = x[:, 0, :]  # (batch, d_model)

        # Shared dense
        x = self.dense(cls_output)

        # Output heads
        direction_logits = self.direction_head(x)
        magnitude = self.magnitude_head(x)

        return direction_logits, magnitude
