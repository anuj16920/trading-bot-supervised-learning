"""Model ensemble for AQRF.

Combines TCN + Transformer with regime-aware weighting.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import structlog

from src.models.tcn import TCN
from src.models.transformer import ForexTransformer
from src.models.regime import RegimeDetector
from src.utils.config import ModelConfig
from src.utils.gpu import log_vram

logger = structlog.get_logger(__name__)


class ModelEnsemble:
    """Ensemble of TCN, Transformer, and Regime Detector."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tcn: Optional[TCN] = None
        self.transformer: Optional[ForexTransformer] = None
        self.regime: Optional[RegimeDetector] = None

        # Regime weights for direction adjustment
        self.regime_weights = {
            0: {"up": 1.2, "down": 0.8},   # trending_up
            1: {"up": 0.8, "down": 1.2},   # trending_down
            2: {"up": 1.0, "down": 1.0},   # ranging
            3: {"up": 0.5, "down": 0.5},   # volatile
        }

    def load_models(
        self,
        tcn_path: Optional[str] = None,
        transformer_path: Optional[str] = None,
        regime_path: Optional[str] = None,
    ) -> None:
        """Load trained model weights."""
        if tcn_path:
            self.tcn = TCN(self.config.tcn).to(self.device)
            self.tcn.load_state_dict(torch.load(tcn_path, map_location=self.device))
            self.tcn.eval()
            logger.info("tcn_loaded", path=tcn_path)

        if transformer_path:
            self.transformer = ForexTransformer(self.config.transformer).to(self.device)
            self.transformer.load_state_dict(
                torch.load(transformer_path, map_location=self.device)
            )
            self.transformer.eval()
            logger.info("transformer_loaded", path=transformer_path)

        if regime_path:
            self.regime = RegimeDetector(self.config.regime).to(self.device)
            self.regime.load_state_dict(torch.load(regime_path, map_location=self.device))
            self.regime.eval()
            logger.info("regime_loaded", path=regime_path)

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        features_regime: Optional[torch.Tensor] = None,
    ) -> dict:
        """Generate ensemble prediction.

        Args:
            features: (batch, seq=60, features=45) for TCN/Transformer
            features_regime: (batch, seq=100, features=45) for regime

        Returns:
            Dict with direction, confidence, magnitude, regime
        """
        features = features.to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            # TCN prediction
            if self.tcn is not None:
                tcn_dir_logits, tcn_mag = self.tcn(features)
                tcn_probs = torch.softmax(tcn_dir_logits, dim=-1)
                tcn_up = tcn_probs[:, 1].cpu().numpy()
                tcn_conf = tcn_probs.max(dim=-1)[0].cpu().numpy()
            else:
                tcn_up = 0.5
                tcn_conf = 0.0
                tcn_mag = torch.zeros(features.size(0), 1)

            # Transformer prediction
            if self.transformer is not None:
                tf_dir_logits, tf_mag = self.transformer(features)
                tf_probs = torch.softmax(tf_dir_logits, dim=-1)
                tf_up = tf_probs[:, 1].cpu().numpy()
                tf_conf = tf_probs.max(dim=-1)[0].cpu().numpy()
            else:
                tf_up = 0.5
                tf_conf = 0.0
                tf_mag = torch.zeros(features.size(0), 1)

            # Regime detection
            if self.regime is not None and features_regime is not None:
                features_regime = features_regime.to(self.device, non_blocking=True)
                regime_logits = self.regime(features_regime)
                regime_probs = torch.softmax(regime_logits, dim=-1)
                regime = regime_probs.argmax(dim=-1).cpu().numpy()
            else:
                regime = np.array([2] * features.size(0))  # default ranging

        # Ensemble with regime weighting
        results = []
        for i in range(features.size(0)):
            r = int(regime[i])
            weights = self.regime_weights.get(r, {"up": 1.0, "down": 1.0})

            # Weighted average
            raw_up = (
                self.config.tcn_weight * tcn_up[i] +
                self.config.transformer_weight * tf_up[i]
            )

            # Apply regime weight
            final_up = raw_up * weights["up"] / (weights["up"] + weights["down"]) * 2
            final_up = np.clip(final_up, 0, 1)

            confidence = max(final_up, 1 - final_up)
            direction = "up" if final_up > 0.5 else "down"

            magnitude = 0.5 * (tcn_mag[i].item() + tf_mag[i].item())

            results.append({
                "direction": direction,
                "confidence": float(confidence),
                "magnitude": float(magnitude),
                "regime": ["trending_up", "trending_down", "ranging", "volatile"][r],
                "tcn_conf": float(tcn_conf[i]),
                "tf_conf": float(tf_conf[i]),
            })

        return results[0] if len(results) == 1 else results
