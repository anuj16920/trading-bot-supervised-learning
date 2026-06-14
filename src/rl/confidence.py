"""Confidence-aware trading utilities (Module 2 — Phase 3).

The PPO policy outputs action probabilities. These functions extract those
probabilities, filter low-confidence actions to HOLD, and compute policy
entropy for monitoring.

Usage in inference / paper trading:
    probs = get_action_probs(model, obs)
    action = filter_action(raw_action, probs, threshold=0.65)
    entropy = compute_entropy(probs)

NOT called inside ForexTradingEnv — the environment is action-agnostic.
The filter lives at the inference layer (RLPaperTrader, eval scripts).
"""
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO


def get_action_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    """Return action probability vector (shape: [4]) for a single observation.

    Args:
        model: Loaded PPO model.
        obs:   Single observation array of shape (seq_len, n_features).
               Must already be normalized (same scale as training data).

    Returns:
        np.ndarray of shape (4,) — probabilities for [hold, buy, sell, close].
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs[np.newaxis])  # add batch dim
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()
    return probs.astype(np.float32)


def filter_action(action: int, probs: np.ndarray, threshold: float = 0.65) -> int:
    """Return action if max(probs) >= threshold, else 0 (HOLD).

    A low max probability means the policy is uncertain — forcing HOLD avoids
    noisy low-confidence entries that increase overtrading.

    Args:
        action:    Raw action from model.predict().
        probs:     Action probability vector from get_action_probs().
        threshold: Minimum confidence required to execute a trade action.

    Returns:
        Original action if confident, 0 (HOLD) otherwise.
    """
    if float(np.max(probs)) >= threshold:
        return action
    return 0  # force HOLD on low-confidence steps


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of the action probability distribution.

    High entropy = uncertain policy (all actions roughly equal probability).
    Low entropy  = confident policy (one action dominates).

    Args:
        probs: Action probability vector of shape (4,).

    Returns:
        Scalar entropy in nats.
    """
    probs = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def batch_action_probs(model: PPO, obs_batch: np.ndarray) -> np.ndarray:
    """Return action probabilities for a batch of observations.

    Args:
        model:     Loaded PPO model.
        obs_batch: Array of shape (B, seq_len, n_features).

    Returns:
        np.ndarray of shape (B, 4).
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()
    return probs.astype(np.float32)
