"""Similarity scoring, score normalization and threshold calibration."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .numeric import matmul


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(matmul(l2_normalize(a.ravel()), l2_normalize(b.ravel())))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of a (n, d) and b (m, d)."""
    return matmul(l2_normalize(np.atleast_2d(a)), l2_normalize(np.atleast_2d(b)).T)


@dataclass
class ASNorm:
    """Adaptive symmetric score normalization against a cohort of background embeddings.

    Raw cosine scores drift with channel and recording conditions. AS-norm rescales a
    score by the statistics of each side against its most similar cohort speakers,
    which makes a single threshold transfer across microphones far better.
    """

    cohort: np.ndarray
    top_k: int = 300

    def __post_init__(self):
        self.cohort = l2_normalize(np.atleast_2d(np.asarray(self.cohort, dtype=np.float32)))
        self.top_k = min(self.top_k, len(self.cohort))

    def stats(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and std of each row's top-k cohort scores, shape (n,) each."""
        scores = matmul(l2_normalize(np.atleast_2d(embeddings)), self.cohort.T)
        top = np.sort(scores, axis=1)[:, -self.top_k :]
        return top.mean(axis=1), top.std(axis=1) + 1e-9

    def matrix(self, a: np.ndarray, b: np.ndarray, raw: Optional[np.ndarray] = None) -> np.ndarray:
        """AS-normalized scores between rows of a (n, d) and b (m, d), shape (n, m)."""
        raw = cosine_matrix(a, b) if raw is None else raw
        mu_a, sd_a = self.stats(a)
        mu_b, sd_b = self.stats(b)
        return 0.5 * ((raw - mu_a[:, None]) / sd_a[:, None] + (raw - mu_b[None, :]) / sd_b[None, :])

    def __call__(self, enroll: np.ndarray, test: np.ndarray, raw: Optional[float] = None) -> float:
        raw_matrix = None if raw is None else np.array([[raw]])
        return float(self.matrix(enroll.ravel(), test.ravel(), raw_matrix)[0, 0])


def equal_error_rate(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """Return (EER, threshold at the EER). labels: 1 for same speaker, 0 for different."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if not (labels == 1).any() or not (labels == 0).any():
        raise ValueError("EER needs at least one same-speaker and one different-speaker trial")

    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]

    n_target, n_nontarget = int((labels == 1).sum()), int((labels == 0).sum())
    # Sweeping the threshold downward: accepting the top i trials.
    false_accepts = np.concatenate([[0], np.cumsum(labels == 0)])
    false_rejects = n_target - np.concatenate([[0], np.cumsum(labels == 1)])

    far = false_accepts / n_nontarget
    frr = false_rejects / n_target
    idx = int(np.argmin(np.abs(far - frr)))
    # Index i means "accept the top i trials", so the operating point is the last accepted score.
    threshold = float(scores[idx - 1]) if idx > 0 else float(scores[0]) + 1e-6
    return float((far[idx] + frr[idx]) / 2), threshold


SCORE_COSINE = "cosine"
SCORE_ASNORM = "asnorm"


def score_type(asnorm: Optional[ASNorm]) -> str:
    """Name of the units scores are in. A threshold is only valid for scores of its type."""
    return SCORE_ASNORM if asnorm is not None else SCORE_COSINE


def calibrate_threshold(
    embeddings: np.ndarray, speaker_ids: Sequence[str], asnorm: Optional[ASNorm] = None
) -> dict:
    """Derive an operating threshold from labelled embeddings.

    Pass several utterances per speaker for at least a handful of speakers. A hardcoded
    threshold is meaningless across microphones and languages; this is the honest way.
    With asnorm, trials are scored the same way verification will score them, so the
    threshold comes out in AS-norm units; the report's "score_type" says which.
    """
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    ids = np.asarray(speaker_ids)
    if len(embeddings) != len(ids):
        raise ValueError("embeddings and speaker_ids must have the same length")

    sim = asnorm.matrix(embeddings, embeddings) if asnorm is not None else matmul(embeddings, embeddings.T)
    iu = np.triu_indices(len(ids), k=1)
    scores = sim[iu]
    labels = (ids[iu[0]] == ids[iu[1]]).astype(np.int8)

    eer, threshold = equal_error_rate(scores, labels)
    return {
        "threshold": threshold,
        "score_type": score_type(asnorm),
        "eer": eer,
        "n_trials": int(len(scores)),
        "n_target": int(labels.sum()),
        "n_speakers": int(len(set(ids))),
    }
