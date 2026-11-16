"""Who spoke when. Dense embeddings over the timeline, then clustering or enrolled matching."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .audio import load_and_level, speech_mask
from .logging_utils import get_logger
from .numeric import matmul
from .scoring import l2_normalize
from .vad import mask_to_segments

log = get_logger(__name__)


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    score: float

    def __repr__(self) -> str:
        return f"Turn({self.speaker!r}, {self.start:.2f}-{self.end:.2f}s, score={self.score:.3f})"


def agglomerative(embeddings: np.ndarray, threshold: float = 0.55, n_clusters: Optional[int] = None) -> np.ndarray:
    """Average-linkage clustering on cosine similarity.

    Merges the closest pair until either n_clusters remain or the best similarity drops
    below threshold. O(n^2) memory, which is fine for a timeline of a few thousand windows.
    """
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)

    sim = matmul(embeddings, embeddings.T)
    np.fill_diagonal(sim, -np.inf)
    sizes = np.ones(n)
    active = np.ones(n, dtype=bool)
    labels = np.arange(n)

    while active.sum() > 1:
        if n_clusters is not None and active.sum() <= n_clusters:
            break
        masked = np.where(active[:, None] & active[None, :], sim, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        if n_clusters is None and masked[i, j] < threshold:
            break

        # Average linkage: the merged cluster's similarity is the size-weighted mean.
        sim[i] = (sim[i] * sizes[i] + sim[j] * sizes[j]) / (sizes[i] + sizes[j])
        sim[:, i] = sim[i]
        sim[i, i] = -np.inf
        sizes[i] += sizes[j]
        active[j] = False
        labels[labels == j] = i

    _, compact = np.unique(labels, return_inverse=True)
    return compact


def _turns_from_windows(labels, slices, scores, sample_rate: int, n_samples: int) -> List[Turn]:
    """Turn per-window labels into a contiguous timeline.

    Windows overlap, so a boundary is placed halfway between neighbouring window
    centres rather than at the window edges. Otherwise adjacent turns overlap.
    """
    if len(slices) == 0:
        return []
    centers = np.array([(s.start + s.stop) / 2 for s in slices], dtype=np.float64)
    edges = np.concatenate([[slices[0].start], (centers[:-1] + centers[1:]) / 2, [min(slices[-1].stop, n_samples)]])
    edges = np.clip(edges, 0, n_samples)
    return [
        Turn(str(label), float(edges[i] / sample_rate), float(edges[i + 1] / sample_rate), float(score))
        for i, (label, score) in enumerate(zip(labels, scores))
    ]


def _merge(turns: List[Turn], min_duration: float) -> List[Turn]:
    merged: List[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker and turn.start <= merged[-1].end + 1e-6:
            previous = merged[-1]
            merged[-1] = Turn(
                previous.speaker, previous.start, max(previous.end, turn.end), max(previous.score, turn.score)
            )
        else:
            merged.append(turn)
    return [t for t in merged if t.end - t.start >= min_duration]


def _to_source_time(turns: List[Turn], segments) -> List[Turn]:
    """Map turns on the trimmed (speech-only) timeline back to the source timeline.

    segments are the source-time speech spans (seconds, sorted) that were concatenated to
    form the trimmed waveform. A turn that crosses a junction in trimmed time is split,
    so the silence between segments never appears inside a turn.
    """
    offsets = np.concatenate([[0.0], np.cumsum([end - start for start, end in segments])])
    mapped: List[Turn] = []
    k = 0
    for turn in turns:
        while k < len(segments) and offsets[k + 1] <= turn.start:
            k += 1
        j = k
        while j < len(segments) and offsets[j] < turn.end:
            lo, hi = max(turn.start, offsets[j]), min(turn.end, offsets[j + 1])
            if hi > lo:
                shift = segments[j][0] - offsets[j]
                mapped.append(Turn(turn.speaker, lo + shift, hi + shift, turn.score))
            j += 1
    return mapped


def _timeline(recognizer, source, rate: float, source_sr: Optional[int]):
    """Dense embeddings of the speech, plus the source-time spans that speech came from.

    The encoder sees the same speech-only waveform as preprocess() would produce, so
    windows never mix speech with stretches of silence. Turn times are mapped back to
    the source afterwards with _to_source_time().
    """
    config = recognizer.config.preprocess
    wav = load_and_level(source, config, source_sr)
    mask = speech_mask(wav, config)
    speech = np.ascontiguousarray(wav[mask]) if not mask.all() else wav
    embeddings, slices = recognizer.encoder.embed_timeline(speech, rate=rate)
    sample_rate = recognizer.encoder.sample_rate
    segments = [(start / sample_rate, end / sample_rate) for start, end in mask_to_segments(mask)]
    return len(speech), embeddings, slices, segments


def _finish(labels, slices, scores, sample_rate: int, n_samples: int, segments, min_duration: float) -> List[Turn]:
    turns = _merge(_turns_from_windows(labels, slices, scores, sample_rate, n_samples), 0.0)
    return _merge(_to_source_time(turns, segments), min_duration)


def diarize(
    recognizer,
    source,
    rate: float = 4.0,
    n_speakers: Optional[int] = None,
    threshold: float = 0.55,
    min_duration: float = 0.5,
    source_sr: Optional[int] = None,
) -> List[Turn]:
    """Unsupervised diarization: cluster the timeline into "speaker_0", "speaker_1", ...

    Turn times are seconds from the start of the source; silence detected by the VAD is
    left out of the turns. Pass source_sr when source is a raw array.
    """
    n_samples, embeddings, slices, segments = _timeline(recognizer, source, rate, source_sr)
    labels = agglomerative(embeddings, threshold=threshold, n_clusters=n_speakers)
    log.info("diarized %d windows into %d speaker(s)", len(labels), len(set(labels)))

    return _finish(
        [f"speaker_{label}" for label in labels],
        slices,
        np.ones(len(labels)),
        recognizer.encoder.sample_rate,
        n_samples,
        segments,
        min_duration,
    )


def assign_to_enrolled(
    recognizer,
    source,
    rate: float = 4.0,
    threshold: Optional[float] = None,
    min_duration: float = 0.5,
    unknown_label: str = "unknown",
    source_sr: Optional[int] = None,
) -> List[Turn]:
    """Diarization against known speakers: label each window with the best enrolled match.

    Turn times are seconds from the start of the source; silence is left out of the turns.
    """
    names, centroids = recognizer.book.centroids()
    if not names:
        raise ValueError("no speakers enrolled; use diarize() for the unsupervised case")
    threshold = recognizer.threshold if threshold is None else threshold

    n_samples, embeddings, slices, segments = _timeline(recognizer, source, rate, source_sr)
    scores = recognizer.score_matrix(embeddings, centroids)  # same units as recognizer.threshold
    best = np.argmax(scores, axis=1)

    best_scores = scores[np.arange(len(best)), best]
    assigned = [names[b] if sc >= threshold else unknown_label for b, sc in zip(best, best_scores)]
    return _finish(assigned, slices, best_scores, recognizer.encoder.sample_rate, n_samples, segments, min_duration)
