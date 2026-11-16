"""High-level API: enroll speakers, verify a pair, identify against the book."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .audio import AudioInput, preprocess
from .config import Config
from .encoder import SpeakerEncoder
from .exceptions import CalibrationError, EnrollmentError
from .logging_utils import get_logger
from .scoring import SCORE_COSINE, ASNorm, calibrate_threshold, cosine_matrix, score_type
from .store import DEFAULT_COHORT_TOP_K, SpeakerBook

log = get_logger(__name__)


@dataclass
class Match:
    name: str
    score: float
    accepted: bool


@dataclass
class VerificationResult:
    score: float
    threshold: float
    accepted: bool
    normalized: bool

    def __bool__(self) -> bool:
        return self.accepted


class VoiceRecognizer:
    """Ties together preprocessing, the encoder and an enrollment store.

    Note: this answers "does this voice match the enrolled one", not "is a live human
    speaking". A replayed recording of an enrolled speaker will pass. Anti-spoofing is a
    separate problem and deliberately not folded into verify().

    Thresholds: raw cosine and AS-norm scores live on different scales, so a threshold
    always belongs to one score type. The book stores one per type (saved with it), and
    the active one follows whether a cohort is set. An explicit ``threshold=`` argument
    overrides it for this instance only and is never saved. With AS-norm enabled and no
    calibrated threshold there is no sensible default, so scoring decisions raise
    CalibrationError until calibrate() runs or a threshold is given.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        book: Optional[SpeakerBook] = None,
        threshold: Optional[float] = None,
        cohort: Optional[np.ndarray] = None,
    ):
        self.config = config or Config()
        self.encoder = SpeakerEncoder(self.config.encoder, self.config.chunk)
        self.book = book or SpeakerBook(self.encoder.spec.name, self.encoder.embedding_dim)
        self.threshold_override: Optional[float] = threshold
        self._asnorm_cache = (None, -1, None)  # (book, cohort_version, ASNorm)

        if self.book.model != self.encoder.spec.name:
            raise EnrollmentError(
                f"speaker book was built with {self.book.model!r} but the encoder is {self.encoder.spec.name!r}"
            )
        if cohort is not None:
            self.book.set_cohort(cohort)

    # -- scoring setup -------------------------------------------------------------

    @property
    def asnorm(self) -> Optional[ASNorm]:
        """The AS-norm scorer for the book's cohort, or None when scoring raw cosine."""
        book = self.book
        if book.cohort is None:
            return None
        cached_book, version, scorer = self._asnorm_cache
        if cached_book is not book or version != book.cohort_version:
            scorer = ASNorm(book.cohort, book.cohort_top_k)
            self._asnorm_cache = (book, book.cohort_version, scorer)
        return scorer

    @property
    def score_type(self) -> str:
        """Units of every score this instance produces: "cosine" or "asnorm"."""
        return score_type(self.asnorm)

    @property
    def threshold(self) -> float:
        """Decision threshold for the current score type (override > book > model default)."""
        if self.threshold_override is not None:
            return self.threshold_override
        kind = self.score_type
        if kind in self.book.thresholds:
            return self.book.thresholds[kind]
        if kind == SCORE_COSINE:
            return self.encoder.spec.default_threshold
        raise CalibrationError(
            "AS-norm is enabled but no AS-norm threshold is calibrated; the cosine default does not "
            "apply to normalized scores. Run calibrate() (CLI: voicefingerprint calibrate) or pass a threshold."
        )

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the threshold for the current score type in the book (saved with it)."""
        self.book.set_threshold(self.score_type, value)
        self.threshold_override = None

    def _resolve(self, threshold: Optional[float]) -> float:
        return self.threshold if threshold is None else threshold

    def preprocess(self, source: AudioInput, source_sr: Optional[int] = None) -> np.ndarray:
        return preprocess(source, self.config.preprocess, source_sr)

    def embed(self, source: AudioInput, source_sr: Optional[int] = None) -> np.ndarray:
        """Embedding for one audio file, array or preprocessed waveform."""
        return self.encoder.embed_utterance(self.preprocess(source, source_sr))

    def embed_many(self, sources: Sequence[AudioInput], source_sr: Optional[int] = None) -> np.ndarray:
        return np.stack([self.embed(s, source_sr) for s in sources])

    def enroll(self, name: str, sources: Sequence[AudioInput], source_sr: Optional[int] = None, **metadata):
        """Enroll a speaker from several clips. Three or more varied clips beats one long one."""
        if not sources:
            raise EnrollmentError(f"no audio supplied for {name!r}")
        embeddings = self.embed_many(sources, source_sr)
        speaker = self.book.add(name, embeddings, metadata or None)
        if speaker.n_utterances >= 2 and speaker.self_consistency() < 0.5:
            log.warning(
                "enrollment for %r is inconsistent (%.2f); the clips may contain different speakers",
                name,
                speaker.self_consistency(),
            )
        return speaker

    def score(self, a: AudioInput, b: AudioInput, source_sr: Optional[int] = None) -> float:
        return self.score_embeddings(self.embed(a, source_sr), self.embed(b, source_sr))

    def score_embeddings(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self.score_matrix(a, b)[0, 0])

    def score_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Scores between rows of a (n, d) and b (m, d), in this instance's score type."""
        raw = cosine_matrix(a, b)
        asnorm = self.asnorm
        return asnorm.matrix(a, b, raw) if asnorm is not None else raw

    def verify(
        self,
        a: AudioInput,
        b: AudioInput,
        threshold: Optional[float] = None,
        source_sr: Optional[int] = None,
    ) -> VerificationResult:
        threshold = self._resolve(threshold)
        score = self.score(a, b, source_sr)
        return VerificationResult(score, threshold, score >= threshold, self.asnorm is not None)

    def verify_speaker(
        self, name: str, source: AudioInput, threshold: Optional[float] = None, source_sr: Optional[int] = None
    ) -> VerificationResult:
        """Check a clip against one enrolled speaker."""
        threshold = self._resolve(threshold)
        score = self.score_embeddings(self.book[name].centroid, self.embed(source, source_sr))
        return VerificationResult(score, threshold, score >= threshold, self.asnorm is not None)

    def identify(
        self,
        source: AudioInput,
        top_k: int = 3,
        threshold: Optional[float] = None,
        source_sr: Optional[int] = None,
    ) -> List[Match]:
        """Rank enrolled speakers against a clip. An empty accepted set means 'unknown'."""
        if len(self.book) == 0:
            raise EnrollmentError("no speakers enrolled")
        threshold = self._resolve(threshold)

        embedding = self.embed(source, source_sr)
        names, centroids = self.book.centroids()
        scores = self.score_matrix(embedding, centroids)[0]

        order = np.argsort(-scores)[:top_k]
        return [Match(names[i], float(scores[i]), bool(scores[i] >= threshold)) for i in order]

    def calibrate(self) -> dict:
        """Fit a threshold on the enrolled data and adopt it for the current score type.

        Scores trials exactly as verify()/identify() will (AS-norm when a cohort is set), and
        stores the result in the book so save() persists it. Needs >= 2 speakers and at least
        one speaker with >= 2 clips.
        """
        embeddings, labels = self.book.all_embeddings()
        if len(set(labels)) < 2:
            raise CalibrationError("calibration needs at least two enrolled speakers")
        if len(set(labels)) == len(labels):
            raise CalibrationError("calibration needs at least one speaker enrolled with two or more clips")
        report = calibrate_threshold(embeddings, labels, asnorm=self.asnorm)
        self.threshold = report["threshold"]
        log.info(
            "calibrated %s threshold=%.4f (EER %.2f%%)", report["score_type"], report["threshold"], 100 * report["eer"]
        )
        return report

    def set_cohort(self, sources: Sequence[AudioInput], top_k: int = DEFAULT_COHORT_TOP_K) -> None:
        """Enable AS-norm using background audio from speakers you do NOT want to recognize.

        The cohort is stored in the book. Any AS-norm threshold fitted on a different cohort
        is dropped, so run calibrate() again afterwards.
        """
        self.book.set_cohort(self.embed_many(sources), top_k)

    def clear_cohort(self) -> None:
        """Go back to raw cosine scoring (and the cosine threshold)."""
        self.book.set_cohort(None)

    def save(self, path: Union[str, Path]) -> Path:
        return self.book.save(path)

    def load(self, path: Union[str, Path]) -> SpeakerBook:
        """Replace the book. Its thresholds and cohort become the active scoring setup."""
        self.book = SpeakerBook.load(path, expect_model=self.encoder.spec.name)
        return self.book
