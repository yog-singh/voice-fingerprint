"""Persistent enrollment store: named speakers and their embedding centroids."""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .exceptions import EnrollmentError
from .logging_utils import get_logger
from .numeric import matmul
from .scoring import SCORE_ASNORM, SCORE_COSINE, l2_normalize

log = get_logger(__name__)

FORMAT_VERSION = 2
READABLE_VERSIONS = (1, 2)  # v1 had no thresholds and no cohort
SCORE_TYPES = (SCORE_COSINE, SCORE_ASNORM)
DEFAULT_COHORT_TOP_K = 300
SUFFIX = ".npz"


def book_path(path) -> Path:
    """File a book named `path` is written to: `path` itself if it ends in .npz, else path + ".npz".

    Appending (rather than Path.with_suffix) keeps dotted names distinct: "team.v2" is
    "team.v2.npz", not "team.npz".
    """
    path = Path(path)
    return path if path.suffix == SUFFIX else path.with_name(path.name + SUFFIX)


def find_book(path) -> Optional[Path]:
    """Existing file for a book named `path`, or None.

    Falls back to the pre-fix location (with_suffix, so "team.v2" -> "team.npz") so
    books saved by older versions still load; the next save writes the new location.
    """
    current = book_path(path)
    if current.is_file():
        return current
    legacy = Path(path).with_suffix(SUFFIX)
    if legacy != current and legacy.is_file():
        log.warning("loading %s from its legacy location %s; it will be saved to %s", path, legacy, current)
        return legacy
    return None


def _check_metadata(name: str, metadata: dict) -> None:
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise EnrollmentError(
            f"metadata for {name!r} must be JSON-serializable (str, int, float, bool, None, list, dict), "
            f"since it is stored in the book header: {exc}"
        ) from None


@dataclass
class Speaker:
    name: str
    embeddings: np.ndarray  # (n_utterances, dim), each L2-normalized
    metadata: dict = field(default_factory=dict)

    @property
    def centroid(self) -> np.ndarray:
        return l2_normalize(self.embeddings.mean(axis=0))

    @property
    def n_utterances(self) -> int:
        return len(self.embeddings)

    def self_consistency(self) -> float:
        """Mean pairwise similarity of the enrolled utterances. Low values mean noisy enrollment."""
        if self.n_utterances < 2:
            return float("nan")
        sim = matmul(self.embeddings, self.embeddings.T)
        iu = np.triu_indices(self.n_utterances, k=1)
        return float(sim[iu].mean())


class SpeakerBook:
    """A small in-memory database of enrolled speakers, serializable to a single .npz file.

    Besides the speakers it holds the scoring setup they were calibrated under:
    thresholds keyed by score type ("cosine" or "asnorm") and the optional AS-norm cohort.
    An "asnorm" threshold only means something for the cohort it was fitted with, so
    set_cohort() drops it whenever the cohort actually changes.
    """

    def __init__(self, model: str, embedding_dim: int):
        self.model = model
        self.embedding_dim = embedding_dim
        self._speakers: Dict[str, Speaker] = {}
        self.thresholds: Dict[str, float] = {}
        self.cohort: Optional[np.ndarray] = None  # (n_cohort, dim), L2-normalized
        self.cohort_top_k: int = DEFAULT_COHORT_TOP_K
        self.cohort_version = 0  # bumped on every cohort change, so callers can cache an ASNorm

    def __len__(self) -> int:
        return len(self._speakers)

    def __contains__(self, name: str) -> bool:
        return name in self._speakers

    def __getitem__(self, name: str) -> Speaker:
        try:
            return self._speakers[name]
        except KeyError:
            raise EnrollmentError(f"speaker {name!r} is not enrolled; known: {self.names}") from None

    @property
    def names(self) -> List[str]:
        return sorted(self._speakers)

    def add(self, name: str, embeddings: np.ndarray, metadata: Optional[dict] = None) -> Speaker:
        """Enroll a speaker, or append to an existing enrollment."""
        embeddings = l2_normalize(np.atleast_2d(np.asarray(embeddings, dtype=np.float32)))
        if embeddings.shape[1] != self.embedding_dim:
            raise EnrollmentError(f"expected {self.embedding_dim}-d embeddings, got {embeddings.shape[1]}-d")

        metadata = dict(metadata or {})
        merged = {**self._speakers[name].metadata, **metadata} if name in self._speakers else metadata
        _check_metadata(name, merged)  # fail here, not at save() time

        if name in self._speakers:
            existing = self._speakers[name]
            existing.embeddings = np.concatenate([existing.embeddings, embeddings])
            existing.metadata = merged
        else:
            self._speakers[name] = Speaker(name, embeddings, metadata)
        log.info("enrolled %r with %d utterance(s) total", name, self._speakers[name].n_utterances)
        return self._speakers[name]

    def remove(self, name: str) -> None:
        if self._speakers.pop(name, None) is None:
            raise EnrollmentError(f"speaker {name!r} is not enrolled")

    def set_threshold(self, score_type: str, value: float) -> None:
        if score_type not in SCORE_TYPES:
            raise ValueError(f"unknown score type {score_type!r}; expected one of {SCORE_TYPES}")
        self.thresholds[score_type] = float(value)

    def set_cohort(self, embeddings: Optional[np.ndarray], top_k: int = DEFAULT_COHORT_TOP_K) -> None:
        """Store (or with None, clear) the AS-norm cohort. Drops a stale "asnorm" threshold."""
        if embeddings is not None:
            embeddings = l2_normalize(np.atleast_2d(np.asarray(embeddings, dtype=np.float32)))
            if embeddings.shape[1] != self.embedding_dim:
                raise EnrollmentError(f"expected {self.embedding_dim}-d cohort embeddings, got {embeddings.shape[1]}-d")
            if len(embeddings) == 0:
                raise EnrollmentError("cohort is empty")

        unchanged = (
            (embeddings is None and self.cohort is None)
            or (
                embeddings is not None
                and self.cohort is not None
                and top_k == self.cohort_top_k
                and np.array_equal(embeddings, self.cohort)
            )
        )
        if unchanged:
            return
        if self.thresholds.pop(SCORE_ASNORM, None) is not None:
            log.warning("cohort changed; dropping the AS-norm threshold calibrated on the old cohort")
        self.cohort = embeddings
        self.cohort_top_k = int(top_k)
        self.cohort_version += 1

    def centroids(self) -> Tuple[List[str], np.ndarray]:
        """Names and their (n_speakers, dim) centroid matrix, in a stable order."""
        if not self._speakers:
            return [], np.zeros((0, self.embedding_dim), dtype=np.float32)
        names = self.names
        return names, np.stack([self._speakers[n].centroid for n in names])

    def all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Every enrolled embedding with its speaker label. Feeds calibrate_threshold."""
        rows, labels = [], []
        for name in self.names:
            speaker = self._speakers[name]
            rows.append(speaker.embeddings)
            labels.extend([name] * speaker.n_utterances)
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        return np.concatenate(rows), labels

    def save(self, path) -> Path:
        """Write the book to `path` (".npz" appended unless present). Returns the file written.

        The write is atomic: the book goes to a temporary file in the same directory,
        which then replaces the target, so a crash leaves either the old book or the new
        one, never a truncated file.
        """
        path = book_path(path)
        arrays, index = {}, []
        for i, name in enumerate(self.names):
            speaker = self._speakers[name]
            arrays[f"emb_{i}"] = speaker.embeddings
            index.append({"name": name, "key": f"emb_{i}", "metadata": speaker.metadata})

        if self.cohort is not None:
            arrays["cohort"] = self.cohort

        header = json.dumps(
            {
                "version": FORMAT_VERSION,
                "model": self.model,
                "embedding_dim": self.embedding_dim,
                "speakers": index,
                "thresholds": self.thresholds,
                "cohort": {"key": "cohort", "top_k": self.cohort_top_k} if self.cohort is not None else None,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, header=np.array(header), **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        _fsync_dir(path.parent)
        log.info("saved %d speaker(s) to %s", len(self), path)
        return path

    @classmethod
    def load(cls, path, expect_model: Optional[str] = None) -> "SpeakerBook":
        found = find_book(path)
        if found is None:
            raise EnrollmentError(f"speaker book not found: {book_path(path)}")
        path = found

        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            if header.get("version") not in READABLE_VERSIONS:
                raise EnrollmentError(f"unsupported speaker book version {header.get('version')}")
            # Embeddings from different encoders are not comparable, so refuse to mix them.
            if expect_model and header["model"] != expect_model:
                raise EnrollmentError(
                    f"{path} was built with model {header['model']!r} but the encoder is {expect_model!r}"
                )
            book = cls(header["model"], header["embedding_dim"])
            for entry in header["speakers"]:
                book.add(entry["name"], data[entry["key"]], entry.get("metadata"))
            for score_type, value in (header.get("thresholds") or {}).items():
                book.set_threshold(score_type, value)
            cohort = header.get("cohort")
            if cohort:
                book.cohort = l2_normalize(np.asarray(data[cohort["key"]], dtype=np.float32))
                book.cohort_top_k = int(cohort.get("top_k", DEFAULT_COHORT_TOP_K))
                book.cohort_version += 1
        return book

    def summary(self) -> List[dict]:
        return [
            {
                "name": name,
                "utterances": self._speakers[name].n_utterances,
                "self_consistency": round(self._speakers[name].self_consistency(), 4),
            }
            for name in self.names
        ]


def _fsync_dir(directory: Path) -> None:
    """Make the rename itself durable. Not supported on Windows, where it is skipped."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
