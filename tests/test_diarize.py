import numpy as np
import pytest

from voicefingerprint.diarize import Turn, _merge, agglomerative


def _clusters(n_clusters, per_cluster, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    return np.concatenate([c + 0.03 * rng.standard_normal((per_cluster, dim)) for c in centers])


def test_recovers_known_cluster_count():
    labels = agglomerative(_clusters(3, 10), n_clusters=3)
    assert len(set(labels)) == 3
    for start in (0, 10, 20):
        assert len(set(labels[start : start + 10])) == 1


def test_threshold_mode_finds_structure():
    labels = agglomerative(_clusters(2, 12), threshold=0.9)
    assert 1 < len(set(labels)) <= 4


def test_labels_are_compact():
    labels = agglomerative(_clusters(4, 5), n_clusters=4)
    assert sorted(set(labels)) == list(range(len(set(labels))))


def test_empty_input():
    assert len(agglomerative(np.zeros((0, 16)))) == 0


def test_merge_joins_adjacent_and_drops_short_turns():
    turns = [
        Turn("a", 0.0, 1.0, 0.9),
        Turn("a", 1.0, 2.0, 0.8),
        Turn("b", 2.0, 2.1, 0.7),
        Turn("a", 2.1, 4.0, 0.6),
    ]
    merged = _merge(turns, min_duration=0.5)
    assert [t.speaker for t in merged] == ["a", "a"]
    assert merged[0].start == 0.0 and merged[0].end == 2.0


def test_turns_from_windows_are_contiguous_and_non_overlapping():
    from voicefingerprint.diarize import _turns_from_windows

    sr = 16000
    slices = [slice(i * sr, i * sr + 3 * sr) for i in range(5)]
    labels = ["a", "a", "b", "b", "b"]
    turns = _turns_from_windows(labels, slices, np.ones(5), sr, 7 * sr)

    assert turns[0].start == 0.0
    assert turns[-1].end == 7.0
    for previous, following in zip(turns, turns[1:]):
        assert previous.end == following.start


def test_turns_clamp_to_the_waveform_length():
    from voicefingerprint.diarize import _turns_from_windows

    sr = 16000
    slices = [slice(0, 3 * sr), slice(2 * sr, 5 * sr)]
    turns = _turns_from_windows(["a", "b"], slices, np.ones(2), sr, int(4.2 * sr))
    assert turns[-1].end == pytest.approx(4.2)



# --- timestamps stay in source time when the VAD removes silence -----------------------

from types import SimpleNamespace

from voicefingerprint.chunking import compute_chunks
from voicefingerprint.config import ChunkConfig, Config, PreprocessConfig
from voicefingerprint.diarize import _to_source_time, assign_to_enrolled, diarize
from voicefingerprint.scoring import l2_normalize

SR = 16000


class _FakeEncoder:
    """Embeds a window as [low-band power, high-band power]: a 200 Hz and a 1 kHz "speaker"."""

    sample_rate = SR

    def __init__(self):
        self.seen = []

    def embed_timeline(self, wav, rate=4.0):
        self.seen.append(len(wav))
        slices = compute_chunks(len(wav), SR, ChunkConfig(rate=rate))
        padded = np.pad(wav, (0, max(0, slices[-1].stop - len(wav))))
        rows = []
        for s in slices:
            power = np.abs(np.fft.rfft(padded[s])) ** 2
            freqs = np.fft.rfftfreq(s.stop - s.start, 1 / SR)
            rows.append([power[freqs < 600].sum(), power[freqs >= 600].sum(), 1e-9])
        return l2_normalize(np.array(rows)), slices


def _tone(freq, seconds):
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speech_gap_speech(gap_s=6.0):
    """Speaker A (200 Hz) 0-4 s, digital silence, speaker B (1 kHz) for 4 s."""
    return np.concatenate([_tone(200, 4.0), np.zeros(int(gap_s * SR), dtype=np.float32), _tone(1000, 4.0)])


def _fake_recognizer(vad="energy", book=None, threshold=0.4):
    return SimpleNamespace(
        config=Config(preprocess=PreprocessConfig(vad_backend=vad)),
        encoder=_FakeEncoder(),
        book=book,
        threshold=threshold,
    )


def test_to_source_time_shifts_and_splits_at_junctions():
    segments = [(1.0, 3.0), (10.0, 12.0)]  # trimmed timeline: 0-2 s and 2-4 s
    turns = [Turn("a", 0.0, 1.5, 0.9), Turn("b", 1.5, 4.0, 0.8)]
    mapped = _to_source_time(turns, segments)
    assert [(t.speaker, t.start, t.end, t.score) for t in mapped] == [
        ("a", 1.0, 2.5, 0.9), ("b", 2.5, 3.0, 0.8), ("b", 10.0, 12.0, 0.8),
    ]


def test_to_source_time_is_identity_without_silence():
    turns = [Turn("a", 0.0, 2.0, 1.0), Turn("b", 2.0, 5.0, 1.0)]
    assert _to_source_time(turns, [(0.0, 5.0)]) == turns


def test_diarize_reports_times_in_the_source_timeline():
    """Regression: trimming silence used to shift every later turn earlier."""
    recognizer = _fake_recognizer()
    turns = diarize(recognizer, _speech_gap_speech(), n_speakers=2, source_sr=SR)

    assert recognizer.encoder.seen[0] < 9 * SR  # the encoder only saw the speech
    assert len(turns) == 2
    first, second = turns
    assert first.speaker != second.speaker
    assert first.start == pytest.approx(0.0, abs=0.05)
    assert first.end == pytest.approx(4.0, abs=0.3)
    assert second.start == pytest.approx(10.0, abs=0.3)  # was ~4.2 s before the fix
    assert second.end == pytest.approx(14.0, abs=0.05)


def test_assign_to_enrolled_reports_times_in_the_source_timeline():
    book = SimpleNamespace(centroids=lambda: (["alice", "bob"], np.array([[1.0, 0, 0], [0, 1.0, 0]])))
    turns = assign_to_enrolled(_fake_recognizer(book=book), _speech_gap_speech(), source_sr=SR)

    assert [t.speaker for t in turns] == ["alice", "bob"]
    assert turns[0].end == pytest.approx(4.0, abs=0.3)
    assert turns[1].start == pytest.approx(10.0, abs=0.3)


def test_diarize_without_vad_keeps_the_whole_timeline():
    turns = diarize(_fake_recognizer(vad="none"), _speech_gap_speech(), n_speakers=2, source_sr=SR)
    assert turns[0].start == 0.0 and turns[-1].end == pytest.approx(14.0)
    for previous, following in zip(turns, turns[1:]):
        assert previous.end == following.start
