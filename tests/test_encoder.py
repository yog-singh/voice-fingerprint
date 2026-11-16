"""Live-model tests. Skipped automatically when the ONNX weights cannot be fetched."""

import numpy as np
import pytest

from voicefingerprint import Config, VoiceRecognizer
from voicefingerprint.exceptions import EnrollmentError, ModelError
from voicefingerprint.store import SpeakerBook

SR = 16000


def test_signature_matches_the_registry(encoder):
    info = encoder.describe()
    assert info["embedding_dim"] == 256
    assert info["sample_rate"] == SR


def test_embeddings_are_unit_length(encoder, wav_a):
    embedding = encoder.embed_utterance(wav_a)
    assert embedding.shape == (256,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic(encoder, wav_a):
    np.testing.assert_allclose(encoder.embed_utterance(wav_a), encoder.embed_utterance(wav_a), atol=1e-6)


def test_same_source_scores_above_different_source(encoder, wav_a, wav_b):
    from voicefingerprint.scoring import cosine

    first, second = encoder.embed_utterance(wav_a[: 2 * SR]), encoder.embed_utterance(wav_a[2 * SR :])
    other = encoder.embed_utterance(wav_b)
    assert cosine(first, second) > cosine(first, other)


def test_window_count_tracks_the_rate(encoder, wav_a):
    sparse, _ = encoder.embed_timeline(wav_a, rate=1.0)
    dense, slices = encoder.embed_timeline(wav_a, rate=8.0)
    assert len(dense) > len(sparse)
    assert len(dense) == len(slices)


def test_audio_shorter_than_one_frame_raises(encoder):
    with pytest.raises(ModelError, match="need at least"):
        encoder.embed_utterance(np.zeros(50, dtype=np.float32))


def test_recognizer_enroll_identify_roundtrip(encoder, tmp_path, wav_a, wav_b):
    recognizer = VoiceRecognizer(Config())
    recognizer.enroll("a", [wav_a[: 2 * SR], wav_a[2 * SR :]], source_sr=SR)
    recognizer.enroll("b", [wav_b[: 2 * SR], wav_b[2 * SR :]], source_sr=SR)

    matches = recognizer.identify(wav_a[: 3 * SR], source_sr=SR)
    assert matches[0].name == "a"

    path = recognizer.save(tmp_path / "book")
    recognizer.load(path)
    assert recognizer.book.names == ["a", "b"]

    report = recognizer.calibrate()
    assert report["n_speakers"] == 2
    assert recognizer.threshold == report["threshold"]


def test_identify_without_enrollment_raises(encoder, wav_a):
    with pytest.raises(EnrollmentError, match="no speakers enrolled"):
        VoiceRecognizer(Config()).identify(wav_a, source_sr=SR)


def test_diarize_timestamps_survive_a_silent_gap(encoder, wav_a, wav_b):
    """With the real model and default energy VAD, speaker B still starts at ~10 s, not ~4 s."""
    from voicefingerprint import diarize

    signal = np.concatenate([wav_a, np.zeros(6 * SR, dtype=np.float32), wav_b])
    turns = diarize(VoiceRecognizer(Config()), signal, n_speakers=2, source_sr=SR)

    assert turns[0].start == pytest.approx(0.0, abs=0.05)
    assert turns[-1].end == pytest.approx(14.0, abs=0.05)
    assert turns[-1].start == pytest.approx(10.0, abs=0.5)
    assert not any(t.start < 9.0 and t.end > 5.0 for t in turns), "no turn should cover the silent gap"


def _two_speaker_recognizer(wav_a, wav_b, **kwargs):
    recognizer = VoiceRecognizer(Config(), **kwargs)
    recognizer.enroll("a", [wav_a[: 2 * SR], wav_a[2 * SR :]], source_sr=SR)
    recognizer.enroll("b", [wav_b[: 2 * SR], wav_b[2 * SR :]], source_sr=SR)
    return recognizer


def _cohort_embeddings(encoder, n=6):
    from conftest import _voice_like

    return np.stack([encoder.embed_utterance(_voice_like(3.0, 90.0 + 25 * i, seed=100 + i)) for i in range(n)])


def test_calibrated_threshold_is_saved_and_restored(encoder, tmp_path, wav_a, wav_b):
    recognizer = _two_speaker_recognizer(wav_a, wav_b)
    report = recognizer.calibrate()
    assert report["score_type"] == "cosine"
    path = recognizer.save(tmp_path / "book")

    fresh = VoiceRecognizer(Config())
    assert fresh.threshold == encoder.spec.default_threshold
    fresh.load(path)
    assert fresh.threshold == pytest.approx(report["threshold"])


def test_explicit_threshold_overrides_but_is_not_saved(encoder, tmp_path, wav_a, wav_b):
    recognizer = _two_speaker_recognizer(wav_a, wav_b, threshold=0.123)
    assert recognizer.threshold == 0.123
    path = recognizer.save(tmp_path / "book")
    assert VoiceRecognizer(Config(), book=SpeakerBook.load(path)).threshold == encoder.spec.default_threshold


def test_asnorm_needs_its_own_threshold(encoder, wav_a, wav_b):
    from voicefingerprint.exceptions import CalibrationError

    recognizer = _two_speaker_recognizer(wav_a, wav_b)
    recognizer.calibrate()
    cosine_threshold = recognizer.threshold

    recognizer.book.set_cohort(_cohort_embeddings(encoder))
    assert recognizer.score_type == "asnorm"
    with pytest.raises(CalibrationError, match="no AS-norm threshold"):
        recognizer.identify(wav_a[: 3 * SR], source_sr=SR)

    report = recognizer.calibrate()
    assert report["score_type"] == "asnorm"
    assert recognizer.book.thresholds["cosine"] == cosine_threshold  # untouched
    assert recognizer.threshold == report["threshold"]

    matches = recognizer.identify(wav_a[: 3 * SR], source_sr=SR)
    assert matches[0].name == "a"
    assert recognizer.verify(wav_a[: 2 * SR], wav_a[2 * SR :], source_sr=SR).normalized

    recognizer.clear_cohort()
    assert recognizer.score_type == "cosine" and recognizer.threshold == cosine_threshold


def test_identify_with_asnorm_matches_pairwise_scoring(encoder, wav_a, wav_b):
    recognizer = _two_speaker_recognizer(wav_a, wav_b, threshold=0.0)
    recognizer.book.set_cohort(_cohort_embeddings(encoder))
    probe = wav_a[: 3 * SR]
    matches = {m.name: m.score for m in recognizer.identify(probe, source_sr=SR)}
    embedding = recognizer.embed(probe, source_sr=SR)
    for name in ("a", "b"):
        expected = recognizer.asnorm(recognizer.book[name].centroid, embedding)
        assert matches[name] == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_calibrate_needs_repeat_clips(encoder, wav_a, wav_b):
    from voicefingerprint.exceptions import CalibrationError

    recognizer = VoiceRecognizer(Config())
    recognizer.enroll("a", [wav_a], source_sr=SR)
    recognizer.enroll("b", [wav_b], source_sr=SR)
    with pytest.raises(CalibrationError, match="two or more clips"):
        recognizer.calibrate()
