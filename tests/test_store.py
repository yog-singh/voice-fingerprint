import numpy as np
import pytest

from voicefingerprint.exceptions import EnrollmentError
from voicefingerprint.store import SpeakerBook

DIM = 32


def _book():
    book = SpeakerBook("test_model", DIM)
    rng = np.random.default_rng(0)
    book.add("alice", rng.standard_normal((3, DIM)), {"source": "mic"})
    book.add("bob", rng.standard_normal((2, DIM)))
    return book


def test_add_and_lookup():
    book = _book()
    assert book.names == ["alice", "bob"]
    assert len(book) == 2
    assert "alice" in book
    assert book["alice"].n_utterances == 3
    assert book["alice"].metadata["source"] == "mic"


def test_centroid_is_unit_length():
    assert np.linalg.norm(_book()["alice"].centroid) == pytest.approx(1.0, abs=1e-6)


def test_adding_again_appends():
    book = _book()
    book.add("alice", np.random.randn(2, DIM))
    assert book["alice"].n_utterances == 5


def test_unknown_speaker_raises():
    with pytest.raises(EnrollmentError, match="not enrolled"):
        _book()["carol"]


def test_dimension_mismatch_raises():
    with pytest.raises(EnrollmentError, match="expected 32-d"):
        _book().add("dave", np.random.randn(1, 64))


def test_roundtrip(tmp_path):
    book = _book()
    path = book.save(tmp_path / "speakers")
    loaded = SpeakerBook.load(path, expect_model="test_model")

    assert loaded.names == book.names
    assert loaded["alice"].metadata == {"source": "mic"}
    np.testing.assert_allclose(loaded["alice"].centroid, book["alice"].centroid, atol=1e-6)


def test_loading_with_the_wrong_model_is_refused(tmp_path):
    path = _book().save(tmp_path / "speakers")
    with pytest.raises(EnrollmentError, match="built with model"):
        SpeakerBook.load(path, expect_model="other_model")


def test_all_embeddings_labels_line_up():
    embeddings, labels = _book().all_embeddings()
    assert embeddings.shape == (5, DIM)
    assert labels == ["alice"] * 3 + ["bob"] * 2


def test_empty_book_shapes():
    book = SpeakerBook("test_model", DIM)
    names, centroids = book.centroids()
    assert names == [] and centroids.shape == (0, DIM)
    embeddings, labels = book.all_embeddings()
    assert embeddings.shape == (0, DIM) and labels == []


def test_thresholds_and_cohort_roundtrip(tmp_path):
    book = _book()
    book.set_cohort(np.random.default_rng(1).standard_normal((7, DIM)), top_k=5)
    book.set_threshold("cosine", 0.31)
    book.set_threshold("asnorm", 2.5)
    loaded = SpeakerBook.load(book.save(tmp_path / "speakers"))

    assert loaded.thresholds == {"cosine": 0.31, "asnorm": 2.5}
    assert loaded.cohort.shape == (7, DIM) and loaded.cohort_top_k == 5
    np.testing.assert_allclose(loaded.cohort, book.cohort, atol=1e-6)


def test_book_without_cohort_roundtrips_as_none(tmp_path):
    loaded = SpeakerBook.load(_book().save(tmp_path / "speakers"))
    assert loaded.cohort is None and loaded.thresholds == {}


def test_version_1_books_still_load(tmp_path):
    import json

    path = tmp_path / "old.npz"
    header = {"version": 1, "model": "test_model", "embedding_dim": DIM,
              "speakers": [{"name": "alice", "key": "emb_0", "metadata": {}}]}
    np.savez_compressed(path, header=np.array(json.dumps(header)), emb_0=np.ones((2, DIM), dtype=np.float32))
    loaded = SpeakerBook.load(path, expect_model="test_model")
    assert loaded.names == ["alice"] and loaded.thresholds == {} and loaded.cohort is None


def test_changing_the_cohort_drops_the_asnorm_threshold_only():
    rng = np.random.default_rng(2)
    book = _book()
    cohort = rng.standard_normal((5, DIM))
    book.set_cohort(cohort)
    book.set_threshold("cosine", 0.3)
    book.set_threshold("asnorm", 2.0)

    book.set_cohort(cohort.copy())  # same cohort: threshold still valid
    assert book.thresholds == {"cosine": 0.3, "asnorm": 2.0}

    book.set_cohort(rng.standard_normal((5, DIM)))
    assert book.thresholds == {"cosine": 0.3}

    book.set_threshold("asnorm", 2.0)
    book.set_cohort(None)
    assert book.thresholds == {"cosine": 0.3} and book.cohort is None


def test_unknown_score_type_is_rejected():
    with pytest.raises(ValueError, match="unknown score type"):
        _book().set_threshold("plda", 1.0)
