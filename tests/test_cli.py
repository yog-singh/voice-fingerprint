"""End-to-end CLI tests. Need the encoder weights; skipped when they are unavailable."""

import json

import numpy as np
import pytest

from conftest import _voice_like
from voicefingerprint.cli import main

SR = 16000


@pytest.fixture
def clips(tmp_path, encoder):
    sf = pytest.importorskip("soundfile")
    paths = {}
    for name, f0, seed in (("a", 110.0, 1), ("b", 210.0, 2)):
        wav = _voice_like(4.0, f0, seed)
        for i, part in enumerate((wav[: 2 * SR], wav[2 * SR :])):
            paths[f"{name}{i}"] = str(tmp_path / f"{name}{i}.wav")
            sf.write(paths[f"{name}{i}"], part, SR)
    for i in range(6):
        paths[f"bg{i}"] = str(tmp_path / f"bg{i}.wav")
        sf.write(paths[f"bg{i}"], _voice_like(3.0, 90.0 + 25 * i, seed=100 + i), SR)
    return paths


def _run(capsys, *argv):
    code = main(["--vad", "none", *argv])
    out = capsys.readouterr()
    return code, (json.loads(out.out) if out.out.strip() else None), out.err


def test_calibrate_persists_and_later_commands_use_it(tmp_path, clips, capsys):
    book = str(tmp_path / "speakers")
    _run(capsys, "enroll", "--book", book, "a", clips["a0"], clips["a1"])
    _run(capsys, "enroll", "--book", book, "b", clips["b0"], clips["b1"])

    code, report, _ = _run(capsys, "calibrate", "--book", book)
    assert code == 0 and report["score_type"] == "cosine" and report["book"].endswith(".npz")

    _, listing, _ = _run(capsys, "list", "--book", book)
    assert listing["thresholds"]["cosine"] == pytest.approx(report["threshold"])

    _, identified, _ = _run(capsys, "identify", "--book", book, clips["a0"])
    assert identified["threshold"] == pytest.approx(report["threshold"])
    assert identified["score_type"] == "cosine"

    _, verified, _ = _run(capsys, "verify", "--book", book, clips["a0"], clips["a1"])
    assert verified["threshold"] == pytest.approx(report["threshold"])

    _, overridden, _ = _run(capsys, "identify", "--book", book, "--threshold", "0.9", clips["a0"])
    assert overridden["threshold"] == 0.9
    _, listing, _ = _run(capsys, "list", "--book", book)
    assert listing["thresholds"]["cosine"] == pytest.approx(report["threshold"])  # override not saved


def test_cohort_switches_units_and_requires_recalibration(tmp_path, clips, capsys):
    book = str(tmp_path / "speakers")
    _run(capsys, "enroll", "--book", book, "a", clips["a0"], clips["a1"])
    _run(capsys, "enroll", "--book", book, "b", clips["b0"], clips["b1"])
    _, cosine_report, _ = _run(capsys, "calibrate", "--book", book)

    code, result, _ = _run(capsys, "cohort", "--book", book, *[clips[f"bg{i}"] for i in range(6)])
    assert code == 0 and result["score_type"] == "asnorm" and result["cohort_size"] == 6
    assert result["next"]

    code, out, err = _run(capsys, "identify", "--book", book, clips["a0"])
    assert code == 2 and out is None and "no AS-norm threshold" in err

    _, asnorm_report, _ = _run(capsys, "calibrate", "--book", book)
    assert asnorm_report["score_type"] == "asnorm"
    _, identified, _ = _run(capsys, "identify", "--book", book, clips["a0"])
    assert identified["score_type"] == "asnorm"
    assert identified["threshold"] == pytest.approx(asnorm_report["threshold"])

    _, cleared, _ = _run(capsys, "cohort", "--book", book, "--clear")
    assert cleared["score_type"] == "cosine" and set(cleared["thresholds"]) == {"cosine"}
    _, identified, _ = _run(capsys, "identify", "--book", book, clips["a0"])
    assert identified["threshold"] == pytest.approx(cosine_report["threshold"])


def test_cohort_needs_audio_or_clear(tmp_path, clips, capsys):
    code, _, err = _run(capsys, "cohort", "--book", str(tmp_path / "speakers"))
    assert code == 2 and "--clear" in err
