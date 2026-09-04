"""
Lightweight automated tests for faceid.face_encode.

These only exercise the code paths that don't require a real human face
(missing file, no-face-found), since a genuine "does it correctly encode
a real face" check needs an actual photo — that's what
scripts/demo_encode.py is for (run manually against your own selfie).

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from faceid.face_encode import (  # noqa: E402
    NoFaceDetectedError,
    encode_face_from_path,
    load_encoding,
    save_encoding,
)


def test_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    try:
        encode_face_from_path(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_blank_image_raises_no_face(tmp_path):
    # A plain white square has no face in it -> should raise cleanly
    # instead of crashing.
    blank_path = tmp_path / "blank.jpg"
    Image.new("RGB", (400, 400), color=(255, 255, 255)).save(blank_path)

    try:
        encode_face_from_path(blank_path)
        assert False, "expected NoFaceDetectedError"
    except NoFaceDetectedError:
        pass


def test_find_best_match_picks_single_closest():
    """
    Regression test for a real false positive found during Stage 1 testing:
    in a two-person group photo, BOTH faces scored under the old 0.6
    tolerance against the reference selfie. find_best_match must return
    exactly one index (the closest), never "everything under threshold".
    """
    from faceid.face_encode import find_best_match

    ref = np.zeros(128)
    near = np.zeros(128)
    near[0] = 0.41           # the real person, different photo
    far = np.zeros(128)
    far[0] = 0.58            # a different person who still sneaks under 0.6

    idx, dist, is_match = find_best_match(ref, [far, near])
    assert idx == 1, "should pick the nearer candidate, not the first one"
    assert is_match is True
    assert dist == pytest.approx(0.41)

    # and the impostor alone must be rejected at the project default (0.5)
    idx, dist, is_match = find_best_match(ref, [far])
    assert idx == 0
    assert is_match is False, "0.58 must not count as a match at tolerance 0.5"


def test_find_best_match_empty_candidates():
    from faceid.face_encode import find_best_match

    idx, dist, is_match = find_best_match(np.zeros(128), [])
    assert idx is None
    assert is_match is False


def test_default_tolerance_is_stricter_than_library_default():
    from faceid.face_encode import DEFAULT_TOLERANCE

    assert DEFAULT_TOLERANCE < 0.6


def test_save_and_load_encoding_roundtrip(tmp_path):
    fake_encoding = np.random.rand(128).astype(np.float64)
    out_path = tmp_path / "enc.json"

    save_encoding(fake_encoding, out_path)
    loaded = load_encoding(out_path)

    assert loaded.shape == (128,)
    np.testing.assert_allclose(loaded, fake_encoding)
