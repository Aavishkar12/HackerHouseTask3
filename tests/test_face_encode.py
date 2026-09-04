"""
Lightweight automated tests for faceid.face_encode (DeepFace / ArcFace).

These only exercise code paths that don't require a real human face
(missing file, no-face-found, distance maths, gallery logic). A genuine
"does it correctly encode a real face" check needs actual photos — that's
what scripts/verify_stage1.py and scripts/build_gallery.py are for.

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
    DEFAULT_MODEL,
    DEFAULT_TOLERANCE,
    EMBEDDING_DIM,
    FaceGallery,
    NoFaceDetectedError,
    cosine_distance,
    encode_face_from_path,
    find_best_match,
    load_encoding,
    save_encoding,
)


# --- basic error handling ---------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        encode_face_from_path(tmp_path / "does_not_exist.jpg")


def test_blank_image_raises_no_face(tmp_path):
    blank_path = tmp_path / "blank.jpg"
    Image.new("RGB", (400, 400), color=(255, 255, 255)).save(blank_path)
    with pytest.raises(NoFaceDetectedError):
        encode_face_from_path(blank_path)


# --- distance metric --------------------------------------------------

def test_cosine_distance_identical_is_zero():
    v = np.random.rand(EMBEDDING_DIM)
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-12)


def test_cosine_distance_orthogonal_is_one():
    a = np.zeros(EMBEDDING_DIM); a[0] = 1.0
    b = np.zeros(EMBEDDING_DIM); b[1] = 1.0
    assert cosine_distance(a, b) == pytest.approx(1.0)


def test_cosine_distance_is_scale_invariant():
    """Cosine distance must ignore vector magnitude — only direction."""
    a = np.random.rand(EMBEDDING_DIM)
    assert cosine_distance(a, a * 7.5) == pytest.approx(0.0, abs=1e-12)


def test_cosine_distance_zero_vector_does_not_crash():
    """A zero vector has no direction; must not raise ZeroDivisionError."""
    a = np.zeros(EMBEDDING_DIM)
    b = np.random.rand(EMBEDDING_DIM)
    assert cosine_distance(a, b) == float("inf")


# --- matching logic ---------------------------------------------------

def _unit(*components):
    """Build a unit-ish vector from the first few components."""
    v = np.zeros(EMBEDDING_DIM)
    for i, c in enumerate(components):
        v[i] = c
    return v


def test_find_best_match_picks_single_closest():
    """
    Regression test for a real false positive found during Stage 1
    testing: in a two-person group photo, BOTH faces scored under the
    original tolerance. find_best_match must return exactly one index —
    the closest — never "everything under threshold".
    """
    ref = _unit(1.0, 0.0)
    near = _unit(1.0, 0.30)   # the real person, different photo
    far = _unit(1.0, 1.10)    # a different person

    idx, dist, is_match = find_best_match(ref, [far, near])
    assert idx == 1, "should pick the nearer candidate, not the first"
    assert is_match is True
    assert dist < cosine_distance(ref, far)


def test_find_best_match_empty_candidates():
    idx, dist, is_match = find_best_match(np.random.rand(EMBEDDING_DIM), [])
    assert idx is None
    assert is_match is False


def test_tolerance_is_stricter_than_deepface_default():
    from faceid.face_encode import DEEPFACE_DEFAULT_TOLERANCE
    assert DEFAULT_TOLERANCE < DEEPFACE_DEFAULT_TOLERANCE


# --- gallery ----------------------------------------------------------

def test_gallery_uses_minimum_distance():
    """
    A gallery must match on the CLOSEST reference, not the first or an
    average. This is what widened the true/impostor margin in testing.
    """
    ref_bad = _unit(1.0, 1.2)    # a bad-angle reference
    ref_good = _unit(1.0, 0.05)  # a good reference
    gallery = FaceGallery(labels=["bad_angle", "good"],
                          encodings=[ref_bad, ref_good])

    candidate = _unit(1.0, 0.0)
    is_match, dist, via = gallery.match(candidate)
    assert is_match is True
    assert via == "good", "must report which reference matched"
    assert dist == pytest.approx(cosine_distance(candidate, ref_good))


def test_empty_gallery_never_matches():
    is_match, dist, via = FaceGallery(labels=[], encodings=[]).match(
        np.random.rand(EMBEDDING_DIM))
    assert is_match is False
    assert via is None


def test_gallery_save_load_roundtrip(tmp_path):
    g = FaceGallery(labels=["a", "b"],
                    encodings=[np.random.rand(EMBEDDING_DIM),
                               np.random.rand(EMBEDDING_DIM)])
    p = tmp_path / "g.json"
    g.save(p)
    loaded = FaceGallery.load(p)

    assert loaded.labels == g.labels
    assert len(loaded) == 2
    for a, b in zip(loaded.encodings, g.encodings):
        np.testing.assert_allclose(a, b)


def test_gallery_rejects_mismatched_model(tmp_path):
    """
    Embeddings from different models are not comparable. Loading a
    gallery built with another model must fail loudly rather than
    silently producing nonsense distances.
    """
    import json
    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "model": "SomeOtherModel",
        "metric": "cosine",
        "labels": ["a"],
        "encodings": [np.random.rand(128).tolist()],
    }))
    with pytest.raises(ValueError, match="SomeOtherModel"):
        FaceGallery.load(p)


# --- persistence ------------------------------------------------------

def test_save_and_load_encoding_roundtrip(tmp_path):
    fake = np.random.rand(EMBEDDING_DIM).astype(np.float64)
    out_path = tmp_path / "enc.json"

    save_encoding(fake, out_path)
    loaded = load_encoding(out_path)

    assert loaded.shape == (EMBEDDING_DIM,)
    np.testing.assert_allclose(loaded, fake)
