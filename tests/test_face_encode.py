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


def test_save_and_load_encoding_roundtrip(tmp_path):
    fake_encoding = np.random.rand(128).astype(np.float64)
    out_path = tmp_path / "enc.json"

    save_encoding(fake_encoding, out_path)
    loaded = load_encoding(out_path)

    assert loaded.shape == (128,)
    np.testing.assert_allclose(loaded, fake_encoding)
