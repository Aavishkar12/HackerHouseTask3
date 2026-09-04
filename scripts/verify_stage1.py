#!/usr/bin/env python3
"""
verify_stage1.py — full self-check for Stage 1 (face detection & encoding).

Goes beyond `demo_encode.py`: it proves the encoder actually works rather
than just exiting 0. Useful for the screen recording, since it shows the
whole stage exercised in one run.

Checks performed against your sample photo:
  1. Face detected, and where (draws a bounding box you can eyeball)
  2. Encoding shape / dtype / value range
  3. Determinism — re-encoding the same photo gives distance ~0
  4. JSON save/load round-trip is lossless
  5. Robustness — a downscaled, re-compressed copy still matches (same person)
  6. Two faces in one image  -> MultipleFacesDetectedError (and
     allow_multiple=True returns both)
  7. Background-only crop    -> NoFaceDetectedError
  8. Missing file            -> FileNotFoundError

Usage:
    python scripts/verify_stage1.py [path/to/photo.jpg]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from faceid.face_encode import (  # noqa: E402
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    compare_encodings,
    detect_faces,
    encode_face_from_path,
    load_encoding,
    save_encoding,
)

DEFAULT_IMAGE = REPO_ROOT / "data" / "sample_images" / "me.jpg"
OUTPUT_DIR = REPO_ROOT / "data" / "output"
TMP_DIR = OUTPUT_DIR / "_verify_tmp"


def ok(label, passed, detail=""):
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not image_path.exists():
        print(f"[!] Image not found: {image_path}")
        print("    Put a photo at data/sample_images/me.jpg or pass a path.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"\nVerifying Stage 1 against: {image_path}\n")

    # 1. detection + bounding box
    print("1) Detection")
    matches = detect_faces(image_path)
    results.append(ok("exactly one face found", len(matches) == 1,
                      f"{len(matches)} face(s)"))
    if len(matches) != 1:
        print("\n[!] Need exactly one face to continue verification.")
        return 1

    top, right, bottom, left = matches[0].location
    print(f"       bbox (top,right,bottom,left) = {matches[0].location} "
          f"-> {right - left}x{bottom - top}px")
    annotated = Image.open(image_path).convert("RGB")
    ImageDraw.Draw(annotated).rectangle(
        [left, top, right, bottom], outline=(0, 255, 0), width=8)
    annotated_path = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"
    annotated.save(annotated_path, quality=90)
    print(f"       annotated image -> {annotated_path}")

    # 2. encoding shape / stats
    print("\n2) Encoding")
    enc = encode_face_from_path(image_path)
    results.append(ok("shape (128,) float64",
                      enc.shape == (128,) and enc.dtype == np.float64,
                      f"shape={enc.shape} dtype={enc.dtype}"))
    print(f"       norm={np.linalg.norm(enc):.4f} min={enc.min():.4f} "
          f"max={enc.max():.4f} mean={enc.mean():.4f}")

    # 3. determinism
    print("\n3) Determinism")
    _, dist = compare_encodings(enc, encode_face_from_path(image_path))
    results.append(ok("re-encoding gives distance ~0", dist < 1e-9,
                      f"distance={dist:.9f}"))

    # 4. save/load round-trip
    print("\n4) Persistence")
    enc_path = OUTPUT_DIR / f"{image_path.stem}_encoding.json"
    save_encoding(enc, enc_path)
    reloaded = load_encoding(enc_path)
    results.append(ok("JSON round-trip lossless", np.allclose(reloaded, enc),
                      f"{enc_path}"))

    # 5. robustness to resize/recompression
    print("\n5) Robustness (same person, degraded image)")
    src = Image.open(image_path).convert("RGB")
    small_path = TMP_DIR / "small.jpg"
    src.resize((src.width // 3, src.height // 3)).save(small_path, quality=70)
    is_match, dist = compare_encodings(enc, encode_face_from_path(small_path))
    results.append(ok("3x downscaled + recompressed still matches", is_match,
                      f"distance={dist:.4f} (tolerance 0.6)"))

    # 6. multiple faces
    print("\n6) Error handling — multiple faces")
    w, h = src.size
    two = Image.new("RGB", (w * 2, h))
    two.paste(src, (0, 0))
    two.paste(src, (w, 0))
    two_path = TMP_DIR / "two_faces.jpg"
    two.save(two_path, quality=90)
    try:
        encode_face_from_path(two_path)
        results.append(ok("raises MultipleFacesDetectedError", False,
                          "no error raised"))
    except MultipleFacesDetectedError as e:
        results.append(ok("raises MultipleFacesDetectedError", True,
                          f"num_faces={e.num_faces}"))
    encs = encode_face_from_path(two_path, allow_multiple=True)
    results.append(ok("allow_multiple=True returns both", len(encs) == 2,
                      f"{len(encs)} encodings"))

    # 7. no face
    print("\n7) Error handling — no face")
    no_face_path = TMP_DIR / "no_face.jpg"
    src.crop((0, 0, w, max(50, top // 3))).save(no_face_path, quality=90)
    try:
        encode_face_from_path(no_face_path)
        results.append(ok("raises NoFaceDetectedError", False, "no error raised"))
    except NoFaceDetectedError:
        results.append(ok("raises NoFaceDetectedError", True))

    # 8. missing file
    print("\n8) Error handling — missing file")
    try:
        encode_face_from_path(TMP_DIR / "definitely_missing.jpg")
        results.append(ok("raises FileNotFoundError", False, "no error raised"))
    except FileNotFoundError:
        results.append(ok("raises FileNotFoundError", True))

    # cleanup temp files
    for p in TMP_DIR.glob("*"):
        p.unlink()
    TMP_DIR.rmdir()

    passed, total = sum(results), len(results)
    print(f"\n{'=' * 52}")
    print(f"  Stage 1 verification: {passed}/{total} checks passed")
    print(f"{'=' * 52}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
