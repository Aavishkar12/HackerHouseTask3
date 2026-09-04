#!/usr/bin/env python3
"""
build_gallery.py — build a multi-photo reference gallery for one person.

A single reference encoding is not reliable enough (see README: measured
same-person distances overlap impostor distances when only one reference
is used). This builds a gallery from several photos and saves it for
Stage 2 to match search results against.

Usage
-----
    # use every image in data/sample_images/refs/
    python scripts/build_gallery.py

    # or name the photos explicitly
    python scripts/build_gallery.py photo1.jpg photo2.jpg photo3.jpg

Output
------
    data/output/gallery.json

It also runs a leave-one-out check: each reference is held out and
matched against the rest, so you can see the worst-case true-match
distance for your own photos before trusting the threshold.
"""

import sys
from pathlib import Path

import numpy as np  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from faceid.face_encode import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TOLERANCE,
    FaceGallery,
    cosine_distance,
)

REFS_DIR = REPO_ROOT / "data" / "sample_images" / "refs"
OUT_PATH = REPO_ROOT / "data" / "output" / "gallery.json"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        if not REFS_DIR.exists():
            print(f"[!] No reference folder at {REFS_DIR}")
            print("    Put several photos of the same person there, or pass "
                  "paths as arguments.")
            return 1
        paths = sorted(p for p in REFS_DIR.iterdir() if p.suffix.lower() in EXTS)

    if not paths:
        print(f"[!] No images found in {REFS_DIR}")
        return 1

    print(f"[*] Building gallery from {len(paths)} photo(s)...")
    gallery = FaceGallery.from_paths(paths)

    if len(gallery) == 0:
        print("[!] No usable faces found — gallery is empty.")
        return 1
    if len(gallery) < 3:
        print(f"[!] Warning: only {len(gallery)} reference(s). Three or more "
              "is strongly recommended; fewer leaves the match threshold "
              "poorly constrained.")

    gallery.save(OUT_PATH)
    print(f"[+] Gallery with {len(gallery)} reference(s) -> {OUT_PATH}")
    for label in gallery.labels:
        print(f"      - {label}")

    # leave-one-out sanity check
    print("\n[*] Leave-one-out check (each photo vs the rest):")
    worst = 0.0
    for i, label in enumerate(gallery.labels):
        others = [e for j, e in enumerate(gallery.encodings) if j != i]
        if not others:
            continue
        # MUST be cosine — ArcFace embeddings are not calibrated for
        # Euclidean distance, and mixing the two silently produces
        # meaningless numbers.
        d = min(cosine_distance(gallery.encodings[i], o) for o in others)
        worst = max(worst, d)
        flag = "" if d <= DEFAULT_TOLERANCE else "   <-- ABOVE TOLERANCE"
        print(f"      {label:<20s} closest match: {d:.3f}{flag}")

    print(f"\n    worst true-match distance: {worst:.3f} "
          f"(tolerance {DEFAULT_TOLERANCE})")
    if worst > DEFAULT_TOLERANCE:
        print("    [!] At least one of your own photos would NOT match the "
              "gallery. Either that photo is an outlier (odd angle/lighting) "
              "or the tolerance is too strict.")
    else:
        headroom = DEFAULT_TOLERANCE - worst
        print(f"    headroom before a real match would be missed: {headroom:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
