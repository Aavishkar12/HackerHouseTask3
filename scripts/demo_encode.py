#!/usr/bin/env python3
"""
demo_encode.py — Task 1 demo/verification script.

Loads a sample photo, runs it through faceid.face_encode, and prints +
saves the resulting 128-d face encoding so you can confirm the pipeline
works before moving on to Task 2 (web/social search).

Usage
-----
    # uses the default path: data/sample_images/me.jpg
    python scripts/demo_encode.py

    # or point it at any photo
    python scripts/demo_encode.py path/to/photo.jpg

Output
------
Prints a summary (first few dims of the encoding, its shape/dtype) to
stdout, and writes the full encoding as JSON to:
    data/output/<image_stem>_encoding.json
for later stages to load via faceid.face_encode.load_encoding(...).
"""

import sys
from pathlib import Path

# Make `import faceid...` work when running this script directly, without
# requiring the package to be pip-installed.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from faceid.face_encode import (  # noqa: E402
    FaceEncodingError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    encode_face_from_path,
    save_encoding,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "data" / "sample_images" / "me.jpg"
OUTPUT_DIR = REPO_ROOT / "data" / "output"


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE

    if not image_path.exists():
        print(f"[!] Image not found: {image_path}")
        if image_path == DEFAULT_IMAGE:
            print(
                "    Place a clear, front-facing photo of yourself at "
                f"{DEFAULT_IMAGE}\n"
                "    (or pass a path explicitly: "
                "python scripts/demo_encode.py path/to/photo.jpg)"
            )
        return 1

    print(f"[*] Encoding face(s) in: {image_path}")
    try:
        encoding = encode_face_from_path(image_path)
    except NoFaceDetectedError as e:
        print(f"[!] No face detected: {e}")
        return 1
    except MultipleFacesDetectedError as e:
        print(f"[!] Multiple faces detected: {e}")
        print(
            "    Tip: re-run with a single-person crop, or use "
            "encode_face_from_path(path, allow_multiple=True) in your "
            "own code to get encodings for every face."
        )
        return 1
    except FaceEncodingError as e:
        print(f"[!] Face encoding failed: {e}")
        return 1

    print("[+] Success — got one face encoding.")
    print(f"    shape: {encoding.shape}, dtype: {encoding.dtype}")
    print(f"    first 8 values: {encoding[:8]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{image_path.stem}_encoding.json"
    save_encoding(encoding, out_path)
    print(f"[+] Saved full encoding to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
