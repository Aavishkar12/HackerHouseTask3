#!/usr/bin/env python3
"""
scan_face.py — capture a face scan LIVE from a webcam.

This is the "face scan input" step of the pipeline, and it's what the
screen recording should show: a live camera feed, not a file picker.
Encoding logic is unchanged — this is a thin capture layer in front of
the already-tested faceid.face_encode module.

Controls (a window opens showing the live feed):
    SPACE  - capture the current frame and encode it
    ESC    - quit without capturing

What it does on capture:
    1. Saves the raw frame to data/output/scan_<timestamp>.jpg (this is
       the artifact Stage 2 hands to the web search step).
    2. Runs face detection + encoding immediately, so you see right away
       whether it worked, exactly like scripts/demo_encode.py does for a
       file.
    3. If data/output/gallery.json exists, also matches the captured
       face against it and reports the distance — a live self-check
       that "yes, this is really you" before the recording moves on.

Usage:
    python scripts/scan_face.py
    python scripts/scan_face.py --camera 1      # if you have >1 camera
    python scripts/scan_face.py --countdown 3    # seconds before auto-capture

IMPORTANT: this needs an actual display and a real webcam attached to
the machine it runs on — it will not work over SSH, in this cloud
sandbox, or in any headless environment. Run it locally on your own
computer, in a normal terminal, with the webcam connected.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from faceid.face_encode import (  # noqa: E402
    DEFAULT_TOLERANCE,
    FaceGallery,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    encode_face_from_array,
    save_encoding,
)

OUTPUT_DIR = REPO_ROOT / "data" / "output"
GALLERY_PATH = OUTPUT_DIR / "gallery.json"

WINDOW_TITLE = "Face Scan - SPACE to capture, ESC to quit"


def try_import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        print("[!] opencv-python is required for the live camera feed.")
        print("    It's already in requirements.txt, but if you're missing it:")
        print("        pip install opencv-python")
        sys.exit(1)


def process_captured_frame(frame, cv2, gallery_path=None):
    """Save, encode, and (if a gallery exists) match the captured frame."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_path = OUTPUT_DIR / f"scan_{timestamp}.jpg"
    cv2.imwrite(str(scan_path), frame)
    print(f"\n[*] Saved raw scan -> {scan_path}")

    try:
        embedding = encode_face_from_array(frame)
    except NoFaceDetectedError:
        print("[!] No face detected in the captured frame.")
        print("    Try again with better lighting, facing the camera directly.")
        return False
    except MultipleFacesDetectedError as e:
        print(f"[!] {e}")
        print("    Make sure only one person is in frame.")
        return False

    print(f"[+] Face encoded: shape={embedding.shape}, dtype={embedding.dtype}")

    emb_path = OUTPUT_DIR / f"scan_{timestamp}_encoding.json"
    save_encoding(embedding, emb_path)
    print(f"[+] Saved embedding -> {emb_path}")

    gallery_path = Path(gallery_path) if gallery_path else GALLERY_PATH
    if gallery_path.exists():
        gallery = FaceGallery.load(gallery_path)
        is_match, distance, via = gallery.match(embedding)
        verdict = "MATCH" if is_match else "NO MATCH"
        print(f"\n[*] Checked against your {len(gallery)}-photo gallery:")
        print(f"    closest reference: {via}")
        print(f"    distance: {distance:.4f}  (tolerance {DEFAULT_TOLERANCE})")
        print(f"    verdict: {verdict}")
        if not is_match:
            print("    (this scan would NOT be treated as you by Stage 2/3 — "
                  "recapture with better lighting/angle)")
    else:
        print(f"\n[i] No gallery found at {gallery_path} — skipping match check.")
        print("    Run scripts/build_gallery.py first if you want live "
              "self-verification during the scan.")

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index (default 0, the default webcam)")
    parser.add_argument("--gallery", default=str(GALLERY_PATH),
                        help="gallery to self-check the capture against")
    parser.add_argument("--countdown", type=int, default=0,
                        help="auto-capture after N seconds instead of "
                             "waiting for SPACE (useful for a hands-free "
                             "recording)")
    args = parser.parse_args()

    cv2 = try_import_cv2()

    print(f"[*] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Could not open camera {args.camera}.")
        print("    Check that a webcam is connected and not in use by "
              "another application (close Zoom/Teams/Camera app), or try "
              "--camera 1.")
        sys.exit(1)

    print("[*] Camera open. Press SPACE to capture, ESC to quit.")
    captured = False
    countdown_start = time.time() if args.countdown > 0 else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[!] Failed to read a frame from the camera.")
                break

            display = frame.copy()
            if countdown_start is not None:
                remaining = args.countdown - (time.time() - countdown_start)
                if remaining <= 0:
                    process_captured_frame(frame, cv2, args.gallery)
                    captured = True
                    break
                cv2.putText(display, f"capturing in {remaining:.1f}s",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 0), 2)
            else:
                cv2.putText(display, "SPACE = capture   ESC = quit",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                           (0, 255, 0), 2)

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[*] Quit without capturing.")
                break
            if key == 32 and countdown_start is None:  # SPACE
                process_captured_frame(frame, cv2, args.gallery)
                captured = True
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    sys.exit(0 if captured else 1)


if __name__ == "__main__":
    main()
