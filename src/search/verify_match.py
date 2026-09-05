"""
verify_match.py — confirm a search hit is actually the same person.

A reverse image search returns pages where a *visually similar* image
appears. That is not the same as "this page shows the person we scanned":
search engines return crops, re-uploads, look-alikes and outright wrong
matches.

This module closes that gap by pulling the candidate image and running it
back through Stage 1's face pipeline, comparing it against the reference
gallery. The result is a real measurement — a cosine distance — rather
than a claim, and the pipeline can then report *why* it believes a match
is genuine.

That matters here specifically: Stage 3 writes the discovered post to a
blockchain permanently. A wrong match is not recoverable, so the bar for
"this is really them" should be evidence, not the search engine's word.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Stage 1 — reused as-is, no reimplementation.
from faceid.face_encode import (
    DEFAULT_TOLERANCE,
    FaceEncodingError,
    FaceGallery,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    encode_face_from_array,
    find_best_match,
)

DEFAULT_HTTP_TIMEOUT = 20
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # don't pull down huge files

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class FaceVerification:
    """Outcome of re-checking a candidate image against the gallery."""

    attempted: bool = False          # did we manage to fetch + decode an image?
    verified: Optional[bool] = None  # True/False if measured, None if not attempted
    distance: Optional[float] = None
    matched_reference: Optional[str] = None
    tolerance: float = DEFAULT_TOLERANCE
    faces_found: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        if not self.attempted:
            return f"not verified ({self.note})"
        if self.verified is None:
            return f"inconclusive ({self.note})"
        if self.distance is None:
            # A verdict was reached without measuring a distance — in
            # practice this means no face was found at all, which is not
            # the same claim as "this is a different person".
            return f"no match ({self.note})"
        verdict = "SAME PERSON" if self.verified else "different person"
        return f"{verdict} (distance {self.distance:.4f}, tol {self.tolerance})"


def _decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    """
    Decode image bytes to a BGR numpy array (what DeepFace expects).

    Uses PIL rather than cv2.imdecode because PIL handles a wider range
    of web formats (WebP, progressive JPEG, palettes) which is what
    social media CDNs actually serve.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(io.BytesIO(data)) as im:
            rgb = np.array(im.convert("RGB"))
    except Exception:
        return None

    # PIL gives RGB; DeepFace expects BGR (OpenCV convention).
    return rgb[:, :, ::-1].copy()


def fetch_image(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Optional[bytes]:
    """
    Download an image. Returns raw bytes, or None on any failure.

    Never raises — a candidate we can't fetch is simply unverifiable, not
    a crash. Many social platforms block hotlinking, so failures here are
    expected and normal.
    """
    try:
        import requests
    except ImportError:
        return None

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get("Content-Type", "")
        if content_type and not content_type.lower().startswith("image"):
            return None

        data = b""
        for chunk in resp.iter_content(64 * 1024):
            data += chunk
            if len(data) > MAX_IMAGE_BYTES:
                return None
        return data or None
    except Exception:
        return None


def verify_image_against_gallery(
    image_url: str,
    gallery: FaceGallery,
    tolerance: float = DEFAULT_TOLERANCE,
    timeout: int = DEFAULT_HTTP_TIMEOUT,
) -> FaceVerification:
    """
    Fetch `image_url`, find faces in it, and compare against the gallery.

    For an image containing several faces, the CLOSEST face is used — the
    same rule Stage 1 established. A group photo can legitimately contain
    the person plus strangers; requiring every face to match would be
    wrong, and accepting any face under the threshold would risk matching
    on a stranger.

    Never raises: every failure mode becomes a FaceVerification with
    attempted/verified set appropriately and a human-readable note.
    """
    if not image_url:
        return FaceVerification(note="no image URL available")

    data = fetch_image(image_url, timeout=timeout)
    if data is None:
        return FaceVerification(
            note="could not download the image (hotlink-blocked, "
                 "removed, or not an image)"
        )

    frame = _decode_image_bytes(data)
    if frame is None:
        return FaceVerification(note="downloaded file was not a decodable image")

    try:
        encodings = encode_face_from_array(frame, allow_multiple=True)
    except NoFaceDetectedError:
        return FaceVerification(
            attempted=True, verified=False, faces_found=0,
            note="no face found in the candidate image",
        )
    except (MultipleFacesDetectedError, FaceEncodingError) as e:
        return FaceVerification(attempted=True, note=f"encoding failed: {e}")
    except Exception as e:  # decoding/model errors shouldn't kill the pipeline
        return FaceVerification(attempted=True, note=f"unexpected error: {e}")

    if not isinstance(encodings, list):
        encodings = [encodings]
    if not encodings:
        return FaceVerification(
            attempted=True, verified=False, faces_found=0,
            note="no face found in the candidate image",
        )

    # Compare every face in the candidate against every gallery reference,
    # keeping the single best (smallest) distance.
    best_distance = float("inf")
    best_ref: Optional[str] = None
    for enc in encodings:
        idx, dist, _ = find_best_match(enc, gallery.encodings, tolerance)
        if idx is not None and dist < best_distance:
            best_distance = dist
            best_ref = gallery.labels[idx]

    if best_ref is None:
        return FaceVerification(
            attempted=True, faces_found=len(encodings),
            note="gallery was empty — nothing to compare against",
        )

    return FaceVerification(
        attempted=True,
        verified=best_distance <= tolerance,
        distance=round(float(best_distance), 6),
        matched_reference=best_ref,
        tolerance=tolerance,
        faces_found=len(encodings),
        note=f"compared {len(encodings)} face(s) against "
             f"{len(gallery)} reference(s)",
    )
