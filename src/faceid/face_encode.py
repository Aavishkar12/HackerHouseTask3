"""
face_encode.py — Task 1 of the HH Goa 2026 pipeline: face detection & encoding.

Reusable, importable functions for turning a photo into a face embedding
using **DeepFace with the ArcFace model** (RetinaFace for detection).

Later pipeline stages (web/social search, blockchain verification) import
`encode_face_from_path` / `FaceGallery` from this module rather than
re-implementing detection or matching.

Why ArcFace (and not dlib/face_recognition)
-------------------------------------------
Both were benchmarked on the same 7 real photos (6 of one person across
varied lighting/pose/glasses, plus 1 impostor), leave-one-out:

    backend                separation ratio   normalised margin
    dlib/face_recognition       1.263               0.208
    DeepFace Facenet512         1.334               0.250
    DeepFace ArcFace            1.423               0.297   <- chosen

ArcFace gave ~43% more headroom between the worst true match and the
impostor, and was the only backend where matching still separated
correctly *without* a gallery. Speed was comparable (~3.7s vs ~4.0s per
image on CPU). ArcFace is also trained on more demographically diverse
data than dlib's model, which matters here — the false positive that
prompted this comparison was between two South Asian men.

Distance metric
---------------
ArcFace embeddings are compared with **cosine distance** (0 = identical
direction, 1 = orthogonal, 2 = opposite). This is NOT the Euclidean
distance dlib used, so thresholds from face_recognition tutorials do not
transfer. See DEFAULT_TOLERANCE.

Typical usage
-------------
    from faceid.face_encode import encode_face_from_path

    embedding = encode_face_from_path("data/sample_images/me.jpg")
    # numpy.ndarray of shape (512,), dtype float64

Error handling
--------------
  * No face found            -> NoFaceDetectedError
  * More than one face found -> MultipleFacesDetectedError
    (unless allow_multiple=True)

Both subclass FaceEncodingError.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

# Quieten TensorFlow's startup noise before DeepFace imports it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from deepface import DeepFace
except ImportError as exc:  # pragma: no cover - import-time guidance
    raise ImportError(
        "The 'deepface' package is not installed in this environment. Run:\n"
        "    pip install -r requirements.txt\n"
        "Note: DeepFace pulls in TensorFlow and downloads ~335MB of model "
        "weights to ~/.deepface on first use. See README.md."
    ) from exc


PathLike = Union[str, Path]

# ArcFace: 512-d embeddings, strong accuracy, good demographic robustness.
DEFAULT_MODEL = "ArcFace"

# RetinaFace is the most accurate detector DeepFace ships. "opencv" is much
# faster but misses angled/small faces; "mtcnn" sits in between.
DEFAULT_DETECTOR = "retinaface"

# Embedding dimensionality for DEFAULT_MODEL (sanity-check value).
EMBEDDING_DIM = 512

# DeepFace's own calibrated threshold for ArcFace + cosine is 0.68.
# We use 0.65 — slightly stricter — based on our own measurements:
#
#   worst true match (same person, harsh sunlight photo) .... 0.5384
#   impostor (different person) ............................. 0.7662
#
# 0.65 sits near the midpoint, leaving ~0.11 headroom on both sides.
# We bias strict deliberately: in Stage 2 a false positive means claiming
# a STRANGER's social media post belongs to the user, and Stage 3 then
# writes that claim to a blockchain permanently. Missing a real match is
# recoverable; a wrong match written on-chain is not.
DEFAULT_TOLERANCE = 0.65

# DeepFace's library default, for reference / override.
DEEPFACE_DEFAULT_TOLERANCE = 0.68


class FaceEncodingError(Exception):
    """Base class for all face-encoding errors raised by this module."""


class NoFaceDetectedError(FaceEncodingError):
    """Raised when zero faces are found in the given image."""

    def __init__(self, image_path: PathLike):
        super().__init__(
            f"No face detected in '{image_path}'. Make sure the photo "
            "clearly shows a face, is well lit, and isn't too "
            "low-resolution or heavily cropped."
        )
        self.image_path = str(image_path)


class MultipleFacesDetectedError(FaceEncodingError):
    """Raised when more than one face is found and allow_multiple=False."""

    def __init__(self, image_path: PathLike, num_faces: int):
        super().__init__(
            f"Found {num_faces} faces in '{image_path}', expected exactly "
            "one. Crop the photo to a single person, or call "
            "encode_face_from_path(..., allow_multiple=True) if you "
            "intentionally want embeddings for every face in the image."
        )
        self.image_path = str(image_path)
        self.num_faces = num_faces


@dataclass
class FaceMatch:
    """One detected face: its embedding plus where it was found."""

    encoding: np.ndarray  # shape (512,), dtype float64
    location: tuple  # (top, right, bottom, left) in pixels
    confidence: float = 0.0  # detector confidence, 0-1


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance between two embeddings: 0 = identical direction.

    This is the metric ArcFace embeddings are calibrated for. Euclidean
    distance on raw (un-normalised) ArcFace vectors is NOT equivalent and
    should not be substituted.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        # A zero vector has no direction; treat as maximally dissimilar
        # rather than raising a divide-by-zero.
        return float("inf")
    return float(1.0 - np.dot(a, b) / (na * nb))


def _area_to_location(area: dict) -> tuple:
    """DeepFace gives {x,y,w,h}; convert to (top, right, bottom, left)."""
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    return (y, x + w, y + h, x)


def detect_faces(
    image_path: PathLike,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = DEFAULT_DETECTOR,
) -> List[FaceMatch]:
    """
    Detect every face in an image and return an embedding + location for each.

    Low-level building block. Most callers should use
    `encode_face_from_path`, which adds the "exactly one face" validation
    that later pipeline stages rely on.

    Returns an empty list if no face is found (it does NOT raise) — that
    lets callers decide how to handle it.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: '{image_path}'")

    try:
        reps = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
        )
    except ValueError as e:
        # DeepFace signals "no face" by raising ValueError with a message
        # about face detection. Anything else is a real error.
        if "face could not be detected" in str(e).lower() or "detected" in str(e).lower():
            return []
        raise

    return [
        FaceMatch(
            encoding=np.asarray(r["embedding"], dtype=np.float64),
            location=_area_to_location(r["facial_area"]),
            confidence=float(r.get("face_confidence", 0.0)),
        )
        for r in reps
    ]


def encode_face_from_path(
    image_path: PathLike,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = DEFAULT_DETECTOR,
    allow_multiple: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Main entry point: encode the single face in an image.

    Returns a numpy array of shape (512,) when exactly one face is found.
    With allow_multiple=True, returns a list of such arrays instead.

    Raises:
        FileNotFoundError, NoFaceDetectedError, MultipleFacesDetectedError
    """
    matches = detect_faces(
        image_path, model_name=model_name, detector_backend=detector_backend
    )

    if len(matches) == 0:
        raise NoFaceDetectedError(image_path)
    if len(matches) > 1 and not allow_multiple:
        raise MultipleFacesDetectedError(image_path, len(matches))

    if allow_multiple:
        return [m.encoding for m in matches]
    return matches[0].encoding


def encode_face_from_array(
    image: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    detector_backend: str = DEFAULT_DETECTOR,
    allow_multiple: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Same as encode_face_from_path but takes an already-loaded image array
    (e.g. a frame from a webcam, or an image downloaded from a search
    result). Stage 2 uses this to avoid writing scraped images to disk.

    `image` should be an RGB or BGR numpy array.
    """
    try:
        reps = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
        )
    except ValueError as e:
        if "detected" in str(e).lower():
            raise NoFaceDetectedError("<in-memory image>") from e
        raise

    encodings = [np.asarray(r["embedding"], dtype=np.float64) for r in reps]

    if len(encodings) == 0:
        raise NoFaceDetectedError("<in-memory image>")
    if len(encodings) > 1 and not allow_multiple:
        raise MultipleFacesDetectedError("<in-memory image>", len(encodings))

    if allow_multiple:
        return encodings
    return encodings[0]


def compare_encodings(
    known_encoding: np.ndarray,
    candidate_encoding: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[bool, float]:
    """
    Compare two face embeddings using cosine distance.

    Returns (is_match, distance). Lower distance = more similar.
    See DEFAULT_TOLERANCE for why this project uses 0.65.
    """
    distance = cosine_distance(known_encoding, candidate_encoding)
    return distance <= tolerance, distance


def find_best_match(
    known_encoding: np.ndarray,
    candidates: Sequence[np.ndarray],
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[Optional[int], float, bool]:
    """
    Given one reference embedding and several candidates (e.g. every face
    in a group photo, or in an image from a search result), return the
    SINGLE closest candidate.

    Stage 2 must use this rather than "every face under the threshold":
    in a group photo more than one face can sit under the tolerance, and
    accepting all of them would mean claiming a stranger's post.

    Returns (index_of_closest, distance, is_match). index is None if
    `candidates` was empty.
    """
    if len(candidates) == 0:
        return None, float("inf"), False

    distances = [cosine_distance(known_encoding, c) for c in candidates]
    best_idx = int(np.argmin(distances))
    best_distance = distances[best_idx]
    return best_idx, best_distance, best_distance <= tolerance


@dataclass
class FaceGallery:
    """
    Several reference embeddings for ONE person.

    Even with ArcFace, a gallery materially improves matching. Measured
    leave-one-out on 6 photos of one person + 1 impostor (cosine):

        single reference : same-person up to 0.7435, impostor 0.7662
                           -> separable, but only just (0.023 apart)
        gallery (min)    : same-person up to 0.5384, impostor 0.7662
                           -> margin 0.2278, ~10x wider

    The candidate only has to resemble the person in ONE reference photo,
    which is what makes it robust to lighting and pose.
    """

    labels: List[str]
    encodings: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.encodings)

    def match(
        self, candidate: np.ndarray, tolerance: float = DEFAULT_TOLERANCE
    ) -> tuple[bool, float, Optional[str]]:
        """
        Compare a candidate face against every reference in the gallery,
        using the MINIMUM distance.

        Returns (is_match, best_distance, label_of_closest_reference).
        """
        if not self.encodings:
            return False, float("inf"), None
        idx, dist, is_match = find_best_match(candidate, self.encodings, tolerance)
        return is_match, dist, self.labels[idx] if idx is not None else None

    def save(self, out_path: PathLike) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": DEFAULT_MODEL,
            "metric": "cosine",
            "dim": EMBEDDING_DIM,
            "labels": self.labels,
            "encodings": [np.asarray(e).tolist() for e in self.encodings],
        }
        out_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, in_path: PathLike) -> "FaceGallery":
        data = json.loads(Path(in_path).read_text())
        stored_model = data.get("model")
        if stored_model and stored_model != DEFAULT_MODEL:
            raise ValueError(
                f"Gallery at '{in_path}' was built with model "
                f"'{stored_model}', but this module uses '{DEFAULT_MODEL}'. "
                "Embeddings from different models are not comparable — "
                "rebuild the gallery with scripts/build_gallery.py."
            )
        return cls(
            labels=data["labels"],
            encodings=[np.array(e, dtype=np.float64) for e in data["encodings"]],
        )

    @classmethod
    def from_paths(
        cls,
        image_paths: Sequence[PathLike],
        model_name: str = DEFAULT_MODEL,
        detector_backend: str = DEFAULT_DETECTOR,
        skip_failures: bool = True,
    ) -> "FaceGallery":
        """
        Build a gallery from several photos of the same person.

        Photos with no detectable face are skipped with a warning when
        skip_failures=True. For multi-face photos the largest face is
        used, assuming the subject is closest to the camera.
        """
        labels, encodings = [], []
        for p in image_paths:
            p = Path(p)
            try:
                matches = detect_faces(
                    p, model_name=model_name, detector_backend=detector_backend
                )
                if not matches:
                    raise NoFaceDetectedError(p)
                matches.sort(
                    key=lambda m: (m.location[2] - m.location[0]), reverse=True
                )
                labels.append(p.stem)
                encodings.append(matches[0].encoding)
            except (FaceEncodingError, FileNotFoundError) as e:
                if not skip_failures:
                    raise
                print(f"  [skip] {p.name}: {e}")
        return cls(labels=labels, encodings=encodings)


def save_encoding(encoding: np.ndarray, out_path: PathLike) -> None:
    """Persist an embedding to disk as JSON, for later pipeline stages."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(np.asarray(encoding).tolist()))


def load_encoding(in_path: PathLike) -> np.ndarray:
    """Load an embedding previously written by `save_encoding`."""
    data = json.loads(Path(in_path).read_text())
    return np.array(data, dtype=np.float64)
