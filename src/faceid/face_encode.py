"""
face_encode.py — Task 1 of the HH Goa 2026 pipeline: face detection & encoding.

Reusable, importable functions for turning a photo into a 128-dimensional
face encoding using the `face_recognition` library (dlib's ResNet-based
face recognition model under the hood).

Later pipeline stages (web/social search, blockchain verification) import
`encode_face_from_path` (or `encode_face_from_array`) from this module
rather than re-implementing detection/encoding.

Typical usage
-------------
    from faceid.face_encode import encode_face_from_path

    encoding = encode_face_from_path("data/sample_images/me.jpg")
    # encoding is a numpy.ndarray of shape (128,), dtype float64

Error handling
--------------
Two things can go wrong with a real-world photo and both are treated as
first-class, catchable errors rather than crashes:

  * No face found at all               -> NoFaceDetectedError
  * More than one face found           -> MultipleFacesDetectedError
    (unless `allow_multiple=True`, in which case all encodings are
    returned instead of raising)

Both errors subclass FaceEncodingError, so callers that don't care about
the distinction can just catch that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

try:
    import face_recognition
except ImportError as exc:  # pragma: no cover - import-time guidance
    raise ImportError(
        "The 'face_recognition' package (and its dlib dependency) is not "
        "installed in this environment. Run:\n"
        "    pip install -r requirements.txt\n"
        "See README.md for platform-specific dlib build notes "
        "(cmake / a C++ compiler are required)."
    ) from exc


PathLike = Union[str, Path]

# "hog" is CPU-only and fast (good default for a laptop / this prototype).
# "cnn" is far more accurate but needs a GPU (or is very slow on CPU).
DEFAULT_MODEL = "hog"

# face_recognition's own documented default is 0.6, but that is measurably
# too loose for this pipeline. Measured on our own test data:
#
#   same person, same photo downscaled 3x + recompressed .... 0.09
#   same person, different photo/pose/lighting/camera ........ 0.41
#   DIFFERENT people (two friends in one group photo) ........ 0.58
#
# At 0.6 the different person is a false positive; at 0.5 the real match
# still passes with room to spare and the impostor is correctly rejected.
# Stage 2 will be comparing against faces found on the open web, where a
# false positive means claiming a stranger's post belongs to you — so we
# bias toward strictness. Override per-call if you need to.
DEFAULT_TOLERANCE = 0.5


class FaceEncodingError(Exception):
    """Base class for all face-encoding errors raised by this module."""


class NoFaceDetectedError(FaceEncodingError):
    """Raised when zero faces are found in the given image."""

    def __init__(self, image_path: PathLike):
        super().__init__(
            f"No face detected in '{image_path}'. Make sure the photo "
            "clearly shows a front-facing face, is well lit, and isn't "
            "too low-resolution or heavily cropped."
        )
        self.image_path = str(image_path)


class MultipleFacesDetectedError(FaceEncodingError):
    """Raised when more than one face is found and allow_multiple=False."""

    def __init__(self, image_path: PathLike, num_faces: int):
        super().__init__(
            f"Found {num_faces} faces in '{image_path}', expected exactly "
            "one. Crop the photo to a single person, or call "
            "encode_face_from_path(..., allow_multiple=True) if you "
            "intentionally want encodings for every face in the image."
        )
        self.image_path = str(image_path)
        self.num_faces = num_faces


@dataclass
class FaceMatch:
    """One detected face: its encoding plus where it was found in the image."""

    encoding: np.ndarray  # shape (128,), dtype float64
    location: tuple  # (top, right, bottom, left) in pixels


def _load_image(image_path: PathLike) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: '{image_path}'")
    # face_recognition.load_image_file returns an RGB numpy array and
    # raises its own error for unreadable / corrupt files.
    return face_recognition.load_image_file(str(image_path))


def detect_faces(
    image_path: PathLike,
    model: str = DEFAULT_MODEL,
    upsample: int = 1,
    num_jitters: int = 1,
) -> List[FaceMatch]:
    """
    Detect every face in an image and return an encoding + location for each.

    This is the low-level building block. Most callers should use
    `encode_face_from_path` instead, which adds the "exactly one face"
    validation that later pipeline stages rely on.

    Args:
        image_path: path to a JPEG/PNG (or anything Pillow can read).
        model: "hog" (fast, CPU) or "cnn" (accurate, needs GPU / is slow).
        upsample: how many times to upsample the image before detecting —
            increase (e.g. to 2) to find small/far-away faces, at the
            cost of speed.
        num_jitters: how many times to re-sample the face when computing
            the encoding — higher is more accurate but slower.

    Returns:
        A list of FaceMatch (possibly empty if no faces were found).
    """
    image = _load_image(image_path)
    locations = face_recognition.face_locations(
        image, number_of_times_to_upsample=upsample, model=model
    )
    encodings = face_recognition.face_encodings(
        image, known_face_locations=locations, num_jitters=num_jitters
    )
    return [
        FaceMatch(encoding=enc, location=loc)
        for enc, loc in zip(encodings, locations)
    ]


def encode_face_from_path(
    image_path: PathLike,
    model: str = DEFAULT_MODEL,
    upsample: int = 1,
    num_jitters: int = 1,
    allow_multiple: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    The main entry point: encode the single face in an image.

    Args:
        image_path: path to the input photo.
        model: "hog" (default, fast/CPU) or "cnn" (slow/GPU, more accurate).
        upsample: see `detect_faces`.
        num_jitters: see `detect_faces`.
        allow_multiple: if False (default), raises MultipleFacesDetectedError
            when more than one face is found. If True, returns a list of
            encodings (one per face) instead of a single array.

    Returns:
        A single numpy array of shape (128,) when exactly one face is
        found. If allow_multiple=True, returns a list of such arrays
        (one per detected face, in the order face_recognition found them).

    Raises:
        FileNotFoundError: image_path does not exist.
        NoFaceDetectedError: zero faces found.
        MultipleFacesDetectedError: more than one face found and
            allow_multiple=False.
    """
    matches = detect_faces(
        image_path, model=model, upsample=upsample, num_jitters=num_jitters
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
    model: str = DEFAULT_MODEL,
    upsample: int = 1,
    num_jitters: int = 1,
    allow_multiple: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Same as encode_face_from_path, but takes an already-loaded RGB image
    array (e.g. a frame grabbed from a webcam or downloaded from the web)
    instead of a file path. Useful for later stages that fetch images
    over the network rather than reading them from disk.
    """
    locations = face_recognition.face_locations(
        image, number_of_times_to_upsample=upsample, model=model
    )
    encodings = face_recognition.face_encodings(
        image, known_face_locations=locations, num_jitters=num_jitters
    )

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
    Compare two face encodings.

    Returns (is_match, distance). Lower distance = more similar.

    See DEFAULT_TOLERANCE above for why this project uses 0.5 rather
    than face_recognition's documented 0.6 default.
    """
    distance = float(
        np.linalg.norm(np.asarray(known_encoding) - np.asarray(candidate_encoding))
    )
    return distance <= tolerance, distance


def find_best_match(
    known_encoding: np.ndarray,
    candidates: Sequence[np.ndarray],
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[Optional[int], float, bool]:
    """
    Given one reference encoding and several candidate encodings (e.g.
    every face found in a group photo, or in an image scraped from a
    search result), return the SINGLE closest candidate.

    This is what Stage 2 should use rather than "every face under the
    threshold": in a group photo more than one face can sit under the
    tolerance, and picking all of them would mean claiming a stranger's
    post belongs to you.

    Returns:
        (index_of_closest, distance, is_match)
        index is None if `candidates` was empty. `is_match` is False
        when even the closest candidate is beyond `tolerance`.
    """
    if len(candidates) == 0:
        return None, float("inf"), False

    distances = [
        float(np.linalg.norm(np.asarray(known_encoding) - np.asarray(c)))
        for c in candidates
    ]
    best_idx = int(np.argmin(distances))
    best_distance = distances[best_idx]
    return best_idx, best_distance, best_distance <= tolerance


@dataclass
class FaceGallery:
    """
    Several reference encodings for ONE person.

    A single reference photo is not enough. Measured on 6 photos of the
    same person plus 1 impostor:

        single reference  -> same-person distances spanned 0.297-0.602,
                             impostor distances 0.562-0.707. Those
                             OVERLAP by 0.040, so no threshold can be
                             both safe and correct.
        gallery (min over 6 refs, leave-one-out)
                          -> every true photo landed <= 0.445,
                             impostor at 0.562. Margin 0.117 — about 3x
                             wider, and cleanly separable.

    So Stage 2 should always match against a gallery, not one photo.
    """

    labels: List[str]
    encodings: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.encodings)

    def match(
        self, candidate: np.ndarray, tolerance: float = DEFAULT_TOLERANCE
    ) -> tuple[bool, float, Optional[str]]:
        """
        Compare a candidate face against every reference in the gallery.

        Returns (is_match, best_distance, label_of_closest_reference).
        Uses the MINIMUM distance across references: the candidate only
        has to look like the person in *one* of their reference photos,
        which is what makes the gallery robust to lighting/pose/angle.
        """
        if not self.encodings:
            return False, float("inf"), None
        idx, dist, is_match = find_best_match(candidate, self.encodings, tolerance)
        return is_match, dist, self.labels[idx] if idx is not None else None

    def save(self, out_path: PathLike) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "labels": self.labels,
            "encodings": [np.asarray(e).tolist() for e in self.encodings],
        }
        out_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, in_path: PathLike) -> "FaceGallery":
        data = json.loads(Path(in_path).read_text())
        return cls(
            labels=data["labels"],
            encodings=[np.array(e, dtype=np.float64) for e in data["encodings"]],
        )

    @classmethod
    def from_paths(
        cls,
        image_paths: Sequence[PathLike],
        model: str = DEFAULT_MODEL,
        skip_failures: bool = True,
    ) -> "FaceGallery":
        """
        Build a gallery from several photos of the same person.

        Photos that contain no face (or more than one) are skipped with a
        warning when skip_failures=True, so one bad photo doesn't sink
        the whole gallery. For multi-face photos the largest face is
        used, on the assumption it's the subject closest to the camera.
        """
        labels, encodings = [], []
        for p in image_paths:
            p = Path(p)
            try:
                matches = detect_faces(p, model=model)
                if not matches:
                    matches = detect_faces(p, model=model, upsample=2)
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
    """Persist an encoding to disk as JSON, for later pipeline stages to load."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(np.asarray(encoding).tolist()))


def load_encoding(in_path: PathLike) -> np.ndarray:
    """Load an encoding previously written by `save_encoding`."""
    data = json.loads(Path(in_path).read_text())
    return np.array(data, dtype=np.float64)
