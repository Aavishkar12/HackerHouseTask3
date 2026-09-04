# HH Goa 2026 — Task 3: Face ID + Blockchain Verification Pipeline

Pipeline shape: **face scan → web/social search for a matching post →
blockchain upload & re-verification of the discovered data.**

This repo is built in stages. This README will grow as each stage lands.

## Status

- [x] **Stage 1 — Face detection & encoding** (this stage) — verified
      end-to-end against a real photo, 9/9 checks passing
- [ ] Stage 2 — Web/social media search for a matching post
- [ ] Stage 3 — Blockchain upload + re-verification

## Stage 1: Face detection & encoding

Detects a face in a photo and computes its 128-dimensional encoding
using the [`face_recognition`](https://github.com/ageitgeit/face_recognition)
library (dlib's ResNet face-recognition model under the hood).

### Project layout

```
hh-goa-2026-pipeline/
├── requirements.txt          # pinned runtime deps
├── requirements-dev.txt      # + pytest, for running tests
├── src/
│   └── faceid/
│       ├── __init__.py
│       └── face_encode.py    # reusable, importable encoding functions
├── scripts/
│   ├── demo_encode.py        # CLI demo: encode a sample photo, save the result
│   └── verify_stage1.py      # full self-check (detection, determinism, error paths)
├── tests/
│   └── test_face_encode.py   # automated tests for error handling
└── data/
    ├── sample_images/        # put your test photo here (git-ignored)
    └── output/                # saved encodings land here (git-ignored)
```

Later stages (web search, blockchain) will import from
`src/faceid/face_encode.py` rather than duplicating detection logic.

### Setup

```bash
cd hh-goa-2026-pipeline
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Install notes (dlib):** `face_recognition` depends on `dlib`, which is
compiled from source by pip and needs a C++ toolchain + CMake:

- **Ubuntu/Debian:** `sudo apt-get install -y cmake build-essential`
- **macOS:** `xcode-select --install` and `brew install cmake`
- **Windows:** install "Desktop development with C++" via the Visual
  Studio Build Tools, plus [CMake](https://cmake.org/download/). If the
  dlib build still fails, the easiest fix is usually
  `pip install cmake` first, then retry `pip install dlib`.

If dlib/`face_recognition` genuinely won't build on your machine, the
recommended fallback is `mediapipe` (`pip install mediapipe`) or
`insightface`, either of which can be swapped in behind the same
`encode_face_from_path(...)` function signature in `face_encode.py`.
**In this environment, dlib and face_recognition built and installed
without any issues**, so no fallback was needed — see "Why
face_recognition" below.

### Add your test photo

Drop a clear, front-facing photo of yourself at:

```
data/sample_images/me.jpg
```

(or pass any other path as an argument — see below).

### Run the demo

```bash
python scripts/demo_encode.py
# or: python scripts/demo_encode.py path/to/any/photo.jpg
```

Expected output on success:

```
[*] Encoding face(s) in: data/sample_images/me.jpg
[+] Success — got one face encoding.
    shape: (128,), dtype: float64
    first 8 values: [...]
[+] Saved full encoding to: data/output/me_encoding.json
```

The saved JSON file is what Stage 2 (web search) will eventually load
via `faceid.face_encode.load_encoding(...)`.

### Verify the whole stage

`demo_encode.py` just proves it runs. To prove it actually *works*:

```bash
python scripts/verify_stage1.py
```

This runs 9 checks — detection + bounding box (saved as an annotated
image you can eyeball), encoding shape/dtype, determinism, JSON
round-trip, robustness to downscaling/re-compression, and all three
error paths. Verified output on a real test photo:

```
1) Detection
  [PASS] exactly one face found — 1 face(s)
       bbox (top,right,bottom,left) = (759, 605, 1221, 142) -> 463x462px
2) Encoding
  [PASS] shape (128,) float64
       norm=1.3703 min=-0.3127 max=0.3744 mean=-0.0028
3) Determinism
  [PASS] re-encoding gives distance ~0 — distance=0.000000000
4) Persistence
  [PASS] JSON round-trip lossless
5) Robustness (same person, degraded image)
  [PASS] 3x downscaled + recompressed still matches — distance=0.0900 (tolerance 0.6)
6) Error handling — multiple faces
  [PASS] raises MultipleFacesDetectedError — num_faces=2
  [PASS] allow_multiple=True returns both — 2 encodings
7) Error handling — no face
  [PASS] raises NoFaceDetectedError
8) Error handling — missing file
  [PASS] raises FileNotFoundError

  Stage 1 verification: 9/9 checks passed
```

Encoding a single photo takes ~4s on CPU with the `hog` model.

### Error handling

- **No face found** → raises `NoFaceDetectedError` with a clear message
  (demo script prints it and exits non-zero instead of crashing).
- **More than one face found** → raises `MultipleFacesDetectedError`
  by default. Callers that *want* every face can pass
  `allow_multiple=True` to get a list of encodings back instead.
- **Missing file** → a plain `FileNotFoundError` with the path that was
  looked for.

These are exercised by the automated tests:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

(The automated tests only cover the missing-file and no-face-found
paths, since a true positive-path test needs a real face — that's what
running `scripts/demo_encode.py` against your own photo is for.)

### Why `face_recognition`

Per the task brief's suggested approach, `face_recognition` (dlib-based)
was tried first since it gets encoding + matching working the fastest.
It built and installed cleanly in this environment (cmake and a C++
compiler were already present), so **no fallback to `mediapipe` /
`insightface` was needed**. If you hit a dlib build failure on your own
machine, see the install notes above, or swap the implementation inside
`encode_face_from_path()` for a `mediapipe`/`insightface`-based one — the
function signature (`image_path -> np.ndarray` of a fixed-length
encoding) is designed to stay the same either way, so nothing downstream
would need to change.

### API reference (for later stages)

```python
from faceid.face_encode import (
    encode_face_from_path,      # main entry point
    encode_face_from_array,     # for in-memory / downloaded images
    detect_faces,               # low-level: all faces + locations
    compare_encodings,          # (is_match, distance) between two encodings
    find_best_match,            # closest of N candidates -> (index, distance, is_match)
    save_encoding, load_encoding,  # persist/reload an encoding as JSON
    NoFaceDetectedError,
    MultipleFacesDetectedError,
)

encoding = encode_face_from_path("data/sample_images/me.jpg")
# encoding: np.ndarray, shape (128,), dtype float64
```

## Matching threshold — why 0.5, not 0.6

`face_recognition`'s documented default tolerance is **0.6**. Testing on
real photos showed that is **too loose for this pipeline**, so this
project defaults to **0.5** (`DEFAULT_TOLERANCE` in `face_encode.py`).

Measured distances on our own test images:

| Comparison | Distance |
|---|---|
| Same person, same photo downscaled 3× + recompressed | **0.09** |
| Same person, different photo (pose/lighting/camera differ) | **0.41** |
| **Different people** (two friends in one group photo) | **0.58** |

At tolerance 0.6 the *different person* is a false positive. A threshold
sweep confirms the safe band:

| Tolerance | Real match found | Impostor rejected | Verdict |
|---|---|---|---|
| 0.60 | yes | **no** | false positive |
| 0.55 | yes | yes | correct |
| **0.50** | yes | yes | **correct (project default)** |
| 0.45 | yes | yes | correct |
| 0.40 | **no** | yes | too strict, misses real match |

Raising `num_jitters` from 1 → 50 barely moved the numbers (0.5775 →
0.5635 for the impostor), so this is a threshold problem, not an
encoding-quality problem — jittering is not a fix.

This matters most for Stage 2: a false positive there means claiming a
**stranger's** social media post belongs to you. Hence `find_best_match()`,
which returns the *single closest* face rather than every face under the
threshold — in a group photo more than one face can pass.

## Known limitations (Stage 1)

- **Matching is not identity proof.** A 128-d encoding distance is a
  similarity score, not a guarantee. The measured gap between "same
  person, different photo" (0.41) and "different person" (0.58) is only
  ~0.17 wide, so genuinely similar-looking people — especially relatives,
  or people of a similar age/build wearing similar glasses — can land
  close to the boundary.
- **Known model bias.** dlib's face-recognition model was trained largely
  on Western/white face datasets and has documented higher error rates on
  other demographics. The false positive found during testing was between
  two South Asian men, which is consistent with that. The stricter 0.5
  threshold mitigates but does not remove this.
- Uses the `"hog"` detection model (fast, CPU-only); it's less accurate
  on small, angled, or poorly-lit faces than the `"cnn"` model, which
  needs a GPU to run at reasonable speed. Pass `model="cnn"` to
  `encode_face_from_path` if a GPU is available. (Note: the CNN model
  did *not* fix the false positive above — again, threshold, not model.)
- Only handles one face per image by default (by design, since this
  pipeline is about identifying one person); `allow_multiple=True` is
  available for multi-face images.
- Encoding takes ~4s per photo on CPU with `hog`; the `cnn` model is
  meaningfully slower on CPU.

## Stages 2 & 3

To follow, once Stage 1 is confirmed working end-to-end against a real
photo:

- **Stage 2:** use the face encoding to run a genuine web/reverse-image
  search (e.g. a Vision API's web-detection feature) and find a real
  matching social media post.
- **Stage 3:** hash the discovered post and write it to a blockchain
  (local/simulated, testnet, or mainnet), then demonstrate
  re-verification against the on-chain record.
