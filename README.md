# HH Goa 2026 — Task 3: Face ID + Blockchain Verification Pipeline

Pipeline shape: **face scan → web/social search for a matching post →
blockchain upload & re-verification of the discovered data.**

This repo is built in stages. This README grows as each stage lands.

## Status

- [x] **Stage 1 — Face detection & encoding** — validated on 7 real
      photos; 14 unit tests + 9 integration checks passing
- [x] **Stage 2 — Web/social search** — reverse image search, social
      filtering, face re-verification; 27 unit tests passing
- [ ] Stage 3 — Blockchain upload + re-verification

---

## Stage 1: Face detection & encoding

Detects a face in a photo and computes a 512-dimensional embedding using
**[DeepFace](https://github.com/serengil/deepface) with the ArcFace
model** and RetinaFace for detection.

### Project layout

```
hh-goa-2026-pipeline/
├── requirements.txt          # pinned runtime deps
├── requirements-dev.txt      # + pytest
├── .env.example              # credential template (copy to .env)
├── src/
│   ├── faceid/
│   │   ├── __init__.py
│   │   └── face_encode.py    # Stage 1: encoding + gallery matching
│   └── search/
│       ├── __init__.py
│       ├── reverse_search.py # Stage 2: browser-driven reverse image search
│       ├── social_filter.py  # Stage 2: classify/rank social results
│       ├── verify_match.py   # Stage 2: re-verify the face on the found page
│       └── record.py         # Stage 2->3 canonical record + hashing
├── scripts/
│   ├── demo_encode.py        # CLI demo: encode one photo, save the result
│   ├── scan_face.py          # LIVE webcam capture -> encode (the real input step)
│   ├── build_gallery.py      # build a multi-photo reference gallery
│   ├── verify_stage1.py      # full self-check (9 integration checks)
│   └── find_match.py         # Stage 2: search -> filter -> verify -> record
├── tests/
│   ├── test_face_encode.py   # Stage 1 unit tests
│   ├── test_search.py        # Stage 2 unit tests
│   └── fixtures/             # saved HTML for offline parser tests
└── data/
    ├── sample_images/        # your photos (git-ignored)
    │   └── refs/             # gallery reference photos
    └── output/               # embeddings + gallery land here (git-ignored)
```

Stages 2 and 3 import from `src/faceid/face_encode.py` rather than
duplicating detection or matching logic.

### Setup

```bash
cd hh-goa-2026-pipeline
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Install notes:** DeepFace pulls in TensorFlow, so the install is large
(~2.5GB virtualenv). On first run it downloads ~335MB of model weights
to `~/.deepface/weights/` — the first encode takes ~15s, subsequent ones
~3.7s per image on CPU. No compiler or CMake needed (unlike dlib).

### Add your photos

```
data/sample_images/me.jpg          # single test photo
data/sample_images/refs/*.jpg      # 4-6 varied photos for the gallery
```

Use varied conditions for the gallery — different days, lighting,
angles, with and without glasses. Variety matters more than count.

### Run

```bash
# encode one photo
python scripts/demo_encode.py
python scripts/demo_encode.py path/to/any/photo.jpg

# capture a face scan LIVE from your webcam (needs a real display + camera)
python scripts/scan_face.py
python scripts/scan_face.py --countdown 3   # hands-free, for the screen recording

# build the reference gallery (+ leave-one-out validation)
python scripts/build_gallery.py

# full self-check
python scripts/verify_stage1.py
```

### Live webcam capture

`scripts/scan_face.py` is the actual "face scan input" step the task
describes — the pipeline's real input is a live camera capture, not a
photo picked from disk. It opens the default webcam, shows a live
preview, and on `SPACE` (or after a `--countdown`, for a hands-free
recording):

1. saves the raw frame to `data/output/scan_<timestamp>.jpg`
2. encodes it with the same `encode_face_from_array()` used everywhere
   else in this module — no separate code path
3. if `data/output/gallery.json` exists, matches it against the gallery
   immediately and prints the distance and verdict, so you get a live
   "yes, this is really you" check before recording continues

This needs a real display and webcam attached to the machine it runs
on — it will not work over SSH or in a headless/cloud environment.
`ESC` quits without capturing. A bad or busy camera prints some noisy
OpenCV/FFmpeg warnings before a clean error message — those warnings
are harmless.

Verified in development: passing a real photo through the same code
path `scan_face.py` uses gave `distance=0.0000` against that photo's
own file-based encoding (expected — identical pixels), and a non-face
frame was rejected with no crash. The live-camera loop itself needs a
physical webcam to test and could not be run in the environment this
was built in — verify it once on your own machine before recording.

Verified `verify_stage1.py` output on a real photo:

```
1) Detection
  [PASS] exactly one face found — 1 face(s)
       bbox (top,right,bottom,left) = (703, 589, 1251, 172) -> 417x548px
2) Encoding
  [PASS] shape (512,) float64
       norm=3.3293 min=-0.5409 max=0.4123 mean=0.0058
3) Determinism
  [PASS] re-encoding gives distance ~0 — distance=0.000000000
4) Persistence
  [PASS] JSON round-trip lossless
5) Robustness (same person, degraded image)
  [PASS] 3x downscaled + recompressed still matches — distance=0.0478
6) Error handling — multiple faces
  [PASS] raises MultipleFacesDetectedError — num_faces=2
  [PASS] allow_multiple=True returns both — 2 encodings
7) Error handling — no face
  [PASS] raises NoFaceDetectedError
8) Error handling — missing file
  [PASS] raises FileNotFoundError

  Stage 1 verification: 9/9 checks passed
```

### Error handling

- **No face found** → `NoFaceDetectedError`
- **More than one face** → `MultipleFacesDetectedError` (pass
  `allow_multiple=True` to get every face instead)
- **Missing file** → `FileNotFoundError`

All subclass `FaceEncodingError` except the last. A failed encode writes
**no output file**, so bad input can't leak into the Stage 3 hashing step.

Tested against a non-face image (a stock photo of a tree): 0 faces
detected, no false positives, clean rejection.

```bash
pip install -r requirements-dev.txt
pytest tests/          # 41 tests (Stage 1 + Stage 2)
```

---

### Why DeepFace / ArcFace (and not dlib)

Both backends were benchmarked on the **same** 7 real photos — 6 of one
person across varied lighting, indoor/outdoor, glasses/no-glasses and
angles, plus 1 impostor — using identical leave-one-out methodology.

Raw distances aren't comparable across backends (dlib uses Euclidean on
128-d vectors, DeepFace uses cosine on 512-d), so these are scale-free:

| Backend | Separation ratio | Normalised margin | Works without a gallery? | Speed |
|---|---|---|---|---|
| dlib / face_recognition | 1.263 | 0.208 | no — distributions overlap | 4.0s |
| DeepFace Facenet512 | 1.334 | 0.250 | no — distributions overlap | 3.5s |
| **DeepFace ArcFace** | **1.423** | **0.297** | **yes** | **3.7s** |

*separation ratio* = impostor distance ÷ worst true-match distance
(higher is better; 1.0 means they touch).
*normalised margin* = (impostor − worst true) ÷ impostor.

ArcFace gives **~43% more headroom** than dlib and was the only backend
that still separated correctly without a reference gallery. It's trained
on more demographically diverse data than dlib's model — relevant here,
since the false positive that prompted this comparison was between two
South Asian men.

**Trade-off:** DeepFace pulls in TensorFlow — ~2.5GB virtualenv versus
~271MB for dlib, plus ~335MB of weights on first run. Per-image speed is
comparable. dlib remained viable (the gallery fixed its false positive),
but ArcFace was chosen for the wider margin.

### Why matching uses a multi-photo gallery

**One reference photo is not enough.** Leave-one-out on 6 photos of one
person + 1 impostor, cosine distance with ArcFace:

| Approach | Same person | Impostor | Margin |
|---|---|---|---|
| Single reference | up to **0.7435** | **0.7662** | 0.023 — separable, barely |
| **Gallery (minimum over refs)** | up to **0.5384** | **0.7662** | **0.2278 — ~10× wider** |

With dlib the single-reference case was worse still: two photos of the
*same* person scored 0.602 apart while a *different* person sat at
0.562 — the distributions overlapped outright, so no threshold could
work at all.

Matching against a gallery and taking the **minimum** distance means the
candidate only has to resemble the person in *one* reference photo,
which is what makes it robust to lighting and pose.

Per-photo leave-one-out (ArcFace, gallery of 6):

| Reference | Closest match |
|---|---|
| original | 0.421 |
| mirror selfie | 0.527 |
| indoor (bed) | 0.212 |
| **outdoor, harsh sun** | **0.538** ← worst |
| indoor | 0.212 |
| indoor 2 | 0.249 |
| **impostor** | **0.766** |

Real result on a two-person group photo:

| Face | Distance | Verdict |
|---|---|---|
| The subject | **0.346** | MATCH |
| A different person | **0.758** | no match |

### Matching threshold — why 0.65

DeepFace's calibrated default for ArcFace + cosine is **0.68**. This
project uses **0.65**, slightly stricter, based on the measurements
above: worst true match 0.5384, impostor 0.7662 — 0.65 sits near the
midpoint with ~0.11 headroom either side.

The bias toward strictness is deliberate. In Stage 2 a false positive
means claiming a **stranger's** social media post belongs to the user,
and Stage 3 then writes that claim to a blockchain permanently. A missed
match is recoverable; a wrong match written on-chain is not.

### API reference (for later stages)

```python
from faceid.face_encode import (
    encode_face_from_path,      # main entry point -> (512,) ndarray
    encode_face_from_array,     # for images downloaded in Stage 2
    detect_faces,               # low-level: all faces + locations
    compare_encodings,          # (is_match, distance)
    find_best_match,            # closest of N -> (index, distance, is_match)
    cosine_distance,            # the metric ArcFace is calibrated for
    FaceGallery,                # multi-reference matching
    save_encoding, load_encoding,
    NoFaceDetectedError, MultipleFacesDetectedError,
)

gallery = FaceGallery.load("data/output/gallery.json")
is_match, distance, which_ref = gallery.match(candidate_embedding)
```

`FaceGallery.load()` refuses to load a gallery built with a different
model, since embeddings from different models are not comparable.

---

## Known limitations (Stage 1)

- **Matching is not identity proof.** Distance is a similarity score, not
  a guarantee. Even with a 6-photo gallery the gap between the worst true
  match (0.538) and the impostor (0.766) is 0.228 wide, so genuinely
  similar-looking people — relatives especially — can land near the
  boundary.
- **The threshold is validated on a small sample:** 6 photos of one
  person and 1 impostor. Enough to show single-reference matching was
  broken; not a rigorous FAR/FRR evaluation. Stage 2 should surface the
  distance alongside every verdict rather than a bare yes/no.
- **Hard photos degrade fast.** The worst reference (bright direct
  sunlight, heavy shadow, glare on glasses) sat at 0.538 — closest to
  the threshold of any true match. Backlit and strongly side-lit images
  are the weak point.
- **Demographic bias.** ArcFace is better than dlib here but no face
  model is neutral; error rates still vary across demographics.
- **Heavy install.** TensorFlow + model weights make this a ~2.8GB
  footprint, and the first run needs internet to fetch weights.
- Uses the `retinaface` detector (most accurate DeepFace offers). Swap
  to `opencv` via `detector_backend=` for speed at some accuracy cost.

---

## Stage 2: Web / social media search

Takes the face scan, runs a **genuine reverse image search**, filters the
results down to real social media posts, and re-verifies the face on the
page it found.

```bash
python scripts/find_match.py                      # uses latest webcam scan
python scripts/find_match.py path/to/photo.jpg
python scripts/find_match.py --manual             # if automation is blocked
python scripts/find_match.py --engine google
```

Output: `data/output/match_record.json` — the canonical record Stage 3
hashes and writes on-chain.

### Why browser automation instead of an API

The task allows the search step "via reverse image search, an API, or a
scripted search approach". Every hosted reverse-image API was gated
behind payment or identity verification:

| Service | Blocker |
|---|---|
| Google Cloud Vision | billing account + card deposit required |
| SerpApi | phone verification (failed across 5 numbers) |
| TinEye | no free tier at all — search bundles must be purchased |

So this uses the scripted approach the spec explicitly permits: Playwright
drives a real browser through a real reverse image search and parses the
real results page. **Nothing is hardcoded** — the URLs reported are
whatever the engine actually returns.

**Yandex is the default engine**, not Google, because its image index is
markedly better at matching *faces* across the web — which is precisely
this pipeline's use case. Google is supported via `--engine google` but
blocks automation far more aggressively.

### Manual mode

Search engines change their DOM and deploy bot-detection without notice.
`--manual` opens the browser at the search page, you do the upload by
hand, press Enter, and the script scrapes whatever results page is open.

This still satisfies "a scripted search approach" — the script performs
the parsing, filtering and verification on real live results. It exists
so that a DOM change the day before a deadline degrades the demo rather
than breaking it. It works with **any** engine, because it falls back to
generic outbound-link extraction.

### Three steps, and what each guarantees

1. **Reverse image search** (`src/search/reverse_search.py`) — real
   browser, real search, real results. Engine-specific parsing with a
   generic fallback so a DOM change degrades quality instead of failing.
2. **Social filtering** (`src/search/social_filter.py`) — a reverse
   image search returns every page hosting the image: news sites,
   scrapers, CDNs. This classifies results by platform and ranks genuine
   social posts above aggregators like Pinterest (which are usually
   re-pins of someone else's upload, not the original).
3. **Face re-verification** (`src/search/verify_match.py`) — the search
   engine found a *visually similar* image, which is not the same claim
   as "this page shows the person we scanned". This downloads the
   candidate image and runs it back through Stage 1's gallery, producing
   a measured cosine distance rather than trusting the engine.

Step 3 matters because Stage 3 writes the result to a blockchain
permanently. A wrong match is not recoverable, so the standard for "this
is really them" is evidence, not the search engine's word.

### The hand-off record

`match_record.json` separates the **hashed payload** (what was found)
from **metadata** (when it was found, local file paths, result counts):

```json
{
  "content_hash": "dcbc61c9...",
  "hashed_payload": {
    "post_url": "https://www.instagram.com/p/...",
    "platform": "Instagram",
    "query_image_sha256": "1ea08c5f...",
    "face_verified": true,
    "face_distance": 0.0
  },
  "discovered_at": "2026-09-05T21:37:54+00:00"
}
```

Timestamps and local paths are deliberately **excluded from the hash**.
If they were included, re-running verification tomorrow would produce a
different hash and the on-chain check would fail even though nothing was
tampered with. Serialisation is canonical (sorted keys, fixed separators,
UTF-8), so the same discovery always produces identical bytes on any
machine. This is tested directly — see `tests/test_search.py`.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Found a social media post |
| 1 | Search failed (no image, browser error, engine blocked) |
| 2 | Search succeeded but found no social post — a real result, not a bug |

### Known limitations (Stage 2)

- **Reverse image search finds the photo, not the person.** If the exact
  image (or a near-duplicate) has never been posted publicly, the search
  legitimately returns nothing. Use a photo that is actually public — a
  LinkedIn profile picture is usually the most reliably indexed.
- **Instagram Stories will never work** — they're ephemeral and have no
  public crawlable URL. It must be a feed post on a public account.
- **Scraping is more fragile than an API.** Selectors are isolated in one
  place and there is a generic fallback plus manual mode, but a
  sufficiently large redesign will need the selectors updated.
- **Hotlink blocking limits verification.** Most social platforms block
  direct image fetches, so face re-verification often can't run on the
  final post; the record records that honestly (`attempted: false`)
  rather than claiming a verification that didn't happen.
- **Requires live internet**, unlike Stage 1 which is fully offline once
  model weights are cached.

---

## Stage 3 (to follow)

Hash the discovered post (`content_hash` above) and write it to a
blockchain, then demonstrate re-verification against the on-chain record.

Credentials go in `.env` (see `.env.example`). `.env`, service-account
JSONs and `*-key.json` are git-ignored — never commit real keys.
