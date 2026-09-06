# HH Goa 2026 — Task 3: Face ID + Blockchain Verification Pipeline

Pipeline shape: **face scan → web/social search for a matching post →
blockchain upload & re-verification of the discovered data.**

## What this does

1. **Scans a face** from a live webcam and encodes it as a 512-dimensional
   ArcFace embedding, then identifies it against an enrolled gallery.
2. **Searches the web** with a genuine reverse image search (browser
   automation against Yandex/Google — nothing hardcoded), filters the
   results down to real social media posts, and re-verifies the face on
   the page it found.
3. **Anchors the discovery on a blockchain** as a SHA-256 fingerprint,
   then re-verifies it by recomputing that hash and looking it up on
   chain — proving the record hasn't been altered since.

Finding nothing is a real, reported outcome at every stage. Nothing in
this pipeline fabricates a match.

## Status

- [x] **Stage 1 — Face detection & encoding** — validated on 7 real
      photos; 14 unit tests + 9 integration checks passing
- [x] **Stage 2 — Web/social search** — reverse image search, social
      filtering, face re-verification; 32 unit tests passing
- [x] **Stage 3 — Blockchain anchoring & verification** — Solidity
      contract on an EVM chain, tamper detection demonstrated; 33 unit
      tests passing against a real in-process EVM

**Quick start:** [Running the whole pipeline end to end](#running-the-whole-pipeline-end-to-end)

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
├── .env.example              # only needed for a public testnet (Stage 3)
├── contracts/
│   ├── MatchRegistry.sol     # Stage 3: the contract (90 lines)
│   └── MatchRegistry.json    # precompiled ABI + bytecode (no solc needed)
├── src/
│   ├── faceid/
│   │   ├── __init__.py
│   │   └── face_encode.py    # Stage 1: encoding + gallery matching
│   ├── search/
│   │   ├── __init__.py
│   │   ├── reverse_search.py # Stage 2: browser-driven reverse image search
│   │   ├── social_filter.py  # Stage 2: classify/rank social results
│   │   ├── verify_match.py   # Stage 2: re-verify the face on the found page
│   │   └── record.py         # Stage 2->3 canonical record + hashing
│   └── chain/
│       ├── __init__.py
│       └── registry.py       # Stage 3: connect / deploy / anchor / verify
├── scripts/
│   ├── demo_encode.py        # CLI demo: encode one photo, save the result
│   ├── scan_face.py          # LIVE webcam capture -> encode (the real input step)
│   ├── build_gallery.py      # build a multi-photo reference gallery
│   ├── verify_stage1.py      # full self-check (9 integration checks)
│   ├── find_match.py         # Stage 2: search -> filter -> verify -> record
│   ├── deploy_contract.py    # Stage 3: deploy MatchRegistry
│   ├── anchor_record.py      # Stage 3: hash the record, write it on chain
│   ├── verify_onchain.py     # Stage 3: recompute + verify (+ --tamper proof)
│   └── compile_contract.py   # optional: rebuild the artifact from the .sol
├── tests/
│   ├── test_face_encode.py   # Stage 1 unit tests
│   ├── test_search.py        # Stage 2 unit tests
│   ├── test_chain.py         # Stage 3 unit tests (real in-process EVM)
│   └── fixtures/             # saved HTML for offline parser tests
└── data/
    ├── sample_images/        # your photos (git-ignored)
    │   └── refs/             # gallery reference photos
    └── output/               # embeddings, gallery, records (git-ignored)
```

Each stage imports from the one before rather than duplicating logic:
Stage 2 and Stage 3 both use `src/faceid/face_encode.py` for detection and
matching, and Stage 3 hashes exactly the record Stage 2 wrote.

### Setup

```bash
cd hh-goa-2026-pipeline
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Stage 2 also needs a browser binary, and Stage 3 optionally needs a local
chain:

```bash
playwright install chromium      # Stage 2
npm install -g ganache           # Stage 3, only for the local-node mode
```

**Install notes:** DeepFace pulls in TensorFlow, so the install is large
(~2.5GB virtualenv). On first run it downloads ~335MB of model weights
to `~/.deepface/weights/` — the first encode takes ~15s, subsequent ones
~3.7s per image on CPU. No compiler or CMake needed (unlike dlib), and no
Solidity compiler either — the contract ships precompiled.

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
pytest tests/          # 46 tests (Stage 1 + Stage 2)
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
python scripts/find_match.py                      # uses newest webcam scan
python scripts/find_match.py --scan path/to/scan.jpg
python scripts/find_match.py --query-image path/to/indexed_photo.jpg
python scripts/find_match.py --manual             # if automation is blocked
python scripts/find_match.py --engine google
```

Output: `data/output/match_record.json` — the canonical record Stage 3
hashes and writes on-chain.

### The scan and the search image are two different things

This is the most important thing to understand about Stage 2.

**A live webcam frame can never be found by a reverse image search**, because
that exact image has never existed on the internet. That's a property of
how reverse image search works — it matches *images*, not faces — not a
limitation of this code. So the two images play different roles:

| Image | Role |
|---|---|
| **scan image** | identifies *who* is in front of the camera, by matching against the enrolled gallery — this is the face-ID step |
| **query image** | what actually gets searched for on the web — the enrolled reference photo of the person just identified |

The flow is therefore:

```
webcam scan ──encode──> match against enrolled gallery ──> "this is <subject>"
                                                              │
                          enrolled reference photo of <subject>
                                                              │
                                                    reverse image search
                                                              │
                                    filter to social posts ──> verify face ──> hash
```

`find_match.py` picks the query image automatically: whichever enrolled
photo the scan matched. Override it with `--query-image`.

The search itself remains completely genuine — choosing *what to query
with* is a design decision; hardcoding *the result* would be cheating,
and nothing here does that. Whatever the engine returns is what gets
filtered, verified and reported, and finding nothing is reported honestly.

**Practical consequence:** the subject you enrol needs at least one photo
that is actually indexed on the web, or the search will correctly find
nothing. Most private individuals have none — LinkedIn serves profile
photos from a CDN that image crawlers generally don't index, so even a
public profile usually isn't searchable by image.

### One gallery = one person

A `FaceGallery` represents a **single identity**. Putting two people's
photos in the same reference folder produces a gallery that matches both,
which silently makes identification meaningless and would let
`verify_match` confirm the wrong person.

Keep separate people in separate folders and separate gallery files:

```bash
# enrol subject A
python scripts/build_gallery.py --refs-dir data/sample_images/subject_a \
                               --out data/output/gallery_a.json

# enrol subject B
python scripts/build_gallery.py --refs-dir data/sample_images/subject_b \
                               --out data/output/gallery_b.json

# run the pipeline against whichever one you want
python scripts/find_match.py --gallery data/output/gallery_a.json
python scripts/scan_face.py  --gallery data/output/gallery_a.json
```

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
  legitimately returns nothing. This is the single biggest constraint on
  the pipeline: **a subject with no web-indexed photo cannot be found by
  any reverse image search, at any price, from any provider.** Tested
  directly during development — a private individual's selfies and even
  their public LinkedIn profile photo returned no matches, because
  LinkedIn serves profile images from a CDN that image crawlers don't
  index.
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

## Stage 3: Blockchain upload & verification

Takes the `content_hash` from Stage 2, writes it to a blockchain, and
then proves — later, from the chain alone — that the match record has not
changed since.

### Which blockchain

Any EVM chain that speaks JSON-RPC. Three ways to run it, in descending
order of "how impressive it looks" and ascending order of "how easily it
can go wrong on the day":

| Mode | What it is | Needs | Persistent? |
|---|---|---|---|
| **Local node** (default) | Ganache / Anvil / Hardhat on `localhost:8545` | Node.js | Yes, until you stop it |
| **Public testnet** | Sepolia or Polygon Amoy | Testnet funds from a faucet | Yes, forever, publicly |
| **Simulated** | in-process EVM (`eth-tester` + `py-evm`) | nothing | No — dies with the process |

All three run **the same compiled contract and the same code path**. The
demo recording uses a local Ganache node: it is a real EVM executing real
transactions, it survives between the anchor step and the verify step
(which is what makes the two-step proof meaningful), and it can't fail
because a faucet was dry or an RPC endpoint was rate-limiting.

Simulated mode exists so the project runs on a machine with nothing
installed. It is honest about what it is — it prints
`in-process simulated chain` and refuses to pretend otherwise.

Mainnet chains are **refused by default** (`MAINNET_CHAIN_IDS` in
`src/chain/registry.py`); a prototype has no business spending real money.

### The contract

`contracts/MatchRegistry.sol` — 90 lines, three functions.

```solidity
function anchor(bytes32 contentHash) external;
function verify(bytes32 contentHash) external view
    returns (bool exists, uint256 timestamp, address submitter);
function total() external view returns (uint256);
```

It stores a `bytes32` fingerprint, the block timestamp, and the submitting
address. That is all.

**No personal data ever goes on chain** — no image, no URL, no name, no
face embedding. This is deliberate, not an oversight: a blockchain is
permanent and public, so putting a real person's identifying data on one
would be irreversible. A hash gives the full tamper-evidence guarantee
with none of that exposure.

Re-anchoring the same hash **reverts** rather than overwriting, so the
first anchor timestamp is immutable. The Python layer treats that as
success rather than an error, since the original record still stands.

The compiled ABI and bytecode are committed at
`contracts/MatchRegistry.json` (solc 0.8.24, optimizer on, 200 runs), so
**no Solidity toolchain is needed to run this project**. Only run
`scripts/compile_contract.py` if you change the `.sol`.

### Setup

Everything comes from `requirements.txt`. For the local-node mode you
also need a chain to talk to:

```bash
# one-time, if you don't have it
npm install -g ganache

# leave this running in its own terminal for the whole demo
npx ganache --wallet.deterministic
```

`--wallet.deterministic` gives the same funded test accounts every time,
which makes a recording reproducible.

No `.env` is needed for a local node — the dev node's first unlocked
account signs, and no key is involved. `.env` is only for a public
testnet:

```
RPC_URL=https://rpc-amoy.polygon.technology
PRIVATE_KEY=0x...        # a TESTNET key, funded from a faucet
```

`.env` is git-ignored. Never put a key with real funds in it.

### Run

```bash
# 1. deploy the contract (once per chain)
python scripts/deploy_contract.py

# 2. anchor the Stage 2 match record
python scripts/anchor_record.py

# 3. verify it against the chain — and prove tampering is caught
python scripts/verify_onchain.py --tamper
```

The contract address is written to `data/output/deployment.json` and
picked up automatically, so there is nothing to copy-paste between steps.

Self-contained run with no node installed at all:

```bash
python scripts/anchor_record.py --simulated
```

That deploys, anchors, verifies and runs the tamper check inside one
process, because the chain cannot outlive it.

### How verification actually proves something

The important detail is that **the hash is recomputed, never read back**:

```
match_record.json
   -> hashed_payload()        (substantive fields only)
   -> canonical JSON          (sorted keys, fixed separators, UTF-8)
   -> SHA-256                 -> 0x47b6eee5...
   -> MatchRegistry.verify()  -> found? when? by whom?
```

`verify_onchain.py` takes only the *contract address* from the receipt
file. Every byte of the hash is derived from the record on disk at the
moment you run it. So:

- **Present on chain** → the record is byte-for-byte what was anchored.
- **Absent** → either it was never anchored, or a hashed field changed.

If the hash were read from the receipt instead of recomputed, the whole
exercise would prove nothing — the receipt could just be edited too.

### Tamper demonstration

`--tamper` changes exactly one character of `post_url` in memory and looks
the result up again:

```
original hash : 47b6eee5c5d0c3709a8226173c2c7006b32d8f6e1e083e58cad5898a7e9ed192
edited   hash : c64b9d97bd7f677150615089c84d6feece900a07f494d15823de870d0a0489da

[*] On-chain lookup of the edited record: NOT ON CHAIN
[+] Tamper detected.
```

Editing `data/output/match_record.json` on disk and re-running
`verify_onchain.py` produces the same outcome with exit code 2 — verified
during development, not just asserted here.

### What is and isn't in the hash

| In the hash (substantive claims) | Excluded (volatile) |
|---|---|
| `post_url`, `platform`, `page_title` | `discovered_at` timestamp |
| `candidate_image_url` | local file paths |
| `query_image_sha256`, `scan_image_sha256` | result counts |
| `identified_subject`, `identification_distance` | search engine / mode |
| `face_verified`, `face_distance` | |

Volatile fields are excluded so that re-saving the record on another
machine, at another time, from another folder still verifies. Substantive
fields are included so that changing *what was claimed* always breaks
verification. Both halves of that are covered by tests
(`test_volatile_fields_do_not_change_the_anchor`,
`test_tampering_with_a_record_breaks_verification`).

### Exit codes (Stage 3)

| Code | `verify_onchain.py` meaning |
|---|---|
| 0 | Verified on chain |
| 1 | Could not run — no node, no contract, no record, bad hash |
| 2 | Ran fine, record is **not** on chain (tampered, or never anchored) |

### Known limitations (Stage 3)

- **A local chain proves integrity, not public notarisation.** Anchoring
  to a node you control shows the mechanism works; it does not give the
  independent third-party timestamp that a public chain would. Pointing
  `RPC_URL` at Sepolia or Polygon Amoy gives that, with no code change —
  the only reason the demo doesn't is faucet/RPC reliability on the day.
- **Local nodes forget everything on restart.** Restart Ganache and the
  contract is gone; you must redeploy and re-anchor. The error message
  says so explicitly when it happens.
- **The chain proves the record didn't change — not that it was true.**
  If Stage 2 found the wrong post, Stage 3 will faithfully anchor that
  wrong post forever. Blockchain gives integrity, not correctness. This
  is a property of the technique, not a bug in this implementation.
- **Anchoring is public.** On a public chain the hash, timestamp and
  submitting address are visible to everyone. The hash reveals nothing
  about its input, but the *fact and time* of an anchoring is exposed.
- **Gas costs are real on a public chain** (~112k gas per anchor,
  ~200k to deploy). Negligible on a testnet, not free on mainnet — which
  is one more reason mainnet is refused by default.
- **No access control.** Anyone can anchor any hash to a deployed
  contract. For this prototype that's fine — the contract is a public
  timestamping service, and the `submitter` address records who did it.
  A production version would want an allowlist or a signature scheme.

---

## Running the whole pipeline end to end

Two terminals. The first just holds the chain:

```bash
# terminal 1 — leave running
npx ganache --wallet.deterministic
```

```bash
# terminal 2
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1

# Stage 1 — enrol the subject (once)
python scripts/build_gallery.py --refs-dir data/sample_images/public_figure \
                                --out data/output/gallery_public_figure.json

# Stage 1 — live face scan from the webcam
python scripts/scan_face.py --gallery data/output/gallery_public_figure.json

# Stage 2 — identify, search, filter, verify
python scripts/find_match.py --gallery data/output/gallery_public_figure.json

# Stage 3 — anchor and verify on chain
python scripts/deploy_contract.py
python scripts/anchor_record.py
python scripts/verify_onchain.py --tamper
```

### Suggested recording order

1. `npx ganache` in a visible terminal — the chain is real and running.
2. `scan_face.py` — webcam opens, capture the subject, gallery match prints.
3. `find_match.py` — browser opens, the search actually happens, a real
   post is found and ranked, the record hash is printed.
4. `deploy_contract.py` → `anchor_record.py` — transaction hash, block
   number, gas used.
5. `verify_onchain.py --tamper` — verified, then the tamper check fails
   the edited record. **This is the moment that proves the whole thing.**
6. Optional, and worth including: run `find_match.py` against a face with
   no web presence and let it exit 2. Showing the system reporting "found
   nothing" is stronger evidence that it isn't faking results than any
   number of successful runs.

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

| Suite | Covers |
|---|---|
| `test_face_encode.py` | Stage 1 encoding, distances, gallery matching |
| `test_search.py` | Stage 2 domain handling, ranking, canonical hashing |
| `test_chain.py` | Stage 3 against a real in-process EVM |

Stage 3's tests run against an actual EVM rather than a mock — a mocked
chain would happily "verify" anything and prove nothing.
