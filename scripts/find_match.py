#!/usr/bin/env python3
"""
find_match.py — Stage 2: identify the scanned face, then find a real
social media post showing that person.

Pipeline
--------
    1. IDENTIFY  scan image  -> encode -> match against enrolled gallery
    2. SEARCH    reverse image search using the enrolled reference photo
    3. FILTER    keep genuine social media posts, rank them
    4. VERIFY    re-check the face on the found page against the gallery
    -> writes a canonical record for Stage 3 to hash and put on-chain

Why the scan and the search image are different files
-----------------------------------------------------
A live webcam frame has never existed on the internet, so a reverse image
search can never find it — that's a property of reverse image search, not
a limitation of this code. So the two images play different roles:

    scan image   identifies WHO is in front of the camera, by matching
                 against the enrolled gallery (this is the face-ID step)
    query image  is what actually gets searched for on the web — the
                 enrolled reference photo of the person just identified

The search itself is completely genuine. Nothing is hardcoded: whatever
the search engine returns is what gets filtered, verified and reported,
and finding nothing is a real, reported outcome.

Usage
-----
    # full flow: newest webcam scan identifies the subject, and the
    # matching enrolled photo is used for the search
    python scripts/find_match.py

    # scan explicitly
    python scripts/find_match.py --scan data/output/scan_2026.jpg

    # override what gets searched for
    python scripts/find_match.py --query-image data/sample_images/refs/ref00.jpg

    # if automation gets CAPTCHA'd, upload by hand and let the script scrape
    python scripts/find_match.py --manual

Output
------
    data/output/match_record.json

Requires a real display + browser:
    pip install playwright && playwright install chromium
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from faceid.face_encode import (  # noqa: E402
    DEFAULT_TOLERANCE,
    FaceEncodingError,
    FaceGallery,
    detect_faces,
)
from search.record import MatchRecord, sha256_file  # noqa: E402
from search.reverse_search import (  # noqa: E402
    ReverseSearchError,
    reverse_image_search,
)
from search.social_filter import rank_results  # noqa: E402
from search.verify_match import (  # noqa: E402
    FaceVerification,
    verify_image_against_gallery,
)

OUTPUT_DIR = REPO_ROOT / "data" / "output"
GALLERY_PATH = OUTPUT_DIR / "gallery.json"
RECORD_PATH = OUTPUT_DIR / "match_record.json"
DEFAULT_SCAN = REPO_ROOT / "data" / "sample_images" / "me.jpg"

BAR = "=" * 68


def banner(step: str, title: str) -> None:
    print(f"\n{BAR}\n  {step} — {title}\n{BAR}")


def pick_scan_image(explicit: str | None) -> Path | None:
    """Explicit path, else the newest webcam scan, else the sample photo."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    scans = sorted(
        OUTPUT_DIR.glob("scan_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if scans:
        print(f"[*] Using newest webcam scan: {scans[0].name}")
        return scans[0]
    if DEFAULT_SCAN.exists():
        print(f"[*] No webcam scan found; falling back to {DEFAULT_SCAN.name}")
        return DEFAULT_SCAN
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2: identify a scanned face and find a matching "
                    "social media post.")
    ap.add_argument("scan", nargs="?",
                    help="face scan image (default: newest webcam scan)")
    ap.add_argument("--scan", dest="scan_flag",
                    help="same as the positional argument")
    ap.add_argument("--query-image",
                    help="image to actually search the web with. Default: "
                         "the enrolled photo of whoever the scan matched.")
    ap.add_argument("--engine", default="yandex", choices=["yandex", "google"])
    ap.add_argument("--manual", action="store_true",
                    help="open the browser and upload by hand, then scrape")
    ap.add_argument("--headless", action="store_true",
                    help="hide the browser window (more bot-detectable)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip re-checking the face on the found page")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--top", type=int, default=10,
                    help="how many ranked results to display")
    args = ap.parse_args()

    # ------------------------------------------------------------ 1. IDENTIFY
    banner("STEP 1/4", "identify the scanned face")

    scan_image = pick_scan_image(args.scan_flag or args.scan)
    if scan_image is None:
        print("[!] No scan image found.")
        print("    Run scripts/scan_face.py first, or pass an image path.")
        return 1

    print(f"[*] Scan image: {scan_image}")
    scan_hash = sha256_file(scan_image)
    print(f"    sha256: {scan_hash}")

    gallery = None
    identified_subject = None
    identification_distance = None
    query_image: Path | None = None

    if GALLERY_PATH.exists():
        gallery = FaceGallery.load(GALLERY_PATH)
        try:
            # A scan can legitimately catch more than one face — someone
            # walking past the camera, or a group photo held up to it. Use
            # the largest (closest to camera) rather than aborting, which
            # is the same rule build_gallery.py and verify_match.py apply.
            faces = detect_faces(scan_image)
            if not faces:
                print(f"[!] No face detected in {scan_image.name}.")
                print("    Re-scan with better lighting, facing the camera.")
                return 1
            if len(faces) > 1:
                print(f"[*] {len(faces)} faces in the scan — using the "
                      "largest (closest to camera).")
            faces.sort(key=lambda m: (m.location[2] - m.location[0]),
                       reverse=True)
            scan_encoding = faces[0].encoding
        except FaceEncodingError as e:
            print(f"[!] Could not encode the scan: {e}")
            return 1

        is_match, distance, ref_label = gallery.match(
            scan_encoding, tolerance=args.tolerance)
        identification_distance = round(float(distance), 6)

        if is_match:
            identified_subject = ref_label
            print(f"[+] Identified as: {ref_label}  "
                  f"(distance {distance:.4f}, tolerance {args.tolerance})")
            source = gallery.source_for(ref_label)
            if source and Path(source).exists():
                query_image = Path(source)
            elif source:
                print(f"[!] Enrolled photo '{source}' no longer exists on disk.")
        else:
            print(f"[!] Scan did NOT match any enrolled subject "
                  f"(closest: {ref_label} at {distance:.4f}, "
                  f"tolerance {args.tolerance}).")
            print("    Continuing, but the identification is unconfirmed — "
                  "this will be recorded honestly.")
    else:
        print(f"[!] No gallery at {GALLERY_PATH} — cannot identify the scan.")
        print("    Run scripts/build_gallery.py to enroll a subject first.")

    # Decide what actually gets searched for.
    if args.query_image:
        query_image = Path(args.query_image)
        if not query_image.exists():
            print(f"[!] --query-image not found: {query_image}")
            return 1
        print(f"[*] Query image (explicit): {query_image}")
    elif query_image is not None:
        print(f"[*] Query image (enrolled photo of {identified_subject}): "
              f"{query_image.name}")
    else:
        query_image = scan_image
        print("[*] Query image: falling back to the scan itself.")
        print("    NOTE: a live webcam frame has no web presence, so this "
              "will almost certainly find nothing.")
        print("    Use --query-image, or enrol the subject with "
              "build_gallery.py so an indexed photo can be used.")

    query_hash = sha256_file(query_image)
    print(f"    sha256: {query_hash}")

    # -------------------------------------------------------------- 2. SEARCH
    banner("STEP 2/4", f"reverse image search ({args.engine})")
    try:
        raw_results = reverse_image_search(
            query_image,
            engine=args.engine,
            manual=args.manual,
            headless=args.headless,
        )
    except ReverseSearchError as e:
        print(f"\n[!] Search failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[!] Unexpected search error: {type(e).__name__}: {e}")
        print("    Try --manual to complete the upload by hand.")
        return 1

    # -------------------------------------------------------------- 3. FILTER
    banner("STEP 3/4", "filter for social media posts")
    ranked = rank_results(r.to_dict() for r in raw_results)
    social = [r for r in ranked if r.is_social]
    print(f"[*] {len(ranked)} unique page(s); {len(social)} on social platforms.")

    if ranked:
        print("\n    Top results:")
        for i, r in enumerate(ranked[:args.top], 1):
            mark = "*" if r.is_social else " "
            print(f"    {mark} {i:2}. [{r.platform or r.domain}] "
                  f"{(r.title or r.url)[:68]}")
            print(f"           {r.url[:96]}")

    if not social:
        print("\n[!] No social media post found for this image.")
        print("    This is a real result, not a bug: the image isn't indexed "
              "on any social platform the engine can see.")
        print("    Use a photo that IS public online, or try --engine google "
              "/ --manual.")
        return 2

    # -------------------------------------------------------------- 4. VERIFY
    best = social[0]
    banner("STEP 4/4", "verify the face on the page we found")
    print(f"[*] Best social result: {best.summary}")
    print(f"    {best.url}")

    verification = FaceVerification(note="verification skipped (--no-verify)")
    if not args.no_verify:
        if gallery is None:
            verification = FaceVerification(
                note="no gallery available to verify against")
            print(f"[!] {verification.note}")
        else:
            target = best.thumbnail_url or best.url
            print(f"[*] Re-encoding the candidate image against the "
                  f"{len(gallery)}-photo gallery ...")
            verification = verify_image_against_gallery(
                target, gallery, tolerance=args.tolerance)
            print(f"[*] {verification.summary}")
            if verification.attempted and verification.verified is False:
                print("    (the search matched this page, but the face on it "
                      "does not match the gallery — treat with caution)")
            elif not verification.attempted:
                print("    (couldn't fetch the image to check — most social "
                      "platforms block hotlinking; the search hit itself "
                      "still stands)")

    # -------------------------------------------------------------- record
    record = MatchRecord(
        query_image_path=str(query_image),
        query_image_sha256=query_hash,
        scan_image_sha256=scan_hash,
        identified_subject=identified_subject,
        identification_distance=identification_distance,
        post_url=best.url,
        platform=best.platform,
        page_title=best.title,
        candidate_image_url=best.thumbnail_url,
        search_engine=args.engine,
        search_mode="manual" if args.manual else "automatic",
        total_results_found=len(ranked),
        social_results_found=len(social),
        face_verified=verification.verified,
        face_distance=verification.distance,
        face_tolerance=verification.tolerance if verification.attempted else None,
        matched_reference=verification.matched_reference,
        verification_note=verification.note,
    )
    record.save(RECORD_PATH)

    print(f"\n{BAR}\n  RESULT\n{BAR}")
    print(f"  Scanned face : identified as "
          f"{identified_subject or 'UNKNOWN'}"
          + (f" (distance {identification_distance:.4f})"
             if identification_distance is not None else ""))
    print(f"  Post found   : {record.post_url}")
    print(f"  Platform     : {record.platform}")
    print(f"  Face on page : {verification.summary}")
    print(f"  Content hash : {record.content_hash()}")
    print(f"\n[+] Saved -> {RECORD_PATH}")
    print("    That hash is what Stage 3 writes to the blockchain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
