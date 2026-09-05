#!/usr/bin/env python3
"""
find_match.py — Stage 2: find a real social media post showing this face.

Pipeline:
    face image -> reverse image search -> filter for social posts
               -> re-verify the face against the Stage 1 gallery
               -> write a canonical record for Stage 3 to hash

The search is genuine. Nothing is hardcoded: whatever the search engine
returns is what gets filtered and reported. An empty result is a real,
reportable outcome, not a failure to paper over.

Usage
-----
    # search using the most recent webcam scan
    python scripts/find_match.py

    # search using a specific image
    python scripts/find_match.py data/sample_images/me.jpg

    # if automation gets CAPTCHA'd, do the upload by hand
    python scripts/find_match.py --manual

    # try Google instead of Yandex
    python scripts/find_match.py --engine google

Output
------
    data/output/match_record.json   (consumed by Stage 3)

Requires a real display + browser:
    pip install playwright && playwright install chromium
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from faceid.face_encode import DEFAULT_TOLERANCE, FaceGallery  # noqa: E402
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
DEFAULT_IMAGE = REPO_ROOT / "data" / "sample_images" / "me.jpg"


def pick_query_image(explicit: str | None) -> Path | None:
    """
    Choose the image to search with.

    Preference order: an explicit path, then the most recent webcam scan
    from scan_face.py, then the default sample photo.
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    scans = sorted(
        OUTPUT_DIR.glob("scan_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if scans:
        print(f"[*] Using most recent webcam scan: {scans[0].name}")
        return scans[0]

    if DEFAULT_IMAGE.exists():
        print(f"[*] No webcam scan found; using {DEFAULT_IMAGE.name}")
        return DEFAULT_IMAGE
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="face image to search with")
    ap.add_argument("--engine", default="yandex", choices=["yandex", "google"],
                    help="reverse image search engine (default: yandex)")
    ap.add_argument("--manual", action="store_true",
                    help="open the browser and upload by hand, then scrape")
    ap.add_argument("--headless", action="store_true",
                    help="hide the browser window (more bot-detectable)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip re-checking the face on the found page")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help=f"face match tolerance (default {DEFAULT_TOLERANCE})")
    ap.add_argument("--top", type=int, default=10,
                    help="how many ranked results to display (default 10)")
    args = ap.parse_args()

    # ---------------------------------------------------------------- input
    query_image = pick_query_image(args.image)
    if query_image is None:
        print("[!] No query image found.")
        print("    Pass one explicitly, run scripts/scan_face.py first, or "
              f"put a photo at {DEFAULT_IMAGE}")
        return 1

    print(f"[*] Query image: {query_image}")
    query_hash = sha256_file(query_image)
    print(f"[*] SHA-256: {query_hash}")

    # -------------------------------------------------------------- search
    print(f"\n{'=' * 66}\n  STEP 1/3 — reverse image search ({args.engine})\n{'=' * 66}")
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

    # -------------------------------------------------------------- filter
    print(f"\n{'=' * 66}\n  STEP 2/3 — filter for social media posts\n{'=' * 66}")
    ranked = rank_results(r.to_dict() for r in raw_results)
    social = [r for r in ranked if r.is_social]

    print(f"[*] {len(ranked)} unique page(s), {len(social)} on social platforms.")
    if ranked:
        print("\n    Top results:")
        for i, r in enumerate(ranked[:args.top], 1):
            tag = r.platform or r.domain
            mark = "*" if r.is_social else " "
            print(f"    {mark} {i:2}. [{tag}] {(r.title or r.url)[:72]}")
            print(f"           {r.url[:96]}")

    if not social:
        print("\n[!] No social media post found for this image.")
        print("    This is a real result, not a bug — it means the photo "
              "isn't indexed on any social platform the engine can see.")
        print("    Options: use a photo that IS public online (e.g. your "
              "LinkedIn profile picture), or try --engine google / --manual.")
        return 2

    # -------------------------------------------------------------- verify
    best = social[0]
    print(f"\n{'=' * 66}\n  STEP 3/3 — verify the face on the found page\n{'=' * 66}")
    print(f"[*] Best social result: {best.summary}")
    print(f"    {best.url}")

    verification = FaceVerification(note="verification skipped (--no-verify)")
    if not args.no_verify:
        if not GALLERY_PATH.exists():
            verification = FaceVerification(
                note=f"no gallery at {GALLERY_PATH}; run build_gallery.py"
            )
            print(f"[!] {verification.note}")
        else:
            gallery = FaceGallery.load(GALLERY_PATH)
            target = best.thumbnail_url or best.url
            print(f"[*] Re-encoding candidate image against your "
                  f"{len(gallery)}-photo gallery ...")
            verification = verify_image_against_gallery(
                target, gallery, tolerance=args.tolerance
            )
            print(f"[*] {verification.summary}")
            if verification.attempted and verification.verified is False:
                print("    (the search matched this page, but the face on it "
                      "did not match yours — treat with caution)")
            elif not verification.attempted:
                print("    (couldn't fetch the image to check — most social "
                      "platforms block hotlinking; the search hit itself "
                      "still stands)")

    # -------------------------------------------------------------- record
    record = MatchRecord(
        query_image_path=str(query_image),
        query_image_sha256=query_hash,
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

    print(f"\n{'=' * 66}\n  RESULT\n{'=' * 66}")
    print(f"  Post      : {record.post_url}")
    print(f"  Platform  : {record.platform}")
    print(f"  Face check: {verification.summary}")
    print(f"  Hash      : {record.content_hash()}")
    print(f"\n[+] Saved -> {RECORD_PATH}")
    print("    This hash is what Stage 3 writes to the blockchain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
