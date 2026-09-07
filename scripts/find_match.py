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
    ap.add_argument("--gallery", default=str(GALLERY_PATH),
                    help="gallery to identify against "
                         f"(default: {GALLERY_PATH.name})")
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

    gallery_path = Path(args.gallery)
    if gallery_path.exists():
        gallery = FaceGallery.load(gallery_path)
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
        print(f"[!] No gallery at {gallery_path} — cannot identify the scan.")
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
    web_srcs = [r for r in ranked if r.is_web_source]
    print(f"[*] {len(ranked)} unique page(s); {len(social)} on social "
          f"platforms; {len(web_srcs)} on attributable web sources.")

    if ranked:
        print("\n    Top results:")
        for i, r in enumerate(ranked[:args.top], 1):
            mark = "*" if r.is_social else " "
            label = r.platform or r.domain
            if r.via_cdn:
                label += " via CDN"
            print(f"    {mark} {i:2}. [{label}] {(r.title or r.url)[:64]}")
            print(f"           {r.url[:96]}")
            if r.cdn_url:
                print(f"           (post URL recovered from {r.domain} CDN link)")

    if not social:
        print("\n[!] No SOCIAL MEDIA post found for this image.")
        if web_srcs:
            # The image is demonstrably online — saying otherwise would be
            # false. Report exactly what was found and where.
            print(f"    The image WAS found online, on "
                  f"{len(web_srcs)} attributable source(s):")
            for r in web_srcs[: args.top]:
                print(f"      - [{r.web_source}] {r.url}")
            print("    These are real, checkable matches — but Wikimedia and "
                  "the like are\n    not social media, so they do not satisfy "
                  "Stage 2's requirement.")
        elif ranked:
            print(f"    The image was found on {len(ranked)} page(s), but "
                  "none of them are\n    social platforms or attributable "
                  "sources — mostly news sites and\n    image scrapers, which "
                  "carry no author to verify against.")
        else:
            print("    The search returned nothing at all: this image does "
                  "not appear to be\n    indexed anywhere the engine can see.")

        print("\n    To get a social hit, search with an image that was "
              "actually POSTED to\n    a social platform, rather than a press "
              "or encyclopedia photo. Or try\n    --engine google, or "
              "--manual to drive the search by hand.")

        # Still record the finding. "Image is public at these locations, no
        # social post" is a real, anchorable result, and Stage 3 should be
        # able to demonstrate on it.
        record = MatchRecord(
            query_image_path=str(query_image),
            query_image_sha256=query_hash,
            scan_image_sha256=scan_hash,
            identified_subject=identified_subject,
            identification_distance=identification_distance,
            post_url=web_srcs[0].url if web_srcs else "",
            platform=None,
            page_title=web_srcs[0].title if web_srcs else "",
            social_post_found=False,
            web_sources=[{"url": r.url, "source": r.web_source}
                         for r in web_srcs],
            search_engine=args.engine,
            search_mode="manual" if args.manual else "automatic",
            total_results_found=len(ranked),
            social_results_found=0,
            verification_note="no social media post found; "
                              f"{len(web_srcs)} attributable web source(s)",
        )
        record.save(RECORD_PATH)
        print(f"\n[+] Recorded the outcome anyway -> {RECORD_PATH}")
        print(f"    Content hash: {record.content_hash()}")
        print("    Stage 3 can anchor this: 'no social post found' is a real "
              "finding,\n    and anchoring it proves it wasn't quietly "
              "rewritten later.")
        return 2

    # -------------------------------------------------------------- 4. VERIFY
    best = social[0]
    banner("STEP 4/4", "verify the face on the page we found")

    platforms = sorted({r.platform for r in social if r.platform})
    if len(platforms) > 1:
        print(f"[*] Found on {len(platforms)} platforms: "
              f"{', '.join(platforms)}")
        print("    All of them are recorded; the highest-ranked one is "
              "verified below.")
    print(f"[*] Best social result: {best.summary}")
    print(f"    {best.url}")
    if best.via_cdn:
        if best.cdn_url:
            print(f"    Identified from a {best.platform} CDN image; the post "
                  "URL above was\n    reconstructed from it and is exact.")
        else:
            print(f"    Identified from a {best.platform} CDN image. The "
                  "platform is certain,\n    but the specific post URL is not "
                  "recoverable from a CDN link — the\n    URL above is the "
                  "image itself, recorded honestly as such.")

    verification = FaceVerification(note="verification skipped (--no-verify)")
    if not args.no_verify:
        if gallery is None:
            verification = FaceVerification(
                note="no gallery available to verify against")
            print(f"[!] {verification.note}")
        else:
            # A CDN link IS the matched image, so it is the best thing to
            # re-encode: no hotlink block, no HTML page to scrape.
            target = best.cdn_url or best.thumbnail_url or best.url
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
        platform_via_cdn=best.via_cdn,
        source_cdn_url=best.cdn_url,
        social_post_found=True,
        all_social_results=[
            {"url": r.url, "platform": r.platform, "via_cdn": r.via_cdn}
            for r in social
        ],
        web_sources=[{"url": r.url, "source": r.web_source} for r in web_srcs],
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
    if len(record.platforms_found) > 1:
        print(f"  Also found on: "
              f"{', '.join(p for p in record.platforms_found if p != record.platform)}"
              f"  ({len(record.canonical_social_results())} posts total)")
    print(f"  Face on page : {verification.summary}")
    print(f"  Content hash : {record.content_hash()}")
    print(f"\n[+] Saved -> {RECORD_PATH}")
    print("    That hash is what Stage 3 writes to the blockchain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
