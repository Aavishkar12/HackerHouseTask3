"""
record.py — the canonical hand-off record from Stage 2 to Stage 3.

Stage 3 writes a hash of the discovered post to a blockchain and must
later be able to re-verify it. That only works if the bytes being hashed
are reproducible: the same discovery must always serialise to exactly the
same bytes, or re-verification fails for reasons that have nothing to do
with tampering.

So this module defines one record type and one canonical serialisation:

  * keys sorted, fixed separators, UTF-8, no trailing whitespace
  * the volatile fields (timestamps, run-specific paths) live OUTSIDE
    the hashed payload

That last point is the important one. If `discovered_at` were inside the
hashed payload, re-running verification tomorrow would produce a
different hash and the on-chain check would fail even though nothing was
tampered with. The hashed payload contains only the facts about *what was
found*; everything else is metadata alongside it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Version history (each bump changes the hashed payload, so hashes across
# versions are not comparable — intended, since they assert different
# things). The version lives INSIDE the hashed payload and is read back
# from the file, so a record saved under an older version still hashes as
# that version and still verifies against its original anchor.
#   v1  original
#   v2  + all_social_results
#   v3  + social_post_found, web_sources
RECORD_VERSION = "stage2-match-record/v3"

# Cap on how many social hits go into the record. A broad search can
# return dozens; the record is evidence, not a crawl dump.
MAX_SOCIAL_RESULTS = 25


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's contents, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """
    Deterministic serialisation of the hashed payload.

    sort_keys makes dict ordering irrelevant; the compact separators and
    ensure_ascii=False fix the encoding so the same logical content always
    produces identical bytes on any machine.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def payload_hash(payload: Dict[str, Any]) -> str:
    """Hex SHA-256 of the canonical payload — what Stage 3 puts on-chain."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


@dataclass
class MatchRecord:
    """
    One complete Stage 2 discovery: what we searched with, what we found,
    and whether the face actually checked out.
    """

    # --- required: what we searched with, and what we found ---
    query_image_path: str
    query_image_sha256: str
    post_url: str

    # --- the face scan that triggered this, and who it was identified as ---
    # The scan and the query image are usually DIFFERENT files. A live
    # webcam frame identifies the subject against the enrolled gallery,
    # but can't be searched for on the web (it has never been published),
    # so the search runs against the enrolled reference photo instead.
    scan_image_sha256: str = ""
    identified_subject: Optional[str] = None
    identification_distance: Optional[float] = None

    # --- more about what the search found ---
    platform: Optional[str] = None
    page_title: str = ""
    candidate_image_url: str = ""

    # --- how it was found (provenance: proves the search was real) ---
    search_engine: str = ""
    search_mode: str = "automatic"          # automatic | manual
    total_results_found: int = 0
    social_results_found: int = 0

    # --- face verification against the Stage 1 gallery ---
    face_verified: Optional[bool] = None
    face_distance: Optional[float] = None
    face_tolerance: Optional[float] = None
    matched_reference: Optional[str] = None
    verification_note: str = ""

    # --- every social platform the search surfaced, not just the best ---
    # "found on YouTube and Pinterest" is a stronger, more complete claim
    # than "found on YouTube", so this IS hashed (see hashed_payload).
    # Each entry: {"url": ..., "platform": ..., "via_cdn": bool}.
    all_social_results: List[Dict[str, Any]] = field(default_factory=list)

    # Whether a genuine SOCIAL post was found. False means the image was
    # located online but not on a social platform — a real, honest outcome
    # that must be distinguishable from "found a post", so it is hashed.
    social_post_found: bool = False

    # Attributable non-social sources (Wikimedia, Wikipedia, Archive.org).
    # Evidence the image is genuinely public even when no social post
    # exists. Same shape as all_social_results, minus via_cdn.
    web_sources: List[Dict[str, Any]] = field(default_factory=list)

    # --- metadata, deliberately NOT hashed ---
    # How the platform was identified. True when it came from a media-CDN
    # host (e.g. i.ytimg.com) rather than a page URL. This is provenance —
    # the same class of thing as search_engine — so it is recorded in the
    # file but kept out of the hash: post_url and platform already carry
    # the substantive claim, and hashing "how we got there" would make the
    # same discovery hash differently across engines.
    platform_via_cdn: bool = False
    # The CDN image URL the post URL was reconstructed from, when it was.
    source_cdn_url: str = ""
    discovered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    record_version: str = RECORD_VERSION

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def hashed_payload(self) -> Dict[str, Any]:
        """
        The subset of fields that constitute the tamper-evident claim.

        Excluded on purpose:
          - local file paths       : differ per machine, so including them
            would make the same discovery hash differently elsewhere
          - discovered_at          : the blockchain supplies its own,
            trustworthy timestamp; ours would only be a claim
          - result counts / engine : provenance worth keeping in the file,
            but not part of "what was found"

        Included: the scan, who it was identified as, what was searched
        with, what was found, and whether the face on the found page
        actually checked out — i.e. the whole substantive claim:
        "this scan was identified as X, and X's face was found at this
        URL, with this much confidence".
        """
        return {
            "record_version": self.record_version,
            "scan_image_sha256": self.scan_image_sha256,
            "identified_subject": self.identified_subject,
            "identification_distance": self.identification_distance,
            "query_image_sha256": self.query_image_sha256,
            "post_url": self.post_url,
            "platform": self.platform,
            "page_title": self.page_title,
            "candidate_image_url": self.candidate_image_url,
            "face_verified": self.face_verified,
            "face_distance": self.face_distance,
            "all_social_results": self.canonical_social_results(),
            "social_post_found": bool(self.social_post_found),
            "web_sources": self.canonical_web_sources(),
        }

    @staticmethod
    def _canonical_list(items: Any, key_name: str,
                        include_cdn: bool) -> List[Dict[str, Any]]:
        """Shared canonicaliser: drop junk, dedupe by URL, sort, cap."""
        cleaned = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            entry: Dict[str, Any] = {"url": url,
                                     key_name: item.get(key_name)}
            if include_cdn:
                entry["via_cdn"] = bool(item.get("via_cdn", False))
            cleaned.append(entry)

        seen, unique = set(), []
        for item in cleaned:
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            unique.append(item)

        unique.sort(key=lambda d: (str(d[key_name] or ""), d["url"]))
        return unique[:MAX_SOCIAL_RESULTS]

    def canonical_web_sources(self) -> List[Dict[str, Any]]:
        """`web_sources` reduced to a stable, hashable form."""
        return self._canonical_list(self.web_sources, "source", False)

    def canonical_social_results(self) -> List[Dict[str, Any]]:
        """
        `all_social_results` reduced to a stable, hashable form.

        Sorted by (platform, url) and trimmed to the three fields that
        constitute the claim, so the hash does not depend on the order the
        engine happened to return results in, nor on any extra annotation
        added later. Capped at MAX_SOCIAL_RESULTS.
        """
        cleaned = []
        for item in self.all_social_results or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            cleaned.append({
                "url": url,
                "platform": item.get("platform"),
                "via_cdn": bool(item.get("via_cdn", False)),
            })

        # Deduplicate on URL, keeping first occurrence.
        seen, unique = set(), []
        for item in cleaned:
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            unique.append(item)

        unique.sort(key=lambda d: (str(d["platform"] or ""), d["url"]))
        return unique[:MAX_SOCIAL_RESULTS]

    @property
    def platforms_found(self) -> List[str]:
        """Distinct platform names, in stable order — for display."""
        names = {
            str(r["platform"]) for r in self.canonical_social_results()
            if r.get("platform")
        }
        return sorted(names)

    def content_hash(self) -> str:
        """Hex SHA-256 of the canonical hashed payload."""
        return payload_hash(self.hashed_payload())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["content_hash"] = self.content_hash()
        data["hashed_payload"] = self.hashed_payload()
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MatchRecord":
        """
        Load a saved record, ignoring the derived fields.

        Note this does NOT re-check the hash — Stage 3 should recompute
        `content_hash()` itself and compare, which is the whole point of
        the re-verification step.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("content_hash", None)
        data.pop("hashed_payload", None)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
