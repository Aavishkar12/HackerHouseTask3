"""
Tests for Stage 2 (web/social search).

Everything here runs offline. The live search is inherently
network-dependent and can't be unit-tested meaningfully, so the parts
that CAN break silently — URL classification, ranking, and above all the
canonical hashing that Stage 3 depends on — are tested directly.

Run with:  pytest tests/test_search.py
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from search.record import (  # noqa: E402
    MatchRecord,
    canonical_bytes,
    payload_hash,
    sha256_file,
)
from search.social_filter import (  # noqa: E402
    best_social_result,
    extract_domain,
    is_low_value_domain,
    is_social_media_url,
    match_social_domain,
    rank_results,
)


# --- domain handling --------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://www.instagram.com/p/A/", "instagram.com"),
    ("https://m.facebook.com/photo", "facebook.com"),
    ("https://IN.LINKEDIN.COM:443/in/x", "linkedin.com"),
    ("http://x.com/u/status/1?s=20", "x.com"),
])
def test_extract_domain_normalises(url, expected):
    assert extract_domain(url) == expected


@pytest.mark.parametrize("bad", [
    "", "not a url", "javascript:alert(1)", "ftp://f.example.com/a.jpg", None,
])
def test_extract_domain_rejects_non_http(bad):
    assert extract_domain(bad) == ""


def test_subdomains_still_match_platform():
    # Regional LinkedIn subdomains are extremely common in search results.
    assert match_social_domain("https://uk.linkedin.com/in/someone") == "LinkedIn"
    assert match_social_domain("https://in.linkedin.com/in/someone") == "LinkedIn"


def test_lookalike_domain_is_not_treated_as_social():
    """
    'instagram.com.evil.net' must NOT count as Instagram — suffix matching
    has to be anchored on a dot boundary, not a substring.
    """
    assert not is_social_media_url("https://instagram.com.evil.net/p/1")
    assert not is_social_media_url("https://notinstagram.com/p/1")


def test_aggregators_flagged_low_value():
    assert is_low_value_domain("https://www.pinterest.com/pin/1/")
    assert not is_low_value_domain("https://www.instagram.com/p/1/")


# --- ranking ----------------------------------------------------------

def test_real_social_outranks_aggregator_and_random_pages():
    raw = [
        {"url": "https://blog.example.com/x"},
        {"url": "https://www.pinterest.com/pin/1/"},
        {"url": "https://www.instagram.com/p/REAL/"},
    ]
    ranked = rank_results(raw)
    assert ranked[0].platform == "Instagram"
    assert ranked[-1].is_social is False


def test_duplicate_urls_are_merged():
    raw = [
        {"url": "https://www.instagram.com/p/REAL/"},
        {"url": "https://www.instagram.com/p/REAL"},        # no trailing slash
        {"url": "https://www.instagram.com/p/REAL/#frag"},   # fragment
    ]
    assert len(rank_results(raw)) == 1


def test_unparseable_results_are_dropped_not_crashed():
    ranked = rank_results([{"url": "garbage"}, {"url": ""}, {"no_url": 1}])
    assert ranked == []


def test_best_social_result_returns_none_when_nothing_social():
    assert best_social_result([{"url": "https://news.example.com/a"}]) is None


# --- canonical hashing (Stage 3 depends on this) ----------------------

def test_canonical_bytes_ignore_key_order():
    a = {"b": 2, "a": 1, "c": {"y": 1, "x": 2}}
    b = {"a": 1, "c": {"x": 2, "y": 1}, "b": 2}
    assert canonical_bytes(a) == canonical_bytes(b)
    assert payload_hash(a) == payload_hash(b)


def test_hash_changes_when_content_changes():
    assert payload_hash({"post_url": "a"}) != payload_hash({"post_url": "b"})


def _record(**overrides) -> MatchRecord:
    base = dict(
        query_image_path="data/output/scan_1.jpg",
        query_image_sha256="a" * 64,
        post_url="https://www.instagram.com/p/REAL/",
        platform="Instagram",
        page_title="A post",
        candidate_image_url="https://cdn.example.com/i.jpg",
        scan_image_sha256="b" * 64,
        identified_subject="ref00_original",
        identification_distance=0.21,
        face_verified=True,
        face_distance=0.34,
    )
    base.update(overrides)
    return MatchRecord(**base)


def test_content_hash_is_stable_across_runs():
    """
    The same discovery must hash identically every time, or Stage 3's
    re-verification would fail for no reason.
    """
    assert _record().content_hash() == _record().content_hash()


def test_timestamp_and_local_path_do_not_affect_the_hash():
    """
    This is the property that makes on-chain re-verification work: the
    hash must depend on WHAT was found, not WHEN it was found or where
    the file happened to sit on disk.
    """
    r1 = _record(discovered_at="2026-01-01T00:00:00+00:00",
                 query_image_path="/tmp/a.jpg")
    r2 = _record(discovered_at="2099-12-31T23:59:59+00:00",
                 query_image_path="C:/somewhere/else/b.jpg")
    assert r1.content_hash() == r2.content_hash()


def test_provenance_fields_do_not_affect_the_hash():
    # How many results the engine happened to return varies run to run.
    r1 = _record(total_results_found=5, search_mode="automatic")
    r2 = _record(total_results_found=99, search_mode="manual")
    assert r1.content_hash() == r2.content_hash()


def test_changing_the_found_post_changes_the_hash():
    r1 = _record()
    r2 = _record(post_url="https://www.instagram.com/p/DIFFERENT/")
    assert r1.content_hash() != r2.content_hash()


def test_changing_the_face_verdict_changes_the_hash():
    assert _record(face_verified=True).content_hash() != \
           _record(face_verified=False).content_hash()


def test_save_load_roundtrip_preserves_hash(tmp_path):
    original = _record()
    path = tmp_path / "match_record.json"
    original.save(path)

    reloaded = MatchRecord.load(path)
    assert reloaded.post_url == original.post_url
    assert reloaded.content_hash() == original.content_hash()


def test_saved_file_contains_the_hash_and_payload(tmp_path):
    import json
    path = tmp_path / "r.json"
    rec = _record()
    rec.save(path)
    data = json.loads(path.read_text())
    assert data["content_hash"] == rec.content_hash()
    assert data["hashed_payload"]["post_url"] == rec.post_url
    # metadata present in the file, absent from the hashed payload
    assert "discovered_at" in data
    assert "discovered_at" not in data["hashed_payload"]


def test_sha256_file_matches_known_value(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    # well-known SHA-256 of b"hello"
    assert sha256_file(p) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_unicode_in_title_hashes_consistently():
    """Titles from real social posts routinely contain emoji/non-ASCII."""
    r1 = _record(page_title="Aavishkar 🎉 फोटो")
    r2 = _record(page_title="Aavishkar 🎉 फोटो")
    assert r1.content_hash() == r2.content_hash()
    assert r1.content_hash() != _record(page_title="different").content_hash()


# --- scan vs query separation (added when the pipeline split the two) ---

def test_identification_is_part_of_the_hashed_claim():
    """
    Who the scan was identified as is part of what goes on-chain — if it
    changes, the hash must change too.
    """
    assert _record(identified_subject="alice").content_hash() != \
           _record(identified_subject="bob").content_hash()


def test_scan_hash_is_part_of_the_claim():
    assert _record(scan_image_sha256="c" * 64).content_hash() != \
           _record(scan_image_sha256="d" * 64).content_hash()


def test_unidentified_scan_still_produces_a_valid_record():
    """
    A scan that matches nobody is a legitimate outcome, not a crash — the
    record just carries identified_subject=None.
    """
    rec = _record(identified_subject=None, identification_distance=None)
    assert rec.content_hash()
    assert rec.hashed_payload()["identified_subject"] is None


def test_gallery_source_tracking_roundtrip(tmp_path):
    """Stage 2 needs to recover the enrolled photo behind a match."""
    import numpy as np
    from faceid.face_encode import FaceGallery

    g = FaceGallery(
        labels=["ref00", "ref01"],
        encodings=[np.random.rand(512), np.random.rand(512)],
        sources=["data/refs/ref00.jpg", "data/refs/ref01.jpg"],
    )
    p = tmp_path / "g.json"
    g.save(p)
    loaded = FaceGallery.load(p)
    assert loaded.sources == g.sources
    assert loaded.source_for("ref01") == "data/refs/ref01.jpg"
    assert loaded.source_for("nonexistent") is None


def test_gallery_without_sources_still_loads(tmp_path):
    """Galleries saved before source tracking existed must not break."""
    import json
    import numpy as np
    from faceid.face_encode import FaceGallery, DEFAULT_MODEL

    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "model": DEFAULT_MODEL,
        "metric": "cosine",
        "labels": ["ref00"],
        "encodings": [np.random.rand(512).tolist()],
    }))
    g = FaceGallery.load(p)
    assert g.sources == []
    assert g.source_for("ref00") is None      # degrades, doesn't crash
