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


# ---------------------------------------------------------------------------
# Media-CDN handling
#
# Reverse image search usually returns the CDN URL of the matched image, not
# the page it appeared on. Before this was handled, genuine social hits were
# scored as ordinary web noise and thrown away — find_match.py reported "no
# social media post found" while a real YouTube post sat in the results.
# ---------------------------------------------------------------------------

from search.social_filter import (  # noqa: E402
    canonical_url_from_cdn,
    match_media_cdn,
)


@pytest.mark.parametrize("url,expected", [
    ("https://i.ytimg.com/vi/uNJO_n_-510/oardefault.jpg", "YouTube"),
    ("https://i.pinimg.com/originals/2f/82/e9/abc.jpg", "Pinterest"),
    ("https://scontent.cdninstagram.com/v/t51/123_n.jpg", "Instagram"),
    ("https://pbs.twimg.com/media/Abc123.jpg", "Twitter/X"),
    ("https://media.licdn.com/dms/image/C4D/profile.jpg", "LinkedIn"),
    ("https://preview.redd.it/abc123.jpg", "Reddit"),
    ("https://live.staticflickr.com/65535/123_abc.jpg", "Flickr"),
])
def test_media_cdn_hosts_are_recognised(url, expected):
    assert match_media_cdn(url) == expected


def test_fbcdn_splits_facebook_from_instagram():
    # fbcdn.net serves both; only Instagram's variants name themselves.
    assert match_media_cdn("https://scontent.fbom1-1.fna.fbcdn.net/v/x.jpg") \
        == "Facebook"
    assert match_media_cdn(
        "https://instagram.fbom1-1.fna.fbcdn.net/v/x.jpg") == "Instagram"


@pytest.mark.parametrize("url", [
    "https://assets.telegraphindia.com/abp/2025/Jul/salman.jpg",
    "https://images.news18.com/ibnlive/uploads/2025/10/x.jpg",
    "https://example.com/photo.jpg",
    "",
    "not-a-url",
])
def test_non_platform_cdns_are_not_social(url):
    assert match_media_cdn(url) is None


def test_youtube_watch_url_is_recovered_from_thumbnail():
    url = ("https://i.ytimg.com/vi/uNJO_n_-510/oardefault.jpg"
           "?sqp=-oaymwEkCJUDENAFSFqQAgHyq4qpAxMIARUAAAAAJQ")
    assert canonical_url_from_cdn(url) == \
        "https://www.youtube.com/watch?v=uNJO_n_-510"


def test_youtube_webp_thumbnail_also_recovers():
    assert canonical_url_from_cdn(
        "https://i.ytimg.com/vi_webp/dQw4w9WgXcQ/hq720.webp") == \
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    # No post id is recoverable from these — guessing one would be fabrication.
    "https://i.pinimg.com/originals/2f/82/e9/abc.jpg",
    "https://scontent.cdninstagram.com/v/t51/123_n.jpg",
    "https://pbs.twimg.com/media/Abc123.jpg",
    "https://i.ytimg.com/an_webp/notavalidid/x.webp",   # id wrong length
    "https://example.com/vi/uNJO_n_-510/x.jpg",         # right shape, wrong host
])
def test_no_canonical_url_is_invented(url):
    assert canonical_url_from_cdn(url) is None


def test_the_exact_run_that_reported_no_match_now_finds_the_post():
    """Regression: these are the four results from the failing run."""
    raw = [
        {"url": "https://i.ytimg.com/vi/uNJO_n_-510/oardefault.jpg?sqp=-oaymw",
         "title": "#happybirthday #salmankhan #fyp - YouTube"},
        {"url": "https://assets.telegraphindia.com/abp/2025/Jul/salman.jpg",
         "title": "Payel-Dwipayan Bengali Serial actress"},
        {"url": "https://i.pinimg.com/originals/2f/82/e9/abc.jpg",
         "title": "210 Salman Khan in Jacket ideas"},
        {"url": "https://images.news18.com/ibnlive/uploads/2025/10/x.jpg",
         "title": "Why A Furious Danny Denzongpa Kept Rejecting Movies"},
    ]
    best = best_social_result(raw)
    assert best is not None, "a real YouTube post was in these results"
    assert best.platform == "YouTube"
    assert best.url == "https://www.youtube.com/watch?v=uNJO_n_-510"
    assert best.via_cdn is True
    assert best.cdn_url.startswith("https://i.ytimg.com/")


def test_recovered_post_outranks_cdn_only_and_news():
    raw = [
        {"url": "https://images.news18.com/x.jpg"},
        {"url": "https://i.pinimg.com/originals/2f/abc.jpg"},
        {"url": "https://i.ytimg.com/vi/uNJO_n_-510/default.jpg"},
    ]
    ranked = rank_results(raw)
    assert ranked[0].platform == "YouTube"      # recovered post URL wins
    assert ranked[0].rank_score == 4
    assert ranked[1].platform == "Pinterest"    # CDN-only, still social
    assert ranked[1].rank_score == 2
    assert ranked[2].is_social is False         # news site


def test_a_real_page_url_still_outranks_any_cdn_hit():
    raw = [
        {"url": "https://i.ytimg.com/vi/uNJO_n_-510/default.jpg"},
        {"url": "https://www.instagram.com/p/REAL123/"},
    ]
    ranked = rank_results(raw)
    assert ranked[0].url == "https://www.instagram.com/p/REAL123/"
    assert ranked[0].rank_score == 5
    assert ranked[0].via_cdn is False


def test_cdn_rewrite_does_not_corrupt_plain_results():
    """A normal social page must pass through completely untouched."""
    raw = [{"url": "https://twitter.com/someone/status/123",
            "title": "hello", "engine": "yandex"}]
    r = rank_results(raw)[0]
    assert r.url == "https://twitter.com/someone/status/123"
    assert r.platform == "Twitter/X"
    assert r.via_cdn is False
    assert r.cdn_url == ""
    assert r.title == "hello"
    assert r.engine == "yandex"


def test_cdn_provenance_does_not_change_the_content_hash():
    """platform_via_cdn is provenance, not a claim — it must stay out of
    the hash, or the same discovery would anchor differently per engine."""
    base = _record(post_url="https://www.youtube.com/watch?v=uNJO_n_-510")
    via_cdn = _record(
        post_url="https://www.youtube.com/watch?v=uNJO_n_-510",
        platform_via_cdn=True,
        source_cdn_url="https://i.ytimg.com/vi/uNJO_n_-510/default.jpg",
    )
    assert via_cdn.content_hash() == base.content_hash()


def test_changing_the_recovered_post_url_still_breaks_the_hash():
    """The URL itself is a substantive claim and must remain hashed."""
    a = _record(post_url="https://www.youtube.com/watch?v=uNJO_n_-510")
    b = _record(post_url="https://www.youtube.com/watch?v=AAAAAAAAAAA")
    assert a.content_hash() != b.content_hash()


# ---------------------------------------------------------------------------
# all_social_results — recording every platform, not just the winner
# ---------------------------------------------------------------------------

def test_all_social_results_is_hashed():
    """'Found on YouTube and Pinterest' is a stronger claim than 'found on
    YouTube' — adding a platform must change the hash."""
    one = _record(all_social_results=[
        {"url": "https://www.youtube.com/watch?v=uNJO_n_-510",
         "platform": "YouTube", "via_cdn": True},
    ])
    two = _record(all_social_results=[
        {"url": "https://www.youtube.com/watch?v=uNJO_n_-510",
         "platform": "YouTube", "via_cdn": True},
        {"url": "https://www.instagram.com/p/ABC123/",
         "platform": "Instagram", "via_cdn": False},
    ])
    assert one.content_hash() != two.content_hash()


def test_social_result_order_does_not_change_the_hash():
    """Engines return results in arbitrary order; the same finding must
    hash identically regardless."""
    a = [{"url": "https://www.instagram.com/p/A/", "platform": "Instagram"},
         {"url": "https://www.youtube.com/watch?v=uNJO_n_-510",
          "platform": "YouTube"}]
    assert _record(all_social_results=a).content_hash() == \
        _record(all_social_results=list(reversed(a))).content_hash()


def test_duplicate_social_results_are_collapsed():
    dupes = _record(all_social_results=[
        {"url": "https://www.instagram.com/p/A/", "platform": "Instagram"},
        {"url": "https://www.instagram.com/p/A/", "platform": "Instagram"},
    ])
    once = _record(all_social_results=[
        {"url": "https://www.instagram.com/p/A/", "platform": "Instagram"},
    ])
    assert len(dupes.canonical_social_results()) == 1
    assert dupes.content_hash() == once.content_hash()


def test_social_results_are_capped():
    from search.record import MAX_SOCIAL_RESULTS
    many = _record(all_social_results=[
        {"url": f"https://www.instagram.com/p/POST{i}/", "platform": "Instagram"}
        for i in range(MAX_SOCIAL_RESULTS + 20)
    ])
    assert len(many.canonical_social_results()) == MAX_SOCIAL_RESULTS


def test_malformed_social_entries_are_dropped_not_crashed():
    r = _record(all_social_results=[
        {"url": "https://www.instagram.com/p/A/", "platform": "Instagram"},
        {"url": "", "platform": "Instagram"},      # no url
        {"platform": "YouTube"},                   # missing url entirely
        "not-a-dict",                              # wrong type
        None,
    ])
    cleaned = r.canonical_social_results()
    assert len(cleaned) == 1
    assert cleaned[0]["url"] == "https://www.instagram.com/p/A/"
    r.content_hash()   # must not raise


def test_platforms_found_is_deduplicated_and_sorted():
    r = _record(all_social_results=[
        {"url": "https://www.youtube.com/watch?v=b", "platform": "YouTube"},
        {"url": "https://www.instagram.com/p/A/", "platform": "Instagram"},
        {"url": "https://www.youtube.com/watch?v=a", "platform": "YouTube"},
        {"url": "https://x.com/a/status/1", "platform": None},
    ])
    assert r.platforms_found == ["Instagram", "YouTube"]


def test_empty_social_results_round_trip(tmp_path):
    r = _record()
    assert r.canonical_social_results() == []
    assert r.platforms_found == []
    p = tmp_path / "rec.json"
    r.save(p)
    assert MatchRecord.load(p).content_hash() == r.content_hash()


def test_social_results_survive_a_save_load_round_trip(tmp_path):
    r = _record(all_social_results=[
        {"url": "https://www.youtube.com/watch?v=uNJO_n_-510",
         "platform": "YouTube", "via_cdn": True},
        {"url": "https://www.instagram.com/p/ABC123/",
         "platform": "Instagram", "via_cdn": False},
    ])
    p = tmp_path / "rec.json"
    r.save(p)
    loaded = MatchRecord.load(p)
    assert loaded.platforms_found == ["Instagram", "YouTube"]
    assert loaded.content_hash() == r.content_hash()


def test_a_v1_record_still_hashes_as_v1(tmp_path):
    """Records anchored before this change must keep verifying."""
    r = _record()
    r.record_version = "stage2-match-record/v1"
    p = tmp_path / "old.json"
    r.save(p)
    loaded = MatchRecord.load(p)
    assert loaded.record_version == "stage2-match-record/v1"
    assert loaded.content_hash() == r.content_hash()


# ---------------------------------------------------------------------------
# Attributable web sources (Wikimedia et al)
#
# Reporting "the image isn't indexed on any social platform the engine can
# see" when Wikimedia had it was false. These are not social media and must
# never count as such — but they ARE real evidence the image is public.
# ---------------------------------------------------------------------------

from search.social_filter import (  # noqa: E402
    canonical_url_from_web_source,
    match_web_source,
    unwrap_proxy_url,
)

WIKI_UPLOAD = ("https://upload.wikimedia.org/wikipedia/commons/a/a4/"
               "Salman_Khan_snapped_at_the_Angry_Young_Men_trailer_launch.jpg")


@pytest.mark.parametrize("url,expected", [
    (WIKI_UPLOAD, "Wikimedia Commons"),
    ("https://commons.wikimedia.org/wiki/File:X.jpg", "Wikimedia Commons"),
    ("https://en.wikipedia.org/wiki/Salman_Khan", "Wikipedia"),
    ("https://web.archive.org/web/2020/http://x.com", "Internet Archive"),
])
def test_web_sources_are_recognised(url, expected):
    assert match_web_source(url) == expected


def test_web_sources_are_never_counted_as_social():
    """The Stage 2 requirement is a SOCIAL post; a Wikipedia hit must not
    quietly satisfy it."""
    r = rank_results([{"url": WIKI_UPLOAD}])[0]
    assert r.is_social is False
    assert r.platform is None
    assert r.is_web_source is True
    assert r.web_source == "Wikimedia Commons"
    assert best_social_result([{"url": WIKI_UPLOAD}]) is None


def test_wikimedia_upload_resolves_to_the_readable_file_page():
    assert canonical_url_from_web_source(WIKI_UPLOAD) == (
        "https://commons.wikimedia.org/wiki/File:"
        "Salman_Khan_snapped_at_the_Angry_Young_Men_trailer_launch.jpg")


def test_wikimedia_thumbnail_resolves_to_the_same_page():
    thumb = ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
             "Some_File.jpg/800px-Some_File.jpg")
    assert canonical_url_from_web_source(thumb) == \
        "https://commons.wikimedia.org/wiki/File:Some_File.jpg"


def test_language_wiki_upload_resolves_to_that_wiki():
    url = "https://upload.wikimedia.org/wikipedia/en/3/3f/Logo.png"
    assert canonical_url_from_web_source(url) == \
        "https://en.wikipedia.org/wiki/File:Logo.png"


@pytest.mark.parametrize("url", [
    "https://commons.wikimedia.org/wiki/File:X.jpg",   # already a page
    "https://example.com/wikipedia/commons/a/a4/X.jpg",
    "https://upload.wikimedia.org/nonsense",
])
def test_no_wiki_page_invented_from_unrecognised_paths(url):
    assert canonical_url_from_web_source(url) is None


def test_wordpress_image_proxy_is_unwrapped():
    """i0.wp.com is a proxy; the real host is inside the path and is what
    belongs in the record."""
    proxied = ("https://i0.wp.com/images.indianexpress.com/2025/06/"
               "Salman-Khan-2-1.jpg?resize=758%2C758&ssl=1")
    assert unwrap_proxy_url(proxied) == \
        "https://images.indianexpress.com/2025/06/Salman-Khan-2-1.jpg"

    r = rank_results([{"url": proxied}])[0]
    assert r.domain == "images.indianexpress.com"


@pytest.mark.parametrize("url", [
    "https://example.com/a/b.jpg",
    "https://i0.wp.com/",
    "https://i0.wp.com/onlyhost",
])
def test_non_proxy_urls_are_left_alone(url):
    assert unwrap_proxy_url(url) is None


def test_social_always_outranks_a_web_source():
    ranked = rank_results([
        {"url": WIKI_UPLOAD},
        {"url": "https://www.instagram.com/p/REAL/"},
    ])
    assert ranked[0].is_social and ranked[0].rank_score == 5
    assert ranked[1].is_web_source and ranked[1].rank_score == 1


def test_web_source_outranks_an_ordinary_news_page():
    ranked = rank_results([
        {"url": "https://images.news18.com/x.jpg"},
        {"url": WIKI_UPLOAD},
    ])
    assert ranked[0].is_web_source
    assert ranked[1].rank_score == 0


def test_the_wikipedia_run_records_the_finding_instead_of_denying_it():
    """Regression for the run that claimed the image wasn't indexed
    anywhere while Wikimedia was holding it."""
    raw = [
        {"url": WIKI_UPLOAD, "title": "File:Salman Khan snapped..."},
        {"url": "https://i0.wp.com/images.indianexpress.com/2025/06/x.jpg?ssl=1",
         "title": "news"},
        {"url": "https://s3.crackedcdn.com/phpimages/imageset/6/3/4/1.jpg",
         "title": "cracked"},
    ]
    ranked = rank_results(raw)
    assert best_social_result(raw) is None, "none of these are social"
    web = [r for r in ranked if r.is_web_source]
    assert len(web) == 1
    assert web[0].web_source == "Wikimedia Commons"
    assert web[0].url.startswith("https://commons.wikimedia.org/wiki/File:")


# --- record-level -----------------------------------------------------------

def test_social_post_found_flag_changes_the_hash():
    """'We found a post' and 'we found none' must never hash alike."""
    found = _record(social_post_found=True)
    not_found = _record(social_post_found=False)
    assert found.content_hash() != not_found.content_hash()


def test_web_sources_are_hashed():
    a = _record(social_post_found=False)
    b = _record(social_post_found=False, web_sources=[
        {"url": "https://commons.wikimedia.org/wiki/File:X.jpg",
         "source": "Wikimedia Commons"}])
    assert a.content_hash() != b.content_hash()


def test_web_source_order_does_not_change_the_hash():
    items = [
        {"url": "https://commons.wikimedia.org/wiki/File:B.jpg",
         "source": "Wikimedia Commons"},
        {"url": "https://en.wikipedia.org/wiki/X", "source": "Wikipedia"},
    ]
    assert _record(web_sources=items).content_hash() == \
        _record(web_sources=list(reversed(items))).content_hash()


def test_malformed_web_sources_are_dropped_not_crashed():
    r = _record(web_sources=[
        {"url": "https://commons.wikimedia.org/wiki/File:X.jpg",
         "source": "Wikimedia Commons"},
        {"url": ""}, {}, "nope", None,
    ])
    assert len(r.canonical_web_sources()) == 1
    r.content_hash()


def test_a_negative_finding_round_trips_and_verifies(tmp_path):
    """A 'no social post' record must be anchorable and re-verifiable like
    any other — that is the point of recording it."""
    r = _record(
        post_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        social_post_found=False,
        web_sources=[{"url": "https://commons.wikimedia.org/wiki/File:X.jpg",
                      "source": "Wikimedia Commons"}],
    )
    p = tmp_path / "rec.json"
    r.save(p)
    loaded = MatchRecord.load(p)
    assert loaded.social_post_found is False
    assert loaded.content_hash() == r.content_hash()


def test_older_record_versions_still_hash_as_themselves(tmp_path):
    for version in ("stage2-match-record/v1", "stage2-match-record/v2"):
        r = _record()
        r.record_version = version
        p = tmp_path / f"{version.replace('/', '_')}.json"
        r.save(p)
        loaded = MatchRecord.load(p)
        assert loaded.record_version == version
        assert loaded.content_hash() == r.content_hash()
