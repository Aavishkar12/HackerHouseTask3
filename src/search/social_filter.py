"""
social_filter.py — decide which search results are actual social media posts.

Stage 2's requirement is to find "at least one real, matching social media
post". A reverse image search returns every page hosting a matching image —
news articles, image aggregators, scraper sites, CDNs. This module separates
genuine social/profile pages from the noise and ranks them.

Pure logic, no network. Fully unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

# Registrable domains treated as social media / personal-profile platforms.
# Value is the human-readable platform name that ends up in the final record.
SOCIAL_DOMAINS: Dict[str, str] = {
    "instagram.com": "Instagram",
    "facebook.com": "Facebook",
    "fb.com": "Facebook",
    "fb.watch": "Facebook",
    "twitter.com": "Twitter/X",
    "x.com": "Twitter/X",
    "linkedin.com": "LinkedIn",
    "tiktok.com": "TikTok",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "reddit.com": "Reddit",
    "pinterest.com": "Pinterest",
    "tumblr.com": "Tumblr",
    "threads.net": "Threads",
    "threads.com": "Threads",
    "snapchat.com": "Snapchat",
    "bsky.app": "Bluesky",
    "t.me": "Telegram",
    "vk.com": "VK",
    "ok.ru": "Odnoklassniki",
    "weibo.com": "Weibo",
    "flickr.com": "Flickr",
    "medium.com": "Medium",
    "github.com": "GitHub",
    "quora.com": "Quora",
    "behance.net": "Behance",
    "dribbble.com": "Dribbble",
    "devpost.com": "Devpost",
}

# Image/media CDNs owned by the platforms above.
#
# Reverse image search frequently returns the CDN URL of the matched image
# rather than the page URL it appeared on — e.g. Yandex returns
# 'https://i.ytimg.com/vi/uNJO_n_-510/oardefault.jpg' for a YouTube video.
# Those hosts are not in SOCIAL_DOMAINS, so without this map a genuine
# social hit is scored as ordinary web noise and discarded. That is a real
# bug: the post WAS found, and we threw it away.
MEDIA_CDN_DOMAINS: Dict[str, str] = {
    "ytimg.com": "YouTube",
    "pinimg.com": "Pinterest",
    "cdninstagram.com": "Instagram",
    "twimg.com": "Twitter/X",
    "licdn.com": "LinkedIn",
    "redd.it": "Reddit",
    "redditmedia.com": "Reddit",
    "redditstatic.com": "Reddit",
    "tiktokcdn.com": "TikTok",
    "tiktokcdn-us.com": "TikTok",
    "staticflickr.com": "Flickr",
    "media.tumblr.com": "Tumblr",
    "ggpht.com": "YouTube",
    "sndcdn.com": "SoundCloud",
    # fbcdn.net serves BOTH Facebook and Instagram. Instagram's variants
    # carry 'instagram' in the host, so that case is split out in
    # match_media_cdn() rather than guessed here.
    "fbcdn.net": "Facebook",
}

# Attributable public web sources: NOT social media, but not noise either.
#
# A Wikimedia Commons file page carries an uploader, a date, a licence and
# often the original source — it is real, checkable evidence that the image
# exists publicly, even though it is not "a social media post". Reporting
# these as "found nothing" was actively misleading: the engine could see
# the image perfectly well.
#
# These are tracked separately from SOCIAL_DOMAINS and never counted as
# social, so the Stage 2 requirement ("find a real social media post")
# cannot be quietly satisfied by a Wikipedia hit.
WEB_SOURCE_DOMAINS: Dict[str, str] = {
    "wikimedia.org": "Wikimedia Commons",
    "wikipedia.org": "Wikipedia",
    "wikidata.org": "Wikidata",
    "wikiquote.org": "Wikiquote",
    "archive.org": "Internet Archive",
    "commons.wikimedia.org": "Wikimedia Commons",
}

# Sites that frequently appear in reverse-image results but are aggregators,
# mirrors or scrapers rather than the original post. Ranked below real
# social results even when they technically match.
LOW_VALUE_DOMAINS = {
    "pinterest.com",       # usually a re-pin of someone else's image
    "imgur.com",
    "flickr.com",
    "wallpapercave.com",
    "pinimg.com",
    "staticflickr.com",
}

# Subdomain prefixes to strip before matching (m.facebook.com -> facebook.com).
_STRIP_PREFIXES = ("www.", "m.", "mobile.", "in.", "en.", "web.")


def extract_domain(url: str) -> str:
    """
    Return the normalised host for a URL: lowercased, port removed, and
    common prefixes like 'www.' / 'm.' stripped.

    Returns "" for anything that isn't a parseable http(s) URL.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return ""
    if parsed.scheme not in ("http", "https"):
        return ""

    host = (parsed.netloc or "").lower()
    if "@" in host:            # strip any userinfo
        host = host.rsplit("@", 1)[-1]
    if ":" in host:            # strip port
        host = host.split(":", 1)[0]

    for prefix in _STRIP_PREFIXES:
        if host.startswith(prefix):
            host = host[len(prefix):]
            break
    return host


def match_social_domain(url: str) -> Optional[str]:
    """
    If the URL belongs to a known social platform, return that platform's
    name; otherwise None.

    Matches the registrable domain and any subdomain of it, so
    'https://uk.linkedin.com/in/someone' -> 'LinkedIn'.
    """
    host = extract_domain(url)
    if not host:
        return None
    for domain, platform in SOCIAL_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return platform
    return None


def match_media_cdn(url: str) -> Optional[str]:
    """
    If the URL is an image/media CDN belonging to a social platform, return
    that platform's name; otherwise None.

    This is what rescues results like
    'https://i.ytimg.com/vi/uNJO_n_-510/oardefault.jpg' — a real YouTube
    post that match_social_domain() alone would reject.
    """
    host = extract_domain(url)
    if not host:
        return None

    # fbcdn.net carries both Facebook and Instagram media; Instagram's
    # variants name themselves (e.g. instagram.fbom1-1.fna.fbcdn.net).
    if host == "fbcdn.net" or host.endswith(".fbcdn.net"):
        return "Instagram" if "instagram" in host else "Facebook"

    for domain, platform in MEDIA_CDN_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return platform
    return None


# A YouTube thumbnail URL embeds the video id: /vi/<id>/name.jpg (or
# /vi_webp/<id>/...). Ids are exactly 11 URL-safe base64 characters.
_YT_THUMB_RE = re.compile(r"/vi(?:_webp)?/([A-Za-z0-9_-]{11})/")


def canonical_url_from_cdn(url: str) -> Optional[str]:
    """
    Recover the real post URL from a media-CDN URL, where the CDN URL
    reliably encodes it.

    Only YouTube qualifies today: its thumbnails contain the video id, so
    the watch page can be reconstructed exactly. Instagram, Pinterest and
    Twitter CDN paths contain opaque storage keys with no post id in them,
    so guessing a post URL from those would be fabrication — we return
    None and keep the CDN URL, flagged as such.
    """
    host = extract_domain(url)
    if not host:
        return None
    if not (host == "ytimg.com" or host.endswith(".ytimg.com")):
        return None
    m = _YT_THUMB_RE.search(url)
    if not m:
        return None
    return f"https://www.youtube.com/watch?v={m.group(1)}"


def match_web_source(url: str) -> Optional[str]:
    """
    If the URL is an attributable public web source (Wikimedia, Wikipedia,
    Internet Archive), return its name; otherwise None.

    Deliberately separate from match_social_domain(): these are evidence
    the image exists publicly, but they are NOT social media posts and must
    never be counted as satisfying that requirement.
    """
    host = extract_domain(url)
    if not host:
        return None
    for domain, name in WEB_SOURCE_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None


# upload.wikimedia.org paths carry the file name, so the human-readable
# Commons/Wikipedia file page can be reconstructed exactly:
#   /wikipedia/commons/a/a4/Some_File.jpg
#   /wikipedia/commons/thumb/a/a4/Some_File.jpg/800px-Some_File.jpg
#   /wikipedia/en/3/3f/Some_File.jpg
_WIKI_UPLOAD_RE = re.compile(
    r"/wikipedia/(?P<wiki>[a-z\-]+)/(?:thumb/)?"
    r"[0-9a-f]/[0-9a-f]{2}/(?P<name>[^/?#]+)"
)

# WordPress's Photon image proxy wraps the real URL:
#   https://i0.wp.com/images.indianexpress.com/2025/06/x.jpg?resize=758&ssl=1
_WP_PROXY_RE = re.compile(r"^i\d+\.wp\.com$")


def unwrap_proxy_url(url: str) -> Optional[str]:
    """
    Undo a known image-proxy wrapper, returning the real underlying URL.

    Only WordPress's Photon proxy today. Reporting 'i0.wp.com' as the
    source is wrong — the image actually lives on the site named inside
    the path, and that is what belongs in the record.
    """
    host = extract_domain(url)
    if not host or not _WP_PROXY_RE.match(host):
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    inner = (parsed.path or "").lstrip("/")
    if not inner or "/" not in inner:
        return None
    scheme = "http" if "ssl=0" in (parsed.query or "") else "https"
    return f"{scheme}://{inner}"


def canonical_url_from_web_source(url: str) -> Optional[str]:
    """
    Recover the human-readable file page from a raw media URL.

    upload.wikimedia.org serves the bytes; the page a person can actually
    read — with uploader, licence and source — is on commons.wikimedia.org
    (or the language wiki). The file name is right there in the path, so
    this is reconstruction, not guesswork.
    """
    host = extract_domain(url)
    if not host or not host.startswith("upload."):
        return None
    m = _WIKI_UPLOAD_RE.search(url)
    if not m:
        return None
    wiki, name = m.group("wiki"), m.group("name")
    if wiki == "commons":
        return f"https://commons.wikimedia.org/wiki/File:{name}"
    return f"https://{wiki}.wikipedia.org/wiki/File:{name}"


def is_social_media_url(url: str) -> bool:
    """True if the URL points at a known social/profile platform."""
    return match_social_domain(url) is not None


def is_low_value_domain(url: str) -> bool:
    """True for aggregator/mirror sites that rarely are the original post."""
    host = extract_domain(url)
    if not host:
        return False
    return any(
        host == d or host.endswith("." + d) for d in LOW_VALUE_DOMAINS
    )


@dataclass
class RankedResult:
    """A search result annotated with why it was (or wasn't) selected."""

    url: str
    title: str = ""
    thumbnail_url: str = ""
    engine: str = ""
    domain: str = ""
    platform: Optional[str] = None      # e.g. "Instagram", None if not social
    is_social: bool = False
    is_low_value: bool = False
    rank_score: int = 0
    # True when the platform was identified from a media-CDN host rather
    # than a page URL. Weaker evidence, and worth reporting honestly.
    via_cdn: bool = False
    # The CDN URL we started from, when `url` was rewritten to a recovered
    # canonical post URL. Empty when no rewrite happened.
    cdn_url: str = ""
    # Attributable non-social source (Wikimedia, Wikipedia, Archive.org).
    # Never counted as social — `is_social` stays False for these.
    web_source: Optional[str] = None

    @property
    def is_web_source(self) -> bool:
        return self.web_source is not None

    @property
    def summary(self) -> str:
        label = self.platform or self.web_source or self.domain or "unknown"
        suffix = " (via CDN)" if self.via_cdn else ""
        return f"[{label}{suffix}] {self.title or self.url}"


def _score(url: str, platform: Optional[str], low_value: bool,
           via_cdn: bool = False, recovered: bool = False,
           web_source: Optional[str] = None) -> int:
    """
    Higher is better.

      5 = social page URL, not an aggregator   (what we actually want)
      4 = social platform via CDN, canonical post URL recovered exactly
      3 = social platform, but an aggregator/mirror (Pinterest, Flickr)
      2 = social platform via CDN only, no post URL recoverable
      1 = attributable non-social web source (Wikimedia, Archive.org)
      0 = ordinary web page, or unusable

    A CDN hit with a recovered page URL outranks an aggregator, because it
    points at a specific post. A CDN hit without one still counts as social
    (the image genuinely is hosted by that platform) but ranks below them,
    since we cannot say WHICH post it came from.

    Web sources rank beneath every social hit but above plain pages: they
    are real, checkable evidence that the image is public, which matters
    when no social post exists — but they never satisfy the social
    requirement, and `is_social` stays False for them.
    """
    if not extract_domain(url):
        return 0
    if platform:
        if via_cdn:
            return 4 if recovered else 2
        return 3 if low_value else 5
    if web_source:
        return 1
    return 0


def rank_results(results: Iterable[dict]) -> List[RankedResult]:
    """
    Annotate, deduplicate and rank raw search results.

    Accepts dicts with at least a 'url' key (extra keys 'title',
    'thumbnail_url', 'engine' are used when present) — this matches what
    reverse_search.py produces.

    Returns results sorted best-first. Duplicate URLs are dropped, keeping
    the first (and richest) occurrence.
    """
    seen: set = set()
    ranked: List[RankedResult] = []

    for raw in results:
        url = (raw.get("url") or "").strip()
        domain = extract_domain(url)
        if not domain:
            continue                     # unparseable / non-http

        # An image proxy hides the real host; unwrap before doing anything
        # else, or we would report 'i0.wp.com' as the source.
        unwrapped = unwrap_proxy_url(url)
        if unwrapped:
            url = unwrapped
            domain = extract_domain(url) or domain

        # Deduplicate on the URL without a trailing slash or fragment.
        key = url.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)

        low_value = is_low_value_domain(url)
        platform = match_social_domain(url)
        via_cdn = False
        cdn_url = ""
        final_url = url

        if platform is None:
            # Not a recognisable social PAGE — but it may still be that
            # platform's image CDN, which is how most reverse-image hits
            # for social posts actually come back.
            cdn_platform = match_media_cdn(url)
            if cdn_platform is not None:
                platform = cdn_platform
                via_cdn = True
                recovered = canonical_url_from_cdn(url)
                if recovered:
                    # We can name the exact post — promote the real URL and
                    # keep the CDN one for provenance.
                    cdn_url = url
                    final_url = recovered
                    domain = extract_domain(recovered)
                    low_value = is_low_value_domain(recovered)

        # Not social at all? It may still be an attributable public source
        # (Wikimedia and friends) — real evidence, just not a social post.
        web_source = None
        if platform is None:
            web_source = match_web_source(url)
            if web_source:
                page = canonical_url_from_web_source(url)
                if page:
                    # upload.wikimedia.org serves bytes; the readable file
                    # page with uploader/licence is the useful URL.
                    cdn_url = url
                    final_url = page
                    domain = extract_domain(page)

        ranked.append(
            RankedResult(
                url=final_url,
                title=(raw.get("title") or "").strip(),
                thumbnail_url=(raw.get("thumbnail_url") or "").strip(),
                engine=(raw.get("engine") or "").strip(),
                domain=domain,
                platform=platform,
                is_social=platform is not None,
                is_low_value=low_value,
                rank_score=_score(url, platform, low_value, via_cdn,
                                  recovered=bool(cdn_url) and via_cdn,
                                  web_source=web_source),
                via_cdn=via_cdn,
                cdn_url=cdn_url,
                web_source=web_source,
            )
        )

    # Stable sort by score descending — preserves engine ordering within a tier.
    ranked.sort(key=lambda r: r.rank_score, reverse=True)
    return ranked


def best_social_result(results: Iterable[dict]) -> Optional[RankedResult]:
    """
    Return the single best social media result, or None if the search
    surfaced no social pages at all.

    'Best' = highest ranked result that is on a social platform. An
    aggregator like Pinterest is returned only if nothing better exists.
    """
    for result in rank_results(results):
        if result.is_social:
            return result
    return None
