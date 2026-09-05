"""
social_filter.py — decide which search results are actual social media posts.

Stage 2's requirement is to find "at least one real, matching social media
post". A reverse image search returns every page hosting a matching image —
news articles, image aggregators, scraper sites, CDNs. This module separates
genuine social/profile pages from the noise and ranks them.

Pure logic, no network. Fully unit-tested.
"""

from __future__ import annotations

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

# Sites that frequently appear in reverse-image results but are aggregators,
# mirrors or scrapers rather than the original post. Ranked below real
# social results even when they technically match.
LOW_VALUE_DOMAINS = {
    "pinterest.com",       # usually a re-pin of someone else's image
    "imgur.com",
    "flickr.com",
    "wallpapercave.com",
    "pinimg.com",
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

    @property
    def summary(self) -> str:
        label = self.platform or self.domain or "unknown"
        return f"[{label}] {self.title or self.url}"


def _score(url: str, platform: Optional[str], low_value: bool) -> int:
    """
    Higher is better.

      3 = social platform, not an aggregator  (what we actually want)
      2 = social platform but an aggregator/mirror (Pinterest, Flickr)
      1 = not social, but a real page
      0 = unusable
    """
    if not extract_domain(url):
        return 0
    if platform and not low_value:
        return 3
    if platform and low_value:
        return 2
    return 1


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
        # Deduplicate on the URL without a trailing slash or fragment.
        key = url.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)

        platform = match_social_domain(url)
        low_value = is_low_value_domain(url)
        ranked.append(
            RankedResult(
                url=url,
                title=(raw.get("title") or "").strip(),
                thumbnail_url=(raw.get("thumbnail_url") or "").strip(),
                engine=(raw.get("engine") or "").strip(),
                domain=domain,
                platform=platform,
                is_social=platform is not None,
                is_low_value=low_value,
                rank_score=_score(url, platform, low_value),
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
