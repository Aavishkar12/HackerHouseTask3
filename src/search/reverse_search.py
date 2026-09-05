"""
reverse_search.py — genuine reverse-image search via browser automation.

The task spec allows the search step to be done "via reverse image search,
an API, or a scripted search approach". Every hosted reverse-image API we
evaluated was gated behind a credit card, a minimum deposit, or phone
verification (Google Cloud Vision, SerpApi, TinEye), so this uses the
scripted approach: it drives a real browser through a real reverse-image
search and parses the real results.

Nothing here is hardcoded — the URLs returned are whatever the search
engine actually finds for the supplied image.

Engines
-------
  yandex  (default) — Yandex Images reverse search. Chosen because its
                      image index is notably better than Google's at
                      matching *faces* across the web, which is exactly
                      this pipeline's use case.
  google            — Google Images / Lens. Supported, but Google blocks
                      automation far more aggressively; best-effort.

Modes
-----
  automatic (default) — the script uploads the image itself.
  manual (--manual)   — the script opens the browser at the search page,
                        you perform the upload by hand, press Enter, and
                        the script scrapes whatever results page is open.

Manual mode exists because search engines change their DOM and deploy
bot-detection without warning. If automation breaks the day before a
deadline, manual mode still produces a real, scripted, parsed search
result rather than leaving you stuck. It works with ANY engine, because
it falls back to generic outbound-link extraction.

IMPORTANT: needs a real display. Run it locally, not headless in CI.
Setup:  pip install playwright && playwright install chromium
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

PathLike = str | Path

# A realistic desktop UA. Automation that advertises itself as headless
# Chrome gets challenged almost immediately.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

ENGINE_URLS = {
    "yandex": "https://yandex.com/images/",
    "google": "https://images.google.com/",
}

# Hosts that are the search engine itself, or infrastructure — never real
# results, so they're dropped during generic link extraction.
_ENGINE_NOISE = (
    "yandex.", "google.", "gstatic.com", "googleusercontent.com",
    "ya.ru", "yastatic.net", "chrome.com", "youtube.com/redirect",
    "policies.google", "support.google", "accounts.google",
    "myactivity.google", "translate.goog",
)


class ReverseSearchError(Exception):
    """Raised when the search could not be completed at all."""


@dataclass
class SearchResult:
    """One page found by the reverse image search."""

    url: str
    title: str = ""
    thumbnail_url: str = ""
    engine: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError as exc:
        raise ReverseSearchError(
            "Playwright is not installed. Run:\n"
            "    pip install playwright\n"
            "    playwright install chromium"
        ) from exc


def _is_noise(url: str) -> bool:
    low = url.lower()
    if not low.startswith(("http://", "https://")):
        return True
    return any(marker in low for marker in _ENGINE_NOISE)


# --------------------------------------------------------------------------
# Result extraction
# --------------------------------------------------------------------------

# Yandex renders its "sites containing this image" list under CbirSites-*.
# Class names have changed over time, so several variants are tried in order.
_YANDEX_ITEM_SELECTORS = [
    ".CbirSites-Item",
    ".CbirSites li",
    "[class*='CbirSites-Item']",
    "[class*='cbir-sites__item']",
]


def _extract_yandex(page, engine: str) -> List[SearchResult]:
    """Parse Yandex's 'sites containing this image' list."""
    results: List[SearchResult] = []
    for selector in _YANDEX_ITEM_SELECTORS:
        items = page.query_selector_all(selector)
        if not items:
            continue
        for item in items:
            link = (item.query_selector("a[href^='http']")
                    or item.query_selector("a"))
            if link is None:
                continue
            href = link.get_attribute("href") or ""
            if _is_noise(href):
                continue
            title_el = (item.query_selector("[class*='ItemTitle']")
                        or item.query_selector("[class*='Title']")
                        or link)
            thumb_el = item.query_selector("img")
            results.append(SearchResult(
                url=href,
                title=(title_el.inner_text() or "").strip() if title_el else "",
                thumbnail_url=(thumb_el.get_attribute("src") or "") if thumb_el else "",
                engine=engine,
            ))
        if results:
            break
    return results


def _extract_generic(page, engine: str) -> List[SearchResult]:
    """
    Engine-agnostic fallback: every outbound link on the page that isn't
    the engine's own infrastructure.

    Deliberately permissive — social_filter.rank_results() does the
    filtering downstream. This is what makes manual mode work on any
    search engine, including ones this module has no selectors for.
    """
    results: List[SearchResult] = []
    for link in page.query_selector_all("a[href^='http']"):
        href = link.get_attribute("href") or ""
        if _is_noise(href):
            continue
        text = ""
        try:
            text = (link.inner_text() or "").strip()
        except Exception:
            pass
        results.append(SearchResult(
            url=href, title=text[:200], engine=engine,
        ))
    return results


def extract_results(page, engine: str) -> List[SearchResult]:
    """
    Pull results off whatever page is currently loaded.

    Tries the engine-specific parser first, then falls back to generic
    link extraction so a DOM change degrades quality rather than failing
    outright.
    """
    results: List[SearchResult] = []
    if engine == "yandex":
        results = _extract_yandex(page, engine)
    if not results:
        results = _extract_generic(page, engine)
    return results


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------

def _upload_image(page, engine: str, image_path: Path, timeout_ms: int) -> None:
    """
    Put the image into the engine's reverse-image search.

    Rather than clicking through the camera-icon UI (whose selectors churn
    constantly), this sets the file directly on the page's file input,
    which is far more stable. The camera button is clicked first only when
    needed to make that input exist.
    """
    # Some engines only mount the file input after the camera button is clicked.
    camera_selectors = [
        "button[class*='CbirPanel']",
        "[class*='input__button-camera']",
        "[aria-label*='image' i]",
        "[aria-label*='Search by image' i]",
        "div[aria-label*='camera' i]",
    ]
    for sel in camera_selectors:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                page.wait_for_timeout(800)
                break
        except Exception:
            continue

    # Now find a file input — hidden ones are fine for set_input_files.
    file_input = None
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        file_input = page.query_selector("input[type='file']")
        if file_input:
            break
        page.wait_for_timeout(300)

    if file_input is None:
        raise ReverseSearchError(
            f"Could not find a file-upload input on {engine}. The page "
            "layout has probably changed, or a consent/CAPTCHA screen is "
            "blocking it.\n"
            "Re-run with --manual to do the upload by hand — the script "
            "will still parse the results."
        )

    file_input.set_input_files(str(image_path))


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def reverse_image_search(
    image_path: PathLike,
    engine: str = "yandex",
    manual: bool = False,
    headless: bool = False,
    timeout_s: int = 60,
    settle_s: float = 4.0,
) -> List[SearchResult]:
    """
    Run a reverse image search and return the pages it found.

    Args:
        image_path: the face scan to search for.
        engine: "yandex" (default) or "google".
        manual: open the browser and wait for you to do the upload
            yourself, then scrape. Use when automation is blocked.
        headless: run without a visible window. Off by default — a visible
            real browser is both less bot-like and better for the demo
            recording.
        timeout_s: how long to wait for navigation/elements.
        settle_s: pause after results load, so lazy-loaded items appear.

    Returns:
        A list of SearchResult, in the order the engine ranked them.
        May be empty — an empty result is a legitimate outcome meaning
        the image genuinely isn't indexed anywhere.

    Raises:
        ReverseSearchError, FileNotFoundError
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    engine = engine.lower().strip()
    if engine not in ENGINE_URLS:
        raise ReverseSearchError(
            f"Unknown engine {engine!r}. Options: {', '.join(ENGINE_URLS)}"
        )

    sync_playwright = _require_playwright()
    timeout_ms = timeout_s * 1000
    results: List[SearchResult] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
        )
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        try:
            print(f"[*] Opening {ENGINE_URLS[engine]}")
            page.goto(ENGINE_URLS[engine], wait_until="domcontentloaded")

            if manual:
                print("\n" + "=" * 66)
                print("  MANUAL MODE")
                print("  1. In the browser window, click the camera / "
                      "'search by image' icon")
                print(f"  2. Upload:  {image_path.resolve()}")
                print("  3. Wait for the results page to finish loading")
                print("  4. Come back here and press ENTER")
                print("=" * 66)
                input("\n  Press ENTER once the results are on screen... ")
            else:
                print(f"[*] Uploading {image_path.name} ...")
                _upload_image(page, engine, image_path, timeout_ms)
                print("[*] Waiting for results ...")
                try:
                    page.wait_for_load_state("networkidle", timeout=timeout_ms)
                except Exception:
                    pass  # networkidle is best-effort; results may already be up

            page.wait_for_timeout(int(settle_s * 1000))

            current = page.url
            print(f"[*] Results page: {current[:110]}")
            results = extract_results(page, engine)
            print(f"[*] Extracted {len(results)} raw link(s) from the page.")

            if not results:
                print("[!] No links extracted. Either the image matched "
                      "nothing, or a CAPTCHA/consent screen is in the way "
                      "(look at the browser window).")

        finally:
            if not headless:
                # Give the user a moment to see the page in the recording.
                page.wait_for_timeout(1500)
            context.close()
            browser.close()

    return results
