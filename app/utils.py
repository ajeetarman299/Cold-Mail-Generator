"""Fetch a careers page and turn it into clean, size-capped text for the LLM."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; cold-mail-generator/2.0)"

# Enough for a careers page or a long job posting (about 3,000 tokens of English
# text, at roughly four characters per token), while keeping each extraction call small.
MAX_PAGE_CHARS = 12_000

# A page with less readable text than this cannot hold a job posting. It is
# usually an empty shell that loads its jobs with JavaScript.
MIN_PAGE_CHARS = 200

# Tags whose text is never part of a job posting.
NOISE_TAGS = ["script", "style", "noscript", "template", "svg", "iframe", "nav", "footer"]

# Tags that start a new line of text, so blocks do not run together.
BLOCK_TAGS = [
    "p", "div", "section", "article", "header", "main", "aside", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd",
    "table", "tr", "blockquote", "pre",
]

BOT_CHECK_MARKERS = (
    "access denied",
    "attention required",
    "are you a robot",
    "verify you are human",
    "unusual traffic",
    "just a moment",
    "captcha",
)


class PageError(Exception):
    """The page could not be used. The message is written for the person using the app."""


def normalize_url(url: str) -> str:
    """Check that the input looks like a web address. Adds https:// when the scheme is missing."""
    url = (url or "").strip()
    if not url:
        raise PageError("Enter the URL of a careers page or a job posting.")
    typed = url
    if "://" not in url:
        url = "https://" + url
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except ValueError:  # for example, an unclosed "[" in the address
        parsed, host = None, ""
    if (
        parsed is None
        or parsed.scheme not in {"http", "https"}
        or " " in url
        or not host
        or ("." not in host and host != "localhost")
    ):
        raise PageError(
            f"'{typed}' is not a valid web address. "
            "Use a full URL such as https://careers.example.com/jobs."
        )
    return url


def fetch_html(url: str, timeout: float = 20.0) -> bytes:
    """Download a page and return its raw HTML. Raises PageError with a readable message."""
    url = normalize_url(url)
    host = urlparse(url).netloc
    headers = {"User-Agent": os.getenv("USER_AGENT", DEFAULT_USER_AGENT)}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise PageError(f"{host} did not respond within {timeout:.0f} seconds. Try again later.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise PageError(
            f"Could not connect to {host}. Check the address and your internet connection."
        ) from exc
    except (requests.exceptions.RequestException, ValueError) as exc:
        # ValueError: urllib3 rejects some malformed host names, such as a..b.com, this way.
        raise PageError(f"Could not download the page: {exc}") from exc

    status = response.status_code
    if status in (401, 403, 429):
        raise PageError(
            f"{host} blocked the request (HTTP {status}). Some careers sites block automated "
            "readers. Try the URL of a single job posting, or another site."
        )
    if status == 404:
        raise PageError("Page not found (HTTP 404). Check the URL.")
    if status >= 400:
        raise PageError(f"{host} returned an error (HTTP {status}). Check the URL or try again later.")

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type and "html" not in content_type and not content_type.startswith("text/"):
        raise PageError(
            f"The URL points to a file of type {content_type}, not a web page. "
            "Use the URL of a careers page."
        )
    return response.content


def html_to_text(html: str | bytes) -> str:
    """Return the visible text of an HTML page, with one line per block of text."""
    # Bytes in, so BeautifulSoup honours the page's own charset.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    # Line breaks inside the HTML source are layout, not content (browsers ignore them too).
    for node in soup.find_all(string=True):
        if type(node) is NavigableString:
            node.replace_with(" ".join(node.split()))
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    """Strip leftover tags, URLs and stray symbols, and collapse whitespace.

    Keeps one line per block of text, letters in any language, and the symbols
    that appear in tech names such as C++, C#, Node.js and CI/CD.
    """
    text = re.sub(r"<[^>]*?>", " ", text)  # leftover HTML tags
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)  # URLs
    text = re.sub(r"[^\w\s.,:;!?()'&%$+#/-]", " ", text)  # stray symbols
    lines: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line and (not lines or line != lines[-1]):  # drop blank and repeated lines
            lines.append(line)
    return "\n".join(lines)


def cap_text(text: str, max_chars: int = MAX_PAGE_CHARS) -> tuple[str, bool]:
    """Cut text to at most max_chars at a line or word break. Returns (text, was_cut)."""
    if len(text) <= max_chars:
        return text, False
    cut = text.rfind("\n", 0, max_chars + 1)
    if cut < max_chars // 2:
        cut = text.rfind(" ", 0, max_chars + 1)
    if cut <= 0:
        cut = max_chars
    return text[:cut].rstrip(), True


def check_page_text(text: str) -> None:
    """Raise PageError when the page text cannot hold job postings."""
    lowered = text.lower()
    if len(text) < 1_000 and any(marker in lowered for marker in BOT_CHECK_MARKERS):
        raise PageError(
            "The site showed a bot check instead of the page. Some careers sites block automated "
            "readers. Try the URL of a single job posting, or another site."
        )
    if len(text) < MIN_PAGE_CHARS:
        raise PageError(
            "The page has almost no readable text. It may load its jobs with JavaScript, "
            "which this app does not run. Try the URL of a single job posting."
        )


def page_text_from_html(html: str | bytes, max_chars: int = MAX_PAGE_CHARS) -> tuple[str, bool]:
    """Turn raw HTML into the text sent to the LLM. Returns (text, was_cut)."""
    text = clean_text(html_to_text(html))
    check_page_text(text)
    return cap_text(text, max_chars)
