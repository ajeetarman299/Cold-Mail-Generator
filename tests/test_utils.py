"""Page fetching and text cleaning."""

import pytest
import requests
from urllib3.exceptions import LocationParseError

import utils
from utils import (
    PageError,
    cap_text,
    clean_text,
    fetch_html,
    html_to_text,
    normalize_url,
    page_text_from_html,
)

JOB_TEXT = "We are hiring a Data Engineer to build pipelines in Python and SQL. " * 5


def test_html_to_text_keeps_content_and_drops_scripts_styles_and_menus():
    html = """
    <html><head><style>body { color: red }</style><script>track("view")</script></head>
    <body>
      <nav><a href="/">Home</a><a href="/shop">Shop</a></nav>
      <h1>Senior Data Engineer</h1>
      <p>Build pipelines in <b>Python</b>
         and SQL.</p>
      <footer>Privacy policy</footer>
    </body></html>
    """
    lines = clean_text(html_to_text(html)).splitlines()

    assert lines == ["Senior Data Engineer", "Build pipelines in Python and SQL."]


def test_clean_text_removes_urls_tags_and_stray_symbols():
    raw = "Apply at https://jobs.example.com/123 now! <b>Remote</b> ★★ Salary: 100% fair & square"

    assert clean_text(raw) == "Apply at now! Remote Salary: 100% fair & square"


def test_clean_text_keeps_tech_names_and_letters_in_other_languages():
    raw = "C++, C#, Node.js, CI/CD, Kubernetes; Entwickler (m/w/d) in München"

    assert clean_text(raw) == raw


def test_clean_text_keeps_one_line_per_block_and_drops_repeats():
    raw = "Search jobs\n\n   Search jobs  \nData   Engineer\n\n\nPython"

    assert clean_text(raw) == "Search jobs\nData Engineer\nPython"


def test_cap_text_leaves_short_text_alone():
    assert cap_text("short text", max_chars=100) == ("short text", False)


def test_cap_text_cuts_long_text_at_a_line_break():
    text = "first line\nsecond line\nthird line"

    capped, was_cut = cap_text(text, max_chars=25)

    assert was_cut
    assert capped == "first line\nsecond line"


def test_page_text_from_html_rejects_a_javascript_shell():
    html = "<html><body><div id='root'></div><noscript>Please enable JavaScript</noscript></body></html>"

    with pytest.raises(PageError, match="almost no readable text"):
        page_text_from_html(html)


def test_page_text_from_html_rejects_a_bot_check():
    html = "<html><body><h1>Just a moment...</h1><p>Verify you are human to continue.</p></body></html>"

    with pytest.raises(PageError, match="bot check"):
        page_text_from_html(html)


def test_page_text_from_html_reports_when_it_cuts_the_text():
    html = f"<html><body><p>{JOB_TEXT * 10}</p></body></html>"

    text, was_cut = page_text_from_html(html, max_chars=1_000)

    assert was_cut
    assert len(text) <= 1_000


@pytest.mark.parametrize(
    "url", ["", "   ", "not a url", "ftp://example.com/jobs", "https://nodot", "https://[careers.example.com"]
)
def test_normalize_url_rejects_bad_input(url):
    with pytest.raises(PageError):
        normalize_url(url)


def test_normalize_url_names_the_address_as_typed():
    with pytest.raises(PageError, match="'not a url' is not a valid web address"):
        normalize_url("not a url")


def test_normalize_url_adds_https_when_missing():
    assert normalize_url("careers.example.com/jobs") == "https://careers.example.com/jobs"


class FakeResponse:
    def __init__(self, status_code=200, content_type="text/html; charset=utf-8", content=b"<p>ok</p>"):
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.content = content


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(403), "blocked the request"),
        (FakeResponse(429), "blocked the request"),
        (FakeResponse(404), "Page not found"),
        (FakeResponse(500), "returned an error"),
        (FakeResponse(content_type="application/pdf"), "not a web page"),
    ],
)
def test_fetch_html_explains_http_problems(monkeypatch, response, message):
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(PageError, match=message):
        fetch_html("https://careers.example.com/jobs")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (requests.exceptions.ConnectionError("dns failure"), "Could not connect"),
        (requests.exceptions.Timeout("slow"), "did not respond"),
        (LocationParseError("a..b.com, label empty or too long"), "Could not download the page"),
    ],
)
def test_fetch_html_explains_network_problems(monkeypatch, error, message):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(utils.requests, "get", fail)

    with pytest.raises(PageError, match=message):
        fetch_html("https://careers.example.com/jobs")


def test_fetch_html_returns_the_page(monkeypatch):
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: FakeResponse(content=b"<p>Jobs</p>"))

    assert fetch_html("https://careers.example.com/jobs") == b"<p>Jobs</p>"
