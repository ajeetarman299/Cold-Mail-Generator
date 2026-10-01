import os
import re

import requests
from bs4 import BeautifulSoup

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; cold-mail-generator/1.0)"


def fetch_page_text(url: str, timeout: float = 20.0) -> str:
    """Download a web page and return its visible text.

    This replaces LangChain's WebBaseLoader (requests + BeautifulSoup under the hood),
    because the langchain-community package that shipped it is being sunset.
    """
    headers = {"User-Agent": os.getenv("USER_AGENT", DEFAULT_USER_AGENT)}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")  # bytes, so the page's own charset is honoured
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<[^>]*?>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove special characters, keeping the few that appear in tech names (C++, C#, Node.js, CI/CD)
    text = re.sub(r"[^a-zA-Z0-9 .,+#/-]", " ", text)
    # Collapse repeated whitespace and trim
    return " ".join(text.split())
