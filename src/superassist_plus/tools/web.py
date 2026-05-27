from __future__ import annotations

import html
import json
import re
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from superassist_plus.config import get_settings

_TAG_RE = re.compile(r"<[^>]+>")
_RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="(?P<href>[^"]+)".*?>(?P<title>.*?)</a>.*?'
    r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
    re.DOTALL,
)


def _ensure_network_enabled() -> str | None:
    if not get_settings().tool_network_enabled:
        return "Error: Network tools are disabled by SUPERASSIST_PLUS_TOOL_NETWORK_ENABLED=false"
    return None


def _fetch_url(url: str, timeout: int = 15) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "SuperAssist-Plus/0.1 (+https://localhost)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        raw = response.read(1_000_000)
    charset_match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
    charset = charset_match.group(1) if charset_match else "utf-8"
    return raw.decode(charset, errors="replace"), content_type


def _clean_html(value: str) -> str:
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = _TAG_RE.sub(" ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


@tool("web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo HTML results.

    Args:
        query: Search query.
        max_results: Maximum number of results to return. Defaults to 5.
    """

    disabled = _ensure_network_enabled()
    if disabled:
        return disabled
    max_results = max(1, min(max_results, 10))
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        body, _content_type = _fetch_url(url)
    except URLError as exc:
        return f"Error: Web search failed: {exc}"
    except Exception as exc:
        return f"Error: Web search failed: {type(exc).__name__}: {exc}"

    results = []
    for match in _RESULT_RE.finditer(body):
        results.append(
            {
                "title": _clean_html(match.group("title")),
                "url": html.unescape(match.group("href")),
                "snippet": _clean_html(match.group("snippet")),
            }
        )
        if len(results) >= max_results:
            break
    if not results:
        return "No search results found."
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool("web_fetch")
def web_fetch(url: str, max_chars: int = 12000) -> str:
    """Fetch a web page and return readable text.

    Args:
        url: HTTP or HTTPS URL to fetch.
        max_chars: Maximum characters to return. Defaults to 12000.
    """

    disabled = _ensure_network_enabled()
    if disabled:
        return disabled
    if not url.lower().startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    max_chars = max(1000, min(max_chars, 50000))
    try:
        body, content_type = _fetch_url(url)
    except URLError as exc:
        return f"Error: Web fetch failed: {exc}"
    except Exception as exc:
        return f"Error: Web fetch failed: {type(exc).__name__}: {exc}"

    text = body if "text/plain" in content_type.lower() else _clean_html(body)
    return text[:max_chars]
