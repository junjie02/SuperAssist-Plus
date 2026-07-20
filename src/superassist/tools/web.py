from __future__ import annotations

import html
import json
import re
from urllib.error import URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from superassist.config import get_settings

_TAG_RE = re.compile(r"<[^>]+>")
_RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="(?P<href>[^"]+)".*?>(?P<title>.*?)</a>.*?'
    r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_LITE_RESULT_RE = re.compile(
    r"<a\s+[^>]*href=[\"'](?P<href>[^\"']+)[\"'][^>]*class=[\"']result-link[\"'][^>]*>"
    r"(?P<title>.*?)</a>.*?"
    r"<td\s+class=[\"']result-snippet[\"'][^>]*>(?P<snippet>.*?)</td>",
    re.DOTALL,
)


def _ensure_network_enabled() -> str | None:
    if not get_settings().tool_network_enabled:
        return "Error: Network tools are disabled by SUPERASSIST_TOOL_NETWORK_ENABLED=false"
    return None


def _fetch_url(url: str, timeout: int = 15) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; SuperAssist/0.1; +https://localhost)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
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


def _normalize_result_url(href: str) -> str:
    href = html.unescape(href).strip()
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


def _parse_search_results(body: str, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for pattern in (_LITE_RESULT_RE, _RESULT_RE):
        for match in pattern.finditer(body):
            results.append(
                {
                    "title": _clean_html(match.group("title")),
                    "url": _normalize_result_url(match.group("href")),
                    "snippet": _clean_html(match.group("snippet")),
                }
            )
            if len(results) >= max_results:
                return results
        if results:
            return results
    return results


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
    urls = [
        f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}",
        f"https://duckduckgo.com/html/?q={quote_plus(query)}",
    ]
    errors: list[str] = []
    for url in urls:
        try:
            body, _content_type = _fetch_url(url)
        except URLError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue

        results = _parse_search_results(body, max_results)
        if results:
            return json.dumps(results, ensure_ascii=False, indent=2)
    if errors:
        return f"Error: Web search failed: {'; '.join(errors)}"
    if not results:
        return "No search results found."


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
