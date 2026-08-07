from __future__ import annotations

import html
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
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
_OFFICIAL_MEDIA_DOMAINS: ContextVar[tuple[str, ...]] = ContextVar(
    "superassist_official_media_domains",
    default=(),
)


@contextmanager
def official_media_web_scope(domains: list[str] | tuple[str, ...] | set[str]) -> Iterator[None]:
    """Restrict web tools to official-media domains for the current agent run."""

    normalized = tuple(sorted({_normalize_domain(item) for item in domains if _normalize_domain(item)}))
    token = _OFFICIAL_MEDIA_DOMAINS.set(normalized)
    try:
        yield
    finally:
        _OFFICIAL_MEDIA_DOMAINS.reset(token)


def is_allowed_official_url(url: str, domains: tuple[str, ...] | list[str] | set[str]) -> bool:
    host = (urlparse(str(url)).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _normalize_domain(value: str) -> str:
    value = str(value or "").strip().lower().rstrip(".")
    if "://" in value:
        value = (urlparse(value).hostname or "").lower().rstrip(".")
    return value.removeprefix("www.")


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
    charset = charset_match.group(1) if charset_match else ""
    if "html" in content_type.lower() or not charset:
        head = raw[:8192].decode("ascii", errors="ignore")
        meta_match = re.search(r"charset\s*=\s*[\"']?([\w.-]+)", head, re.IGNORECASE)
        if meta_match:
            charset = meta_match.group(1)
    charset = "gb18030" if charset.lower() in {"gbk", "gb2312", "gb_2312"} else (charset or "utf-8")
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
    official_domains = _OFFICIAL_MEDIA_DOMAINS.get()
    scoped_query = query
    if official_domains:
        site_filter = " OR ".join(f"site:{domain}" for domain in official_domains)
        scoped_query = f"({query}) ({site_filter})"
    urls = [
        f"https://lite.duckduckgo.com/lite/?q={quote_plus(scoped_query)}",
        f"https://duckduckgo.com/html/?q={quote_plus(scoped_query)}",
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
        if official_domains:
            results = [item for item in results if is_allowed_official_url(item.get("url", ""), official_domains)]
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
    official_domains = _OFFICIAL_MEDIA_DOMAINS.get()
    if official_domains and not is_allowed_official_url(url, official_domains):
        return "Error: URL is outside the configured official-media source list"
    max_chars = max(1000, min(max_chars, 50000))
    try:
        body, content_type = _fetch_url(url)
    except URLError as exc:
        return f"Error: Web fetch failed: {exc}"
    except Exception as exc:
        return f"Error: Web fetch failed: {type(exc).__name__}: {exc}"

    text = body if "text/plain" in content_type.lower() else _clean_html(body)
    return text[:max_chars]
