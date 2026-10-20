"""Fetch an article URL and extract its main text without persisting it."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
import trafilatura


MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
REQUEST_TIMEOUT_SECONDS = 15.0
USER_AGENT = "NewsLensArticleExtractor/0.1 (backend article text extraction)"


@dataclass
class ArticleExtractionError(Exception):
    message: str
    status_code: int = 502

    def __str__(self) -> str:
        return self.message


def _normalize_and_validate_url(raw_url: str) -> str:
    try:
        parsed = urlsplit(raw_url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ArticleExtractionError("The article URL is invalid.", 400) from exc

    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ArticleExtractionError("Only HTTP and HTTPS article URLs are supported.", 400)
    if parsed.username or parsed.password:
        raise ArticleExtractionError("URLs containing credentials are not allowed.", 400)
    if port not in {None, 80, 443}:
        raise ArticleExtractionError("Only standard HTTP and HTTPS ports are allowed.", 400)

    try:
        hostname = hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ArticleExtractionError("The article URL contains an invalid hostname.", 400) from exc
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        raise ArticleExtractionError("Local network URLs are not allowed.", 400)

    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(
                hostname,
                port or (443 if parsed.scheme.lower() == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except (OSError, UnicodeError) as exc:
        raise ArticleExtractionError("The article host could not be resolved.", 400) from exc

    if not addresses:
        raise ArticleExtractionError("The article host could not be resolved.", 400)

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise ArticleExtractionError("Local or non-public network URLs are not allowed.", 400)

    # Drop fragments; they are client-side and are never sent to the publisher.
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, ""))


def _fetch_html(url: str) -> tuple[str, str, bytes]:
    current_url = url
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=5.0)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8"}

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers=headers,
        ) as client:
            for redirect_count in range(MAX_REDIRECTS + 1):
                current_url = _normalize_and_validate_url(current_url)
                with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirect_count == MAX_REDIRECTS:
                            raise ArticleExtractionError("The article URL redirected too many times.", 502)
                        current_url = urljoin(current_url, location)
                        continue

                    if response.status_code < 200 or response.status_code >= 300:
                        raise ArticleExtractionError(
                            f"The publisher returned HTTP {response.status_code}.", 502
                        )

                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                        raise ArticleExtractionError("The URL did not return an HTML or text article.", 415)

                    content_length = response.headers.get("content-length")
                    if content_length and content_length.isdigit() and int(content_length) > MAX_RESPONSE_BYTES:
                        raise ArticleExtractionError("The article page is too large to process.", 413)

                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise ArticleExtractionError("The article page is too large to process.", 413)

                    encoding = response.encoding or "utf-8"
                    return current_url, encoding, bytes(body)
    except ArticleExtractionError:
        raise
    except httpx.TimeoutException as exc:
        raise ArticleExtractionError("The publisher took too long to respond.", 504) from exc
    except httpx.HTTPError as exc:
        raise ArticleExtractionError("The article page could not be fetched.", 502) from exc

    raise ArticleExtractionError("The article page could not be fetched.", 502)


def extract_article(url: str, max_words: int = 1500) -> dict:
    """Return article metadata and a bounded text excerpt; no content is stored."""
    requested_url = _normalize_and_validate_url(url)
    final_url, encoding, body = _fetch_html(requested_url)

    try:
        html = body.decode(encoding, errors="replace")
    except LookupError:
        html = body.decode("utf-8", errors="replace")

    article_text = trafilatura.extract(
        html,
        url=final_url,
        output_format="txt",
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not article_text or not article_text.strip():
        raise ArticleExtractionError("No readable article text could be extracted from this page.", 422)

    metadata = trafilatura.extract_metadata(html, default_url=final_url)
    words = article_text.split()
    excerpt = " ".join(words[:max_words])
    parsed_final_url = urlsplit(final_url)

    canonical_candidate = getattr(metadata, "url", None)
    canonical_url = final_url
    if canonical_candidate:
        try:
            canonical = urlsplit(canonical_candidate)
            if canonical.scheme.lower() in {"http", "https"} and canonical.hostname and not (
                canonical.username or canonical.password
            ):
                canonical_url = canonical_candidate
        except ValueError:
            pass

    # Canonical metadata is untrusted page content and is returned for display
    # only; it is never fetched as part of this request.
    return {
        "requested_url": requested_url,
        "canonical_url": canonical_url,
        "title": getattr(metadata, "title", None),
        "author": getattr(metadata, "author", None),
        "published_at": getattr(metadata, "date", None),
        "site_name": getattr(metadata, "sitename", None),
        "source_domain": parsed_final_url.hostname or "",
        "text": excerpt,
        "word_count": min(len(words), max_words),
        "total_word_count": len(words),
        "truncated": len(words) > max_words,
    }
