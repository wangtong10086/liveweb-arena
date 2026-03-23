"""Browser engine with session isolation for concurrent evaluations"""

import asyncio
import os
from dataclasses import asdict, dataclass, field
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from typing import Any, Optional, TYPE_CHECKING
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from .block_patterns import STEALTH_BROWSER_ARGS, STEALTH_USER_AGENT
from .models import BrowserObservation, BrowserAction
from ..utils.logger import log

if TYPE_CHECKING:
    from .interceptor import CacheInterceptor

# Constants
MAX_CONTENT_LENGTH = 20000  # Max content shown per view
VIEW_MORE_OVERLAP = 2000    # Overlap between views for context continuity
PAGE_TIMEOUT_MS = 30000
NAVIGATION_TIMEOUT_MS = 30000
_STOOQ_ONLY_DOMAINS = {"stooq.com", "www.stooq.com"}
_NON_HTML_FILE_EXTENSIONS = (
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".txt",
    ".pdf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".xlsx",
    ".xls",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
)
_NON_HTML_QUERY_VALUE_HINTS = {
    "attachment",
    "csv",
    "download",
    "file",
    "json",
    "pdf",
    "rss",
    "text",
    "tsv",
    "txt",
    "xhtml+xml",
    "xls",
    "xlsx",
    "xml",
    "zip",
}
_NON_HTML_QUERY_KEYS = {
    "attachment",
    "download",
    "export",
    "filename",
    "format",
    "output",
    "raw",
    "response-content-disposition",
    "type",
}

_BROWSER_TRANSPORT_ERROR_PATTERNS = (
    "handler is closed",
    "transport closed",
    "browser has been closed",
    "target page, context or browser has been closed",
    "connection closed",
    "browser.new_context",
)

_TAOSTATS_LIST_PAGINATE_SELECTORS = (
    ".ant-pagination-next",
    ".paginate_button.next",
    ".next.paginate_button",
    "#subnets-table_paginate .paginate_button.next",
    ".pagination-wrap .next-page",
    "li.page-item:nth-child(5) a.page-link",
)

_TAOSTATS_LIST_SHOW_ALL_SELECTORS = (
    '[data-testid="rows-select"]',
    ".ant-select-selector",
    "select",
)

_TAOSTATS_LIST_SORT_SELECTORS = (
    '.rt-th:nth-child(6)',
    'div.rt-th:has-text("1M")',
    'th:nth-child(7)',
)


def _browser_proxy_mode() -> str:
    return os.getenv("LIVEWEB_BROWSER_PROXY_MODE", "system").strip().lower()


def _browser_should_force_direct() -> bool:
    return _browser_proxy_mode() in {"direct", "no_proxy", "off"}


def _browser_should_direct_stooq() -> bool:
    return os.getenv("LIVEWEB_BROWSER_STOOQ_DIRECT", "0") == "1"


@dataclass
class BrowserNavigationMetadata:
    url: str
    normalized_url: str
    navigation_stage: str
    wait_until: str | None = None
    timeout_ms: int | None = None
    raw_exception_type: str | None = None
    raw_exception_message: str | None = None
    attempt_index: int = 1
    max_attempts: int = 1
    browser_reused: bool = True
    context_reused: bool = True
    page_recreated_before_retry: bool = False
    used_url_normalization: bool = False
    resource_type: str = "document"
    classification_hint: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrowserNavigationError(Exception):
    def __init__(self, message: str, metadata: BrowserNavigationMetadata):
        super().__init__(message)
        self.metadata = metadata


def is_browser_transport_error(exc: BaseException) -> bool:
    """Return True when an exception indicates a dead Playwright/browser transport."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in _BROWSER_TRANSPORT_ERROR_PATTERNS)


def _is_stooq_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return "stooq.com" in hostname


def _normalize_stooq_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if "stooq.com" not in hostname:
        return url

    query = parse_qs(parsed.query, keep_blank_values=True)
    if "q/currency" in parsed.path.lower():
        return urlunparse(parsed._replace(scheme="https", netloc="stooq.com"))

    if "q" in query:
        query["q"] = [query["q"][0].lower()]
    if "s" in query:
        query["s"] = [query["s"][0].lower()]
    if "e" in query:
        query["e"] = [query["e"][0].lower()]

    normalized_query = urlencode([(key, value) for key, values in query.items() for value in values], doseq=True)
    return urlunparse(parsed._replace(scheme="https", netloc="stooq.com", query=normalized_query))


def _looks_like_non_html_navigation_target(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return False

    path = parsed.path.lower()
    if path.endswith(_NON_HTML_FILE_EXTENSIONS):
        return True
    if any(marker in path for marker in ("/download", "/export", "/attachment")):
        return True

    query = parse_qs(parsed.query, keep_blank_values=True)
    for key, values in query.items():
        key_lower = key.lower()
        normalized_values = [value.strip().lower() for value in values if value is not None]
        if key_lower in _NON_HTML_QUERY_KEYS and any(
            not value or value in _NON_HTML_QUERY_VALUE_HINTS or "." in value
            for value in normalized_values
        ):
            return True
        if key_lower in {"e", "format", "output", "type"} and any(
            value in _NON_HTML_QUERY_VALUE_HINTS for value in normalized_values
        ):
            return True

    return False


def _classify_browser_exception(exc: BaseException) -> str | None:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "err_aborted" in text or "frame was detached" in text:
        return "env_nav_aborted"
    if "target page, context or browser has been closed" in text or "targetclosederror" in text:
        return "env_target_closed"
    if "timeout" in text:
        return "env_nav_timeout"
    if is_browser_transport_error(exc):
        return "env_browser_context_invalidated"
    return None


def _is_taostats_list_page_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower().rstrip("/")
    return "taostats.io" in host and path in {"", "/subnets"}


def _infer_taostats_list_target_kind(*, selector: str | None = None, role: str | None = None, name: str | None = None) -> str | None:
    text = " ".join(part for part in [selector or "", role or "", name or ""] if part).lower()
    if not text:
        return None
    if any(marker in text for marker in ("view all", "rows:", "rows-select", "ant-select-selector", "all", "customise table")):
        return "show_all"
    if any(marker in text for marker in ("pagination", "paginate", "page-link", "next-page", "next")) or (name or "").strip().isdigit():
        return "paginate"
    if any(marker in text for marker in ("1m", "1h", "24h", "1w", "sort", "rt-th", "th:nth-child")):
        return "sort"
    return None


def _should_fallback_to_direct_navigation(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "intercepts pointer events" in text
        or "waiting for element to be visible" in text
        or "timeout" in text
    )


class BrowserSession:
    """
    Isolated browser session (context + page).
    Each evaluate() call creates a new session to avoid state interference.

    In strict isolation mode, the session owns its own browser instance.
    """

    # Step size for view_more = viewport size minus overlap
    VIEW_STEP = MAX_CONTENT_LENGTH - VIEW_MORE_OVERLAP

    def __init__(
        self,
        context: BrowserContext,
        page: Page,
        browser: Browser = None,
        *,
        context_options: dict[str, Any] | None = None,
        browser_launch_options: dict[str, Any] | None = None,
    ):
        self._context = context
        self._page = page
        self._browser = browser  # Only set in strict isolation mode
        self._context_options = dict(context_options or {})
        self._browser_launch_options = dict(browser_launch_options or {})
        self._fallback_playwright: Playwright | None = None
        self._stooq_transport_mode = "default"
        # Virtual scroll state for handling truncated content
        self._view_offset = 0
        self._last_full_content = ""
        self._last_url = ""
        self._blocked_patterns = []
        self._allowed_domains = None  # None means allow all
        self._cache_interceptor: Optional["CacheInterceptor"] = None
        self._last_navigation_metadata: BrowserNavigationMetadata | None = None
        self._pending_navigation_override: BrowserObservation | None = None

    def get_last_navigation_metadata(self) -> dict[str, Any] | None:
        return self._last_navigation_metadata.to_dict() if self._last_navigation_metadata else None

    def clear_last_navigation_metadata(self) -> None:
        self._last_navigation_metadata = None
        self._pending_navigation_override = None

    def set_allowed_domains(self, domains: set[str] | list[str] | tuple[str, ...] | None) -> None:
        if not domains:
            self._allowed_domains = None
            return
        self._allowed_domains = {(domain or "").lower() for domain in domains if domain}

    def _record_action_failure_metadata(
        self,
        *,
        url: str,
        action_stage: str,
        exc: BaseException,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(exc, BrowserNavigationError):
            self._last_navigation_metadata = exc.metadata
            return
        self._last_navigation_metadata = BrowserNavigationMetadata(
            url=url,
            normalized_url=_normalize_stooq_url(url),
            navigation_stage=action_stage,
            timeout_ms=NAVIGATION_TIMEOUT_MS,
            raw_exception_type=type(exc).__name__,
            raw_exception_message=str(exc),
            attempt_index=1,
            max_attempts=1,
            browser_reused=self._browser is None,
            context_reused=True,
            page_recreated_before_retry=False,
            classification_hint=_classify_browser_exception(exc) or "ambiguous_navigation_failure",
            evidence=dict(evidence or {}),
        )

    async def _direct_nav_fallback_from_selector(self, selector: str) -> bool:
        try:
            element = await self._page.query_selector(selector)
            if not element:
                return False
            href = await element.get_attribute("href")
            if not href:
                return False
            target = urljoin(self._page.url, href)
            await self._goto_with_recovery(target)
            return True
        except Exception:
            return False

    async def _force_click_selector_fallback(self, selector: str, timeout_ms: int = 2000) -> bool:
        try:
            element = await self._page.query_selector(selector)
            if not element:
                return False
            try:
                await element.click(force=True, timeout=timeout_ms)
                return True
            except Exception:
                await element.evaluate("(el) => el.click()")
                return True
        except Exception:
            return False

    async def _direct_nav_fallback_from_locator(self, locator) -> bool:
        try:
            href = await locator.get_attribute("href")
            if not href:
                return False
            target = urljoin(self._page.url, href)
            await self._goto_with_recovery(target)
            return True
        except Exception:
            return False

    async def _force_click_locator_fallback(self, locator, timeout_ms: int = 2000) -> bool:
        try:
            first = locator.first if hasattr(locator, "first") else locator
            try:
                await first.click(force=True, timeout=timeout_ms)
                return True
            except Exception:
                await first.evaluate("(el) => el.click()")
                return True
        except Exception:
            return False

    async def _click_any_selector(self, selectors: list[str], timeout_ms: int = 2500) -> bool:
        seen: set[str] = set()
        for selector in selectors:
            if not selector or selector in seen:
                continue
            seen.add(selector)
            try:
                await self._page.click(selector, timeout=timeout_ms)
                return True
            except Exception:
                if await self._force_click_selector_fallback(selector, timeout_ms=min(timeout_ms, 1500)):
                    return True
        return False

    async def _taostats_list_selector_fallback(self, selector: str, timeout_ms: int) -> bool:
        if not _is_taostats_list_page_url(self._page.url):
            return False
        kind = _infer_taostats_list_target_kind(selector=selector)
        if kind == "show_all":
            return await self._click_any_selector(list(_TAOSTATS_LIST_SHOW_ALL_SELECTORS), timeout_ms=min(timeout_ms, 3000))
        if kind == "paginate":
            return await self._click_any_selector(list(_TAOSTATS_LIST_PAGINATE_SELECTORS), timeout_ms=min(timeout_ms, 3000))
        if kind == "sort":
            candidates = [selector, *_TAOSTATS_LIST_SORT_SELECTORS]
            return await self._click_any_selector(candidates, timeout_ms=min(timeout_ms, 3000))
        return False

    async def _taostats_list_role_fallback(self, role: str, name: str, timeout_ms: int) -> bool:
        if not _is_taostats_list_page_url(self._page.url):
            return False
        kind = _infer_taostats_list_target_kind(role=role, name=name)
        page_url = self._page.url
        if kind == "show_all":
            if "view all" in (name or "").lower() and page_url.rstrip("/") == "https://taostats.io":
                await self._goto_with_recovery("https://taostats.io/subnets")
                return True
            return await self._click_any_selector(list(_TAOSTATS_LIST_SHOW_ALL_SELECTORS), timeout_ms=min(timeout_ms, 3000))
        if kind == "paginate":
            return await self._click_any_selector(list(_TAOSTATS_LIST_PAGINATE_SELECTORS), timeout_ms=min(timeout_ms, 3000))
        if kind == "sort":
            return await self._click_any_selector(list(_TAOSTATS_LIST_SORT_SELECTORS), timeout_ms=min(timeout_ms, 3000))
        return False

    async def block_urls(self, patterns: list):
        """
        Block URLs matching the given patterns.

        Uses regex-based route interception to properly handle special characters.
        Playwright's glob pattern treats ? as single-char wildcard, but we need
        literal ? for URLs like *?format=*.

        Args:
            patterns: List of URL patterns (glob-style with * wildcard)
                     Example: ["*api.example.com*", "*?format=*"]
        """
        self._blocked_patterns.extend(patterns)
        await self._apply_block_patterns_to_context(self._context, patterns)

    async def _apply_block_patterns_to_context(self, context: BrowserContext, patterns: list[str]) -> None:
        import re

        # Build a combined regex for all patterns (more efficient than multiple routes)
        regex_patterns = []
        for pattern in patterns:
            # Convert glob to regex: escape special chars, then convert \* back to .*
            regex_pattern = re.escape(pattern).replace(r'\*', '.*')
            regex_patterns.append(regex_pattern)

        if regex_patterns:
            combined_regex = re.compile('|'.join(regex_patterns), re.IGNORECASE)

            async def block_handler(route):
                url = route.request.url
                if combined_regex.search(url):
                    await route.abort("blockedbyclient")
                else:
                    await route.continue_()

            # Use **/* to intercept all requests, filter by regex
            await self._context.route("**/*", block_handler)

    async def set_cache_interceptor(self, interceptor: "CacheInterceptor"):
        """
        Set up cache-based request interception.

        Routes all requests through the interceptor for cache handling.

        Args:
            interceptor: CacheInterceptor instance
        """
        from liveweb_arena.core.interceptor import CacheInterceptor
        self._cache_interceptor = interceptor

        # Route all requests through the interceptor
        await self._context.route("**/*", interceptor.handle_route)

    def _stooq_prefers_direct_transport(self, normalized_url: str) -> bool:
        if not _is_stooq_url(normalized_url):
            return False
        if self._allowed_domains is None:
            return True
        return bool(self._allowed_domains) and self._allowed_domains.issubset(_STOOQ_ONLY_DOMAINS)

    async def _switch_to_direct_stooq_browser(self) -> None:
        if not _browser_should_direct_stooq():
            return
        if self._stooq_transport_mode == "direct":
            return

        self._fallback_playwright = await async_playwright().start()
        launch_options = dict(self._browser_launch_options or {})
        launch_options.setdefault("headless", True)
        args = list(launch_options.get("args", []))
        if "--no-proxy-server" not in args:
            args.append("--no-proxy-server")
        launch_options["args"] = args
        browser = await self._fallback_playwright.chromium.launch(**launch_options)

        context = await browser.new_context(**self._context_options)
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        if self._cache_interceptor is not None:
            await context.route("**/*", self._cache_interceptor.handle_route)
        elif self._blocked_patterns:
            await self._apply_block_patterns_to_context(context, self._blocked_patterns)
        page = await context.new_page()

        old_page = self._page
        old_context = self._context
        old_browser = self._browser

        self._page = page
        self._context = context
        self._browser = browser
        self._stooq_transport_mode = "direct"

        try:
            await old_page.close()
        except Exception:
            pass
        try:
            await old_context.close()
        except Exception:
            pass
        if old_browser is not None and old_browser is not browser:
            try:
                await old_browser.close()
            except Exception:
                pass

    async def _stooq_page_has_meaningful_content(self) -> bool:
        current_url = self._page.url or ""
        if current_url.startswith(("about:blank", "chrome-error://", "about:neterror")):
            return False
        if not _is_stooq_url(current_url):
            return False
        try:
            title = (await self._page.title()).strip()
            body = await self._page.evaluate("""
                () => {
                    const body = document.body;
                    if (!body) return '';
                    return body.innerText || body.textContent || '';
                }
            """)
        except Exception:
            return False
        body = (body or "").strip()
        return bool(title) and len(body) >= 120

    def _should_preflight_navigation(self, url: str) -> bool:
        return _looks_like_non_html_navigation_target(url)

    async def _preflight_navigation_request(self, url: str) -> dict[str, Any] | None:
        if not url.startswith(("http://", "https://")):
            return None

        last_error: BaseException | None = None
        for attempt_index in range(1, 3):
            try:
                response = await self._context.request.get(
                    url,
                    fail_on_status_code=False,
                    max_redirects=5,
                    timeout=12000,
                )
                headers = {key.lower(): value for key, value in response.headers.items()}
                content_type = headers.get("content-type", "").lower()
                content_disposition = headers.get("content-disposition", "").lower()
                looks_like_html = (
                    "text/html" in content_type
                    or "application/xhtml+xml" in content_type
                    or not content_type
                )
                if not looks_like_html:
                    try:
                        preview = (await response.text())[:200]
                    except Exception:
                        preview = ""
                    preview_lower = preview.lstrip().lower()
                    looks_like_html = preview_lower.startswith("<!doctype html") or preview_lower.startswith("<html")
                return {
                    "status": response.status,
                    "final_url": response.url,
                    "content_type": content_type,
                    "content_disposition": content_disposition,
                    "is_html": looks_like_html,
                }
            except Exception as exc:
                last_error = exc
                if attempt_index == 1 and self._stooq_prefers_direct_transport(url) and self._stooq_transport_mode != "direct":
                    await self._switch_to_direct_stooq_browser()
                    continue
                return {
                    "preflight_error": str(exc),
                    "preflight_error_type": type(exc).__name__,
                }

        if last_error is None:
            return None
        return {
            "preflight_error": str(last_error),
            "preflight_error_type": type(last_error).__name__,
        }

    def _set_navigation_override(self, *, url: str, title: str, body: str, classification_hint: str, evidence: dict[str, Any]) -> None:
        self._last_navigation_metadata = BrowserNavigationMetadata(
            url=url,
            normalized_url=_normalize_stooq_url(url),
            navigation_stage="goto_preflight",
            timeout_ms=NAVIGATION_TIMEOUT_MS,
            raw_exception_type=None,
            raw_exception_message=None,
            attempt_index=1,
            max_attempts=1,
            browser_reused=self._browser is None,
            context_reused=True,
            page_recreated_before_retry=False,
            classification_hint=classification_hint,
            evidence=evidence,
        )
        self._pending_navigation_override = BrowserObservation(url=url, title=title, accessibility_tree=body)

    def _set_non_html_navigation_override(self, *, url: str, preflight: dict[str, Any]) -> None:
        evidence = {
            key: value
            for key, value in preflight.items()
            if value not in (None, "")
        }
        content_type = str(preflight.get("content_type", "") or "").lower()
        content_disposition = str(preflight.get("content_disposition", "") or "").lower()
        is_download = "attachment" in content_disposition or any(
            marker in content_type
            for marker in ("application/octet-stream", "text/csv", "application/pdf", "application/zip")
        )
        title = "Download" if is_download else "Non-HTML Response"
        summary = "file download" if is_download else "non-HTML response"
        self._set_navigation_override(
            url=url,
            title=title,
            body=(
                f"[This URL resolved to a {summary}, so the browser did not call page.goto().]\n\n"
                f"Original URL: {url}\n"
                f"Final URL: {preflight.get('final_url', url)}\n"
                f"HTTP status: {preflight.get('status', 'unknown')}\n"
                f"Content-Type: {preflight.get('content_type', '(unknown)')}\n"
                f"Content-Disposition: {preflight.get('content_disposition', '(none)')}"
            ),
            classification_hint="env_navigation_download" if is_download else "env_non_html_response",
            evidence=evidence,
        )

    async def goto(self, url: str) -> BrowserObservation:
        """Navigate to URL and return observation.

        Error pages (chrome-error://) are returned as valid observations
        so the AI can see them and decide what to do next.
        """
        # Reset view offset when navigating to a new page
        self._view_offset = 0
        self._last_full_content = ""
        self.clear_last_navigation_metadata()

        # Ensure URL has protocol prefix
        if url and not url.startswith(("http://", "https://", "about:")):
            url = "https://" + url

        try:
            await self._goto_with_recovery(url)
        except Exception as e:
            # Navigation failed — browser may show error page (chrome-error://).
            # Log but don't raise: _get_observation() detects error pages and
            # returns them as visible observations so the AI can react.
            log("Browser", f"Navigation failed for {url[:80]}: {type(e).__name__}: {e}")

        if self._pending_navigation_override is not None:
            obs = self._pending_navigation_override
            self._pending_navigation_override = None
            return obs

        # Return observation regardless of whether it's an error page
        # AI can see the error and decide what to do
        return await self._get_observation()

    async def execute_action(self, action: BrowserAction) -> BrowserObservation:
        """Execute browser action and return new observation.

        Error pages (chrome-error://) are returned as valid observations
        so the AI can see them and decide what to do next.
        """
        action_type = action.action_type
        params = action.params

        try:
            if action_type == "goto":
                url = params.get("url", "")
                # Ensure URL has protocol prefix
                if url and not url.startswith(("http://", "https://", "about:")):
                    url = "https://" + url
                # Navigate and return observation (including error pages)
                try:
                    await self._goto_with_recovery(url)
                except Exception as e:
                    log("Browser", f"Navigation failed for {url[:80]}: {type(e).__name__}: {e}")
                if self._pending_navigation_override is not None:
                    obs = self._pending_navigation_override
                    self._pending_navigation_override = None
                    return obs

            elif action_type == "click":
                selector = params.get("selector", "")
                timeout_ms = params.get("timeout_ms", 5000)
                clicked = False

                # First try the provided selector
                try:
                    await self._page.click(selector, timeout=timeout_ms)
                    clicked = True
                except Exception as click_err:
                    if await self._force_click_selector_fallback(selector, timeout_ms=min(timeout_ms, 2000)):
                        clicked = True
                    elif await self._taostats_list_selector_fallback(selector, timeout_ms):
                        clicked = True
                    elif _should_fallback_to_direct_navigation(click_err):
                        clicked = await self._direct_nav_fallback_from_selector(selector)

                    # If selector contains case-sensitive attribute match, try case-insensitive
                    if not clicked and ('[href*=' in selector or '[src*=' in selector):
                        import re
                        # Extract attribute and value: a[href*='GOOGL.US'] -> (href, GOOGL.US)
                        match = re.search(r"\[(\w+)\*=['\"]([^'\"]+)['\"]\]", selector, re.IGNORECASE)
                        if match:
                            attr_name = match.group(1).lower().replace("'", r"\'")
                            attr_value = match.group(2).lower().replace("'", r"\'")
                            # Use JavaScript to find element with case-insensitive match
                            element_handle = await self._page.evaluate_handle(f"""
                                () => {{
                                    const elements = document.querySelectorAll('a, button, [onclick]');
                                    for (const el of elements) {{
                                        const attr = el.getAttribute('{attr_name}');
                                        if (attr && attr.toLowerCase().includes('{attr_value}')) {{
                                            return el;
                                        }}
                                    }}
                                    return null;
                                }}
                            """)
                            if element_handle:
                                try:
                                    await element_handle.as_element().click()
                                    clicked = True
                                except Exception:
                                    pass

                    # If still not clicked, re-raise the original error
                    if not clicked:
                        raise click_err

                # Wait briefly for potential navigation
                await asyncio.sleep(0.3)

            elif action_type == "type":
                selector = params.get("selector", "")
                text = params.get("text", "")
                press_enter = params.get("press_enter", False)

                # Try provided selector first
                element = await self._page.query_selector(selector)

                # If selector doesn't match, try common fallbacks
                if not element:
                    fallback_selectors = [
                        'input[type="text"]:visible',
                        'input[type="search"]:visible',
                        'input:not([type="hidden"]):not([type="submit"]):visible',
                        'textarea:visible',
                        'input[name="s"]',  # Stooq search
                        'input[name="q"]',  # Common search name
                        '#search',
                        '[role="searchbox"]',
                    ]
                    for fallback in fallback_selectors:
                        try:
                            element = await self._page.query_selector(fallback)
                            if element:
                                selector = fallback
                                break
                        except Exception:
                            continue

                if element:
                    # Click first to trigger any onfocus/onclick handlers that set up form state
                    try:
                        await element.click()
                        await asyncio.sleep(0.1)
                    except Exception:
                        pass

                    # Fix form association for inputs not properly linked to their form
                    # (needed for cached pages where JS form setup hasn't run)
                    try:
                        await self._page.evaluate("""
                            (selector) => {
                                const input = document.querySelector(selector);
                                if (input && !input.form) {
                                    // Try to find and associate with nearest form
                                    const forms = document.forms;
                                    for (let i = 0; i < forms.length; i++) {
                                        const form = forms[i];
                                        if (form.id) {
                                            input.setAttribute('form', form.id);
                                            // Also set global form reference for Stooq-style JS
                                            if (typeof window.cmp_f === 'string' || !window.cmp_f) {
                                                window.cmp_f = form;
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        """, selector)
                    except Exception:
                        pass

                    await self._page.fill(selector, text)
                    if press_enter:
                        await self._page.press(selector, "Enter")
                        # Wait briefly for potential navigation after Enter
                        await asyncio.sleep(0.3)
                else:
                    raise Exception(f"No element found for selector '{selector}'")

            elif action_type == "press":
                key = params.get("key", "Enter")
                await self._page.keyboard.press(key)
                # Wait briefly for potential navigation
                await asyncio.sleep(0.3)

            elif action_type == "scroll":
                direction = params.get("direction", "down")
                amount = params.get("amount", 300)
                delta = amount if direction == "down" else -amount
                await self._page.mouse.wheel(0, delta)

            elif action_type == "view_more":
                # Virtual scrolling for truncated content - doesn't scroll the actual page
                direction = params.get("direction", "down")
                if direction == "down":
                    self._view_offset += self.VIEW_STEP
                else:
                    self._view_offset = max(0, self._view_offset - self.VIEW_STEP)

            elif action_type == "wait":
                seconds = params.get("seconds", 1)
                await asyncio.sleep(seconds)

            elif action_type == "click_role":
                role = params.get("role", "button")
                name = params.get("name", "")
                exact = params.get("exact", False)
                timeout_ms = params.get("timeout_ms", 5000)
                locator = self._page.get_by_role(role, name=name, exact=exact)
                count = await locator.count()

                # If no match with exact=True, try with exact=False
                if count == 0 and exact:
                    locator = self._page.get_by_role(role, name=name, exact=False)
                    count = await locator.count()

                # If still no match, try partial name match
                if count == 0 and name:
                    for keyword in name.split()[:3]:
                        if len(keyword) > 2:
                            partial_locator = self._page.get_by_role(role, name=keyword, exact=False)
                            partial_count = await partial_locator.count()
                            if partial_count > 0:
                                locator = partial_locator.first
                                count = 1
                                break

                if count > 0:
                    try:
                        await locator.click(timeout=timeout_ms)
                    except Exception as click_err:
                        if await self._force_click_locator_fallback(locator):
                            pass
                        elif await self._taostats_list_role_fallback(role, name, timeout_ms):
                            pass
                        elif _should_fallback_to_direct_navigation(click_err) and await self._direct_nav_fallback_from_locator(locator):
                            pass
                        else:
                            raise
                    # Wait briefly for potential navigation
                    await asyncio.sleep(0.3)
                else:
                    if await self._taostats_list_role_fallback(role, name, timeout_ms):
                        await asyncio.sleep(0.3)
                    else:
                        raise Exception(f"No element found with role='{role}' name='{name}'")

            elif action_type == "type_role":
                role = params.get("role", "textbox")
                name = params.get("name", "")
                text = params.get("text", "")
                press_enter = params.get("press_enter", False)

                # Try exact name match first
                locator = self._page.get_by_role(role, name=name)
                count = await locator.count()

                # If no match with given name, try fallbacks for textbox
                if count == 0 and role == "textbox":
                    # Fallback 1: Try common search input selectors
                    search_selectors = [
                        'input[name="s"]',   # Stooq search
                        'input[name="q"]',   # Common search name
                        'input[type="search"]',
                        '#search',
                        '[role="searchbox"]',
                    ]
                    for selector in search_selectors:
                        try:
                            el = await self._page.query_selector(selector)
                            if el:
                                locator = self._page.locator(selector)
                                count = 1
                                break
                        except Exception:
                            continue

                    # Fallback 2: Try partial name match
                    if count == 0 and name:
                        for keyword in name.split()[:3]:
                            if len(keyword) > 2:
                                partial_locator = self._page.get_by_role(role, name=keyword)
                                partial_count = await partial_locator.count()
                                if partial_count > 0:
                                    locator = partial_locator.first
                                    count = 1
                                    break

                    # Fallback 3: Use first visible textbox
                    if count == 0:
                        empty_locator = self._page.get_by_role(role, name="")
                        empty_count = await empty_locator.count()
                        if empty_count > 0:
                            locator = empty_locator.first
                            count = 1

                if count > 0:
                    # Click first to trigger any onfocus/onclick handlers
                    try:
                        await locator.click()
                        await asyncio.sleep(0.1)
                    except Exception:
                        pass

                    # Fix form association for inputs not properly linked to their form
                    # Also fix window.cmp_f for Stooq-style sites where JS expects form reference
                    try:
                        await self._page.evaluate("""
                            () => {
                                const inputs = document.querySelectorAll('input[type="text"], input[type="search"]');
                                inputs.forEach(input => {
                                    if (!input.form) {
                                        const forms = document.forms;
                                        for (let i = 0; i < forms.length; i++) {
                                            const form = forms[i];
                                            if (form.id) {
                                                input.setAttribute('form', form.id);
                                                // Also set global form reference for Stooq-style JS
                                                if (typeof window.cmp_f === 'string' || !window.cmp_f) {
                                                    window.cmp_f = form;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                });
                            }
                        """)
                    except Exception:
                        pass

                    await locator.fill(text)
                    if press_enter:
                        original_url = self._page.url
                        await locator.press("Enter")
                        # Wait briefly for potential navigation
                        await asyncio.sleep(0.5)

                        # If still on same page, try calling the JS redirect directly
                        # (handles cached pages where form handlers fail)
                        if self._page.url == original_url and text:
                            try:
                                # Try calling Stooq's cmp_u function directly
                                import json
                                safe_text = json.dumps(text)
                                await self._page.evaluate(f"""
                                    () => {{
                                        const t = {safe_text};
                                        if (typeof cmp_u === 'function') {{
                                            cmp_u(t);
                                        }} else {{
                                            // Fallback: direct navigation for search-style inputs
                                            const url = window.location.origin;
                                            if (url.includes('stooq')) {{
                                                window.location.href = url + '/q/?s=' + encodeURIComponent(t);
                                            }}
                                        }}
                                    }}
                                """)
                                # Wait for URL to actually change (navigation is async)
                                for _ in range(10):
                                    await asyncio.sleep(0.3)
                                    if self._page.url != original_url:
                                        break
                            except Exception:
                                pass
                else:
                    raise Exception(f"No element found with role='{role}' name='{name}'")

            elif action_type == "stop":
                # Stop action - no browser operation needed
                pass

            else:
                raise ValueError(f"Unknown action type: {action_type}")

        except Exception as e:
            # Re-raise action execution errors so agent_loop can report failure
            action_evidence = {}
            if action_type == "click":
                action_evidence = {"selector": params.get("selector", "")}
            elif action_type == "click_role":
                action_evidence = {
                    "role": params.get("role", "button"),
                    "name": params.get("name", ""),
                    "exact": params.get("exact", False),
                }
            self._record_action_failure_metadata(
                url=self._page.url or params.get("url", "") or self._last_url or "about:blank",
                action_stage=f"action_{action_type}",
                exc=e,
                evidence=action_evidence,
            )
            raise

        return await self._get_observation()

    async def _goto_with_recovery(self, url: str) -> None:
        normalized_url = _normalize_stooq_url(url)
        if self._stooq_prefers_direct_transport(normalized_url):
            await self._switch_to_direct_stooq_browser()

        preflight = None
        if self._should_preflight_navigation(url):
            preflight = await self._preflight_navigation_request(url)
        if preflight and not preflight.get("is_html", True):
            self._set_non_html_navigation_override(url=url, preflight=preflight)
            return

        max_attempts = 2 if _is_stooq_url(normalized_url) else 1
        for attempt_index in range(1, max_attempts + 1):
            wait_until = "domcontentloaded" if attempt_index == 1 else "commit"
            page_recreated = False
            if attempt_index > 1:
                log("Browser", f"Retrying unstable navigation for {normalized_url[:80]}")
                try:
                    await self._page.close()
                except Exception:
                    pass
                self._page = await self._context.new_page()
                page_recreated = True
                await asyncio.sleep(0.2)
            document_failures: list[dict[str, Any]] = []
            download_events: list[dict[str, Any]] = []

            def _on_request_failed(req):
                if req.resource_type != "document":
                    return
                failure = req.failure
                if isinstance(failure, dict):
                    error_text = failure.get("errorText")
                else:
                    error_text = str(failure) if failure is not None else None
                document_failures.append(
                    {
                        "url": req.url,
                        "error_text": error_text,
                    }
                )

            def _on_download(download):
                download_events.append(
                    {
                        "url": download.url,
                        "suggested_filename": download.suggested_filename,
                    }
                )

            self._page.on("requestfailed", _on_request_failed)
            self._page.on("download", _on_download)
            try:
                await self._page.goto(normalized_url, wait_until=wait_until, timeout=NAVIGATION_TIMEOUT_MS)
                try:
                    await self._page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass
                self.clear_last_navigation_metadata()
                return
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}".lower()
                classification_hint = None
                if "err_aborted" in text or "frame was detached" in text:
                    classification_hint = "env_nav_aborted"
                elif "target page, context or browser has been closed" in text or "targetclosederror" in text:
                    classification_hint = "env_target_closed"
                elif "timeout" in text:
                    classification_hint = "env_nav_timeout"
                    if _is_stooq_url(normalized_url) and await self._stooq_page_has_meaningful_content():
                        self.clear_last_navigation_metadata()
                        return
                elif is_browser_transport_error(exc):
                    classification_hint = "env_browser_context_invalidated"
                if download_events or "download is starting" in text:
                    classification_hint = "env_navigation_download"
                if preflight is None and (
                    classification_hint == "env_navigation_download"
                    or download_events
                ):
                    preflight = await self._preflight_navigation_request(url)
                    if preflight and not preflight.get("is_html", True):
                        self._set_non_html_navigation_override(url=url, preflight=preflight)
                        return
                self._last_navigation_metadata = BrowserNavigationMetadata(
                    url=url,
                    normalized_url=normalized_url,
                    navigation_stage=f"goto_{wait_until}",
                    wait_until=wait_until,
                    timeout_ms=NAVIGATION_TIMEOUT_MS,
                    raw_exception_type=type(exc).__name__,
                    raw_exception_message=str(exc),
                    attempt_index=attempt_index,
                    max_attempts=max_attempts,
                    browser_reused=self._browser is None,
                    context_reused=True,
                    page_recreated_before_retry=page_recreated,
                    used_url_normalization=(normalized_url != url),
                    classification_hint=classification_hint,
                    evidence={
                        "preflight": preflight or {},
                        "document_request_failures": document_failures[:3],
                        "download_events": download_events[:3],
                    },
                )
                if attempt_index == max_attempts or not _is_stooq_url(normalized_url) or classification_hint not in {"env_nav_aborted", "env_target_closed"}:
                    raise BrowserNavigationError(str(exc), self._last_navigation_metadata) from exc
            finally:
                self._page.remove_listener("requestfailed", _on_request_failed)
                self._page.remove_listener("download", _on_download)

    async def get_observation(self, max_retries: int = 3) -> BrowserObservation:
        """Get current browser observation with retry logic for navigation timing"""
        return await self._get_observation(max_retries)

    async def _get_observation(self, max_retries: int = 5) -> BrowserObservation:
        """Get current browser observation with retry logic for page loading.

        Key improvements:
        1. Validates content is meaningful before returning to AI
        2. Retries if content is empty/too short (page still loading)
        3. Returns clear error messages for blocked/failed pages
        """
        MIN_VALID_CONTENT_LENGTH = 50  # Minimum chars for valid content

        for attempt in range(max_retries):
            try:
                url = self._page.url

                # Check for error pages - recover browser state via goBack()
                if url.startswith("chrome-error://") or url.startswith("about:neterror"):
                    # goBack() restores the browser to the previous valid page,
                    # preventing cascading errors on subsequent actions.
                    try:
                        await self._page.go_back(timeout=5000)
                    except Exception:
                        pass
                    return BrowserObservation(
                        url=url,
                        title="Error",
                        accessibility_tree="[Page failed to load - network error. Try a different URL.]",
                    )

                # Check for blocked pages (request was aborted)
                if url == "about:blank" and attempt > 0:
                    # Likely blocked by pattern - check if we have a pending URL
                    return BrowserObservation(
                        url=url,
                        title="Blocked",
                        accessibility_tree="[Navigation was blocked. The URL may be restricted. Try using the main website instead of API endpoints.]",
                    )

                # Wait for page to be fully loaded with increased timeout
                page_loaded = False
                try:
                    await self._page.wait_for_load_state("networkidle", timeout=15000)
                    page_loaded = True
                except Exception:
                    # Network idle timeout - page might still be loading
                    # Try domcontentloaded as fallback
                    try:
                        await self._page.wait_for_load_state("domcontentloaded", timeout=5000)
                        page_loaded = True
                    except Exception:
                        pass

                # If page not loaded and we have retries left, wait and retry
                if not page_loaded and attempt < max_retries - 1:
                    await asyncio.sleep(1.5)
                    continue

                title = await self._page.title()

                # Check for cached accessibility tree first (deterministic in cache mode)
                cached_tree = self._cache_interceptor.get_accessibility_tree(url) if self._cache_interceptor else None
                if cached_tree:
                    full_content = cached_tree
                else:
                    # Get accessibility tree from live page
                    a11y_tree = ""
                    try:
                        a11y_snapshot = await self._page.accessibility.snapshot()
                        if a11y_snapshot:
                            a11y_tree = self._format_accessibility_tree(a11y_snapshot)
                    except Exception:
                        pass

                    # If accessibility tree is empty or too short, get page text content
                    # This handles sites like wttr.in that use <pre> tags and ASCII art
                    page_text = ""
                    if len(a11y_tree.strip()) < 100:
                        try:
                            # Get visible text content from the page
                            page_text = await self._page.evaluate("""
                                () => {
                                    // Try to get text from pre elements first (for ASCII art sites)
                                    const preElements = document.querySelectorAll('pre');
                                    if (preElements.length > 0) {
                                        return Array.from(preElements).map(el => el.innerText).join('\\n');
                                    }
                                    // Fall back to body text
                                    return document.body.innerText || '';
                                }
                            """)
                        except Exception:
                            pass

                    # Combine accessibility tree and page text
                    full_content = ""
                    if a11y_tree.strip():
                        full_content = a11y_tree
                    if page_text.strip():
                        if full_content:
                            full_content += "\n\n--- Page Text Content ---\n" + page_text
                        else:
                            full_content = page_text

                # Content validation: if content is too short, page may still be loading
                content_length = len(full_content.strip())
                if content_length < MIN_VALID_CONTENT_LENGTH and attempt < max_retries - 1:
                    # Wait longer and retry - page content not yet available
                    await asyncio.sleep(2.0)
                    continue

                # If content is still empty after all retries, provide helpful message
                if content_length < MIN_VALID_CONTENT_LENGTH:
                    full_content = f"[Page appears empty or content is minimal ({content_length} chars). The page may be:\n" \
                                   f"- Still loading (try scrolling or waiting)\n" \
                                   f"- Blocked (try the main website instead of API endpoints)\n" \
                                   f"- Requiring JavaScript that failed to load\n" \
                                   f"Current URL: {url}]\n\n{full_content}"

                # Store full content and check if URL changed (reset offset if so)
                if url != self._last_url:
                    self._view_offset = 0
                    self._last_url = url
                self._last_full_content = full_content

                # Apply virtual scrolling with view window
                total_len = len(full_content)
                if total_len > MAX_CONTENT_LENGTH:
                    # Clamp view offset to valid range
                    max_offset = max(0, total_len - MAX_CONTENT_LENGTH)
                    self._view_offset = min(self._view_offset, max_offset)

                    # Extract window of content
                    start = self._view_offset
                    end = min(start + MAX_CONTENT_LENGTH, total_len)
                    content = full_content[start:end]

                    # Add position indicators
                    position_info = []
                    if start > 0:
                        position_info.append(f"... (content above, use view_more direction=up to see)")
                    if end < total_len:
                        position_info.append(f"... (content below, use view_more direction=down to see)")

                    if position_info:
                        content = "\n".join(position_info[:1]) + "\n" + content
                        if len(position_info) > 1:
                            content += "\n" + position_info[1]
                        # Add clear truncation notice
                        content += "\n\n[Page content truncated - use view_more action to see more content]"
                else:
                    # Content fits in one view - no scrolling needed
                    content = full_content + "\n\n[Page content complete - no need to scroll]"

                return BrowserObservation(
                    url=url,
                    title=title,
                    accessibility_tree=content,
                )

            except Exception as e:
                # Execution context destroyed - page is navigating
                if attempt < max_retries - 1:
                    # Wait a bit and retry
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # Final attempt failed - raise error instead of returning empty observation
                    # Empty observation would affect agent decisions and GT collection
                    metadata = BrowserNavigationMetadata(
                        url=self._page.url,
                        normalized_url=_normalize_stooq_url(self._page.url),
                        navigation_stage="observation_fetch",
                        timeout_ms=15000,
                        raw_exception_type=type(e).__name__,
                        raw_exception_message=str(e),
                        attempt_index=attempt + 1,
                        max_attempts=max_retries,
                        browser_reused=self._browser is None,
                        context_reused=True,
                        page_recreated_before_retry=False,
                        classification_hint="env_nav_timeout" if "timeout" in f"{type(e).__name__}: {e}".lower() else "ambiguous_navigation_failure",
                    )
                    self._last_navigation_metadata = metadata
                    raise BrowserNavigationError(f"Failed to get browser observation after {max_retries} retries: {e}", metadata) from e

    def _format_accessibility_tree(self, node: dict, indent: int = 0) -> str:
        """Format accessibility tree node recursively"""
        if not node:
            return ""

        lines = []
        prefix = "  " * indent

        role = node.get("role", "")
        name = node.get("name", "")
        value = node.get("value", "")

        # Build node representation
        parts = [role]
        if name:
            parts.append(f'"{name}"')
        if value:
            parts.append(f'value="{value}"')

        lines.append(f"{prefix}{' '.join(parts)}")

        # Process children
        children = node.get("children", [])
        for child in children:
            lines.append(self._format_accessibility_tree(child, indent + 1))

        return "\n".join(lines)

    async def close(self):
        """Close session (context, page, and browser if in strict mode)"""
        # Clear large content to release memory
        self._last_full_content = ""
        self._cache_interceptor = None

        try:
            await self._page.close()
        except Exception:
            pass
        try:
            # Closing context will save HAR file if recording was enabled
            await self._context.close()
        except Exception:
            pass
        # In strict isolation mode, also close the browser instance
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._fallback_playwright is not None:
            try:
                await self._fallback_playwright.stop()
            except Exception:
                pass
            self._fallback_playwright = None


class BrowserEngine:
    """
    Browser engine that manages Playwright and Browser instances.

    Supports two isolation modes:
    - shared: Single browser instance, isolated contexts (default, faster)
    - strict: Separate browser instance per session (stronger isolation)
    """

    def __init__(self, headless: bool = True, isolation_mode: str = "shared"):
        """
        Initialize browser engine.

        Args:
            headless: Run browser in headless mode
            isolation_mode: "shared" (default) or "strict"
                - shared: Single browser, separate contexts (faster, good for most cases)
                - strict: Separate browser per session (stronger isolation, slower)
        """
        self._headless = headless
        self._isolation_mode = isolation_mode
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self._browser_args = [
            *STEALTH_BROWSER_ARGS,
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ]
        if _browser_should_force_direct() and "--no-proxy-server" not in self._browser_args:
            self._browser_args.append("--no-proxy-server")
        self._dirty = False

    def _launch_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "headless": self._headless,
            "args": list(self._browser_args),
        }
        if self._headless:
            # Use Chromium's "new headless" instead of the legacy headless shell.
            options["channel"] = "chromium"
        return options

    def _context_options(self) -> dict[str, Any]:
        return {
            "viewport": {"width": 1280, "height": 720},
            "user_agent": STEALTH_USER_AGENT,
            "ignore_https_errors": False,
            "java_script_enabled": True,
            "bypass_csp": False,
            "accept_downloads": False,
        }

    def mark_dirty(self):
        """Mark the shared browser state as unhealthy so the next session rebuilds it."""
        self._dirty = True

    def is_alive(self) -> bool:
        """Best-effort health check for the shared browser transport."""
        if self._playwright is None:
            return False
        if self._isolation_mode == "strict":
            return True
        if self._browser is None or self._dirty:
            return False
        try:
            return bool(self._browser.is_connected())
        except Exception:
            return False

    async def ensure_healthy(self):
        """Ensure the underlying browser transport is healthy, rebuilding if needed."""
        if self._isolation_mode == "strict":
            if self._playwright is None:
                await self.start()
            return

        if self.is_alive():
            return

        await self.stop()
        await self.start()

    async def start(self):
        """Start Playwright and launch browser (for shared mode)"""
        async with self._lock:
            if self._playwright is None:
                self._playwright = await async_playwright().start()

            if self._isolation_mode == "shared" and self._browser is None:
                self._browser = await self._playwright.chromium.launch(**self._launch_options())
            self._dirty = False

    async def new_session(self) -> BrowserSession:
        """
        Create a new isolated browser session.

        Returns:
            BrowserSession instance
        """
        await self.ensure_healthy()

        # Prepare context options
        context_options = self._context_options()
        launch_options = self._launch_options()

        if self._isolation_mode == "strict":
            browser = await self._playwright.chromium.launch(**launch_options)
            context = await browser.new_context(**context_options)
            context.set_default_timeout(PAGE_TIMEOUT_MS)
            page = await context.new_page()
            return BrowserSession(
                context,
                page,
                browser=browser,
                context_options=context_options,
                browser_launch_options=launch_options,
            )
        else:
            if self._browser is None:
                await self.start()

            for attempt in range(2):
                try:
                    context = await self._browser.new_context(**context_options)
                    context.set_default_timeout(PAGE_TIMEOUT_MS)
                    page = await context.new_page()
                    return BrowserSession(
                        context,
                        page,
                        context_options=context_options,
                        browser_launch_options=launch_options,
                    )
                except Exception as exc:
                    if attempt == 0 and is_browser_transport_error(exc):
                        log("Browser", f"Shared browser unhealthy during new_session(), rebuilding: {exc}", force=True)
                        self.mark_dirty()
                        await self.ensure_healthy()
                        continue
                    raise

    async def stop(self):
        """Stop browser and Playwright with timeout"""
        try:
            # 使用超时避免无限等待锁
            async with asyncio.timeout(5):
                async with self._lock:
                    if self._browser:
                        try:
                            await asyncio.wait_for(self._browser.close(), timeout=3)
                        except Exception:
                            pass
                        self._browser = None

                    if self._playwright:
                        try:
                            await asyncio.wait_for(self._playwright.stop(), timeout=3)
                        except Exception:
                            pass
                        self._playwright = None
                    self._dirty = False
        except asyncio.TimeoutError:
            # 超时则强制清理引用
            self._browser = None
            self._playwright = None
            self._dirty = False
