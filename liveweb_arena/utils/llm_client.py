"""OpenAI-compatible LLM client with retry, streaming, tool calls, and multi-server routing."""

import asyncio
import email.utils
import ipaddress
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai

from .logger import is_verbose, log, progress, progress_done


class LLMFatalError(Exception):
    """
    Raised when LLM errors exhaust all retries.

    This indicates an unrecoverable error that should terminate evaluation
    immediately rather than continuing with degraded results.
    """

    def __init__(self, message: str, original_error: Exception = None, attempts: int = 0):
        super().__init__(message)
        self.original_error = original_error
        self.attempts = attempts


@dataclass
class ToolCall:
    """Parsed tool call from LLM response."""

    id: str
    function: dict  # {"name": str, "arguments": str}


@dataclass
class LLMResponse:
    """Structured LLM response supporting both text and tool_calls."""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[dict] = None
    request_id: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class LLMServerConfig:
    """Static config for one OpenAI-compatible endpoint."""

    server_id: str
    base_url: str
    api_key: str
    model_name: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None


@dataclass
class _LLMServerState:
    config: LLMServerConfig
    inflight: int = 0
    requests: int = 0
    failures: int = 0
    total_latency_s: float = 0.0


@dataclass
class _ServerLease:
    server_id: str
    base_url: str
    api_key: str
    started_at: float


class MultiServerLLMRouter:
    """
    Route LLM requests across multiple OpenAI-compatible servers.

    The default policy is sticky + steal:
    - assign each route_key a preferred server for cache locality
    - allow spilling to the least-loaded server when the preferred one is busier
    """

    def __init__(
        self,
        servers: List[LLMServerConfig],
        route_policy: str = "sticky_steal",
        max_inflight_requests: Optional[int] = None,
        sticky_slack: int = 0,
        sticky_latency_slack_s: float = 10.0,
    ):
        if not servers:
            raise ValueError("MultiServerLLMRouter requires at least one server")

        self._states: Dict[str, _LLMServerState] = {
            server.server_id: _LLMServerState(config=server)
            for server in servers
        }
        self._order: List[str] = [server.server_id for server in servers]
        self._route_policy = route_policy
        self._sticky_slack = max(0, sticky_slack)
        self._sticky_latency_slack_s = max(0.0, sticky_latency_slack_s)
        self._preferred_server: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._global_semaphore = (
            asyncio.Semaphore(max_inflight_requests)
            if max_inflight_requests and max_inflight_requests > 0
            else None
        )

    @property
    def base_urls(self) -> List[str]:
        return [self._states[server_id].config.base_url for server_id in self._order]

    @property
    def primary_base_url(self) -> str:
        return self.base_urls[0]

    @classmethod
    def from_server_pool_file(
        cls,
        path: str,
        route_policy: str = "sticky_steal",
        max_inflight_requests: Optional[int] = None,
        sticky_slack: int = 0,
        sticky_latency_slack_s: float = 10.0,
        default_api_key: Optional[str] = None,
    ) -> "MultiServerLLMRouter":
        import json
        from pathlib import Path

        payload = json.loads(Path(path).read_text())
        servers = []
        for idx, entry in enumerate(payload.get("servers", [])):
            server_id = str(entry.get("server_id") or entry.get("id") or f"server-{idx}")
            api_key = str(entry.get("api_key") or default_api_key or "")
            if not api_key:
                raise ValueError(f"Server {server_id} missing api_key")
            servers.append(
                LLMServerConfig(
                    server_id=server_id,
                    base_url=str(entry["base_url"]).rstrip("/"),
                    api_key=api_key,
                    model_name=entry.get("model_name"),
                    metadata=entry,
                )
            )
        return cls(
            servers=servers,
            route_policy=route_policy,
            max_inflight_requests=max_inflight_requests,
            sticky_slack=sticky_slack,
            sticky_latency_slack_s=sticky_latency_slack_s,
        )

    async def acquire(self, route_key: Optional[str] = None) -> _ServerLease:
        if self._global_semaphore is not None:
            await self._global_semaphore.acquire()

        async with self._lock:
            selected_id = self._select_server_id(route_key)
            state = self._states[selected_id]
            state.inflight += 1
            state.requests += 1
            if route_key:
                self._preferred_server.setdefault(route_key, selected_id)
            return _ServerLease(
                server_id=selected_id,
                base_url=state.config.base_url,
                api_key=state.config.api_key,
                started_at=time.time(),
            )

    async def release(self, lease: _ServerLease, success: bool, latency_s: float):
        async with self._lock:
            state = self._states[lease.server_id]
            state.inflight = max(0, state.inflight - 1)
            state.total_latency_s += max(0.0, latency_s)
            if not success:
                state.failures += 1

        if self._global_semaphore is not None:
            self._global_semaphore.release()

    def snapshot(self) -> Dict[str, object]:
        servers = []
        for server_id in self._order:
            state = self._states[server_id]
            avg_latency = state.total_latency_s / state.requests if state.requests else 0.0
            servers.append(
                {
                    "server_id": server_id,
                    "base_url": state.config.base_url,
                    "inflight": state.inflight,
                    "requests": state.requests,
                    "failures": state.failures,
                    "avg_latency_s": avg_latency,
                }
            )
        return {
            "route_policy": self._route_policy,
            "num_servers": len(servers),
            "servers": servers,
        }

    def _select_server_id(self, route_key: Optional[str]) -> str:
        states = [self._states[server_id] for server_id in self._order]
        least_loaded = min(states, key=self._state_sort_key)

        if self._route_policy != "sticky_steal" or not route_key:
            return least_loaded.config.server_id

        preferred_id = self._preferred_server.get(route_key)
        if preferred_id is None:
            return least_loaded.config.server_id

        preferred = self._states.get(preferred_id)
        if preferred is None:
            return least_loaded.config.server_id

        preferred_latency = self._avg_latency(preferred)
        least_loaded_latency = self._avg_latency(least_loaded)

        if (
            preferred.inflight <= least_loaded.inflight + self._sticky_slack
            and preferred_latency <= least_loaded_latency + self._sticky_latency_slack_s
        ):
            return preferred.config.server_id

        self._preferred_server[route_key] = least_loaded.config.server_id
        return least_loaded.config.server_id

    @staticmethod
    def _avg_latency(state: _LLMServerState) -> float:
        return state.total_latency_s / state.requests if state.requests else 0.0

    def _state_sort_key(self, state: _LLMServerState):
        latency_penalty = min(self._avg_latency(state) / 60.0, 5.0)
        return (state.inflight + latency_penalty, state.failures, state.requests)


class LLMClient:
    """
    OpenAI-compatible LLM client.

    Features:
    - Streaming support with usage tracking
    - Exponential backoff retry for recoverable errors
    - Configurable timeouts
    - Optional multi-server routing with sticky + steal
    """

    RETRY_STATUS_CODES = {429, 503, 502, 500}
    MAX_RETRIES = 10
    BASE_DELAY = 1.0
    MAX_DELAY = 30.0
    RATE_LIMIT_MAX_DELAY = float(os.getenv("LIVEWEB_LLM_RATE_LIMIT_MAX_DELAY", "120"))
    DEFAULT_TIMEOUT = 600
    MAX_CHUNKS = int(os.getenv("LIVEWEB_LLM_MAX_CHUNKS", "32000"))
    _GLOBAL_RATE_LIMIT_UNTIL: Dict[str, float] = {}
    _CONTEXT_LENGTH_DETAILS_RE = re.compile(
        r"maximum context length of (\d+) tokens.*?"
        r"requested a total of (\d+) tokens:\s*"
        r"(\d+) tokens from the input messages and (\d+) tokens for the completion",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_timeout: int = None,
        router: Optional[MultiServerLLMRouter] = None,
        route_key: Optional[str] = None,
        max_retries: Optional[int] = None,
        strict_serial: bool = False,
        enable_thinking: Optional[bool] = None,
        separate_reasoning: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        strip_reasoning_output: bool = False,
    ):
        if router is None and (not base_url or not api_key):
            raise ValueError("LLMClient requires either router or both base_url/api_key")

        self._router = router
        self._route_key = route_key
        self._base_url = base_url.rstrip("/") if base_url else (router.primary_base_url if router else "")
        self._base_urls = router.base_urls if router else [self._base_url]
        self._api_key = api_key
        self._default_timeout = default_timeout or self.DEFAULT_TIMEOUT
        self._strict_serial = strict_serial
        self._max_retries = max(1, int(max_retries or self.MAX_RETRIES))
        self._max_completion_tokens = self._read_max_completion_tokens()
        self._enable_thinking = (
            enable_thinking if enable_thinking is not None else self._read_optional_bool_env("LIVEWEB_ENABLE_THINKING")
        )
        self._separate_reasoning = (
            separate_reasoning
            if separate_reasoning is not None
            else self._read_optional_bool_env("LIVEWEB_SEPARATE_REASONING")
        )
        self._reasoning_effort = (
            reasoning_effort.strip().lower()
            if isinstance(reasoning_effort, str) and reasoning_effort.strip()
            else (os.getenv("LIVEWEB_REASONING_EFFORT", "").strip().lower() or None)
        )
        self._strip_reasoning_output = bool(
            strip_reasoning_output or (os.getenv("LIVEWEB_STRIP_REASONING_OUTPUT", "0") == "1")
        )
        self._format_recovery_temperature = self._read_float_env("LIVEWEB_FORMAT_RECOVERY_TEMPERATURE", 0.35)
        self._format_recovery_top_p = self._read_float_env("LIVEWEB_FORMAT_RECOVERY_TOP_P", 0.95)
        self._last_failure_metadata: Dict[str, object] = {}
        if self._strict_serial:
            self._max_retries = 1

    def _raise_strict_serial_error(self, message: str, error: Exception, attempts: int) -> None:
        raise LLMFatalError(
            message,
            original_error=error,
            attempts=attempts,
        ) from error

    def _generate_request_id(self, request_kind: str) -> str:
        route_key = (self._route_key or "default").replace(":", "-").replace("/", "-")
        route_key = route_key[:48]
        return f"liveweb-{request_kind}-{route_key}-{uuid.uuid4().hex[:12]}"

    def _set_last_failure_metadata(self, metadata: Optional[Dict[str, object]]) -> None:
        self._last_failure_metadata = dict(metadata or {})

    def get_last_failure_metadata(self) -> Dict[str, object]:
        return dict(self._last_failure_metadata)

    @staticmethod
    def _read_max_completion_tokens() -> Optional[int]:
        raw = os.getenv("LIVEWEB_MAX_COMPLETION_TOKENS")
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            log("LLM", f"Invalid LIVEWEB_MAX_COMPLETION_TOKENS={raw!r}; ignoring")
            return None
        return value if value > 0 else None

    @staticmethod
    def _read_optional_bool_env(name: str) -> Optional[bool]:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return None
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        log("LLM", f"Invalid {name}={raw!r}; ignoring")
        return None

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            log("LLM", f"Invalid {name}={raw!r}; using default {default}")
            return default

    @classmethod
    def _extract_context_length_details(cls, message: str) -> Optional[Tuple[int, int, int, int]]:
        match = cls._CONTEXT_LENGTH_DETAILS_RE.search(message)
        if not match:
            return None
        return tuple(int(group) for group in match.groups())

    @classmethod
    def _compute_reduced_completion_cap(
        cls,
        *,
        error: Exception,
        current_cap: Optional[int],
    ) -> Optional[int]:
        details = cls._extract_context_length_details(str(error))
        if details is None:
            return None

        context_limit, _requested_total, prompt_tokens, requested_completion = details
        reserve_tokens = max(64, min(512, context_limit // 64))
        safe_cap = context_limit - prompt_tokens - reserve_tokens
        if current_cap is not None:
            safe_cap = min(safe_cap, current_cap - 1)
        else:
            safe_cap = min(safe_cap, requested_completion - 1)
        if safe_cap < 16:
            return None
        return safe_cap

    @staticmethod
    def _is_openrouter_base_url(base_url: str) -> bool:
        hostname = (urlparse(base_url).hostname or "").lower()
        return hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai")

    @staticmethod
    def _is_openrouter_kimi_model(model: str) -> bool:
        normalized = (model or "").strip().lower()
        return normalized.startswith("moonshotai/kimi-k2.5")

    def _apply_reasoning_controls(self, params: Dict[str, object], *, base_url: str, model: str) -> None:
        extra_body = dict(params.get("extra_body") or {})
        if self._enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}
        if self._separate_reasoning is not None:
            extra_body["separate_reasoning"] = self._separate_reasoning
        reasoning_payload: Dict[str, object] | None = None
        if self._enable_thinking is False:
            if self._is_openrouter_base_url(base_url) and self._is_openrouter_kimi_model(model):
                reasoning_payload = {"enabled": False}
            elif self._is_openrouter_base_url(base_url):
                reasoning_payload = {"effort": "none", "exclude": True}
            else:
                reasoning_payload = {"enabled": False}
        elif self._reasoning_effort:
            reasoning_payload = {"effort": self._reasoning_effort}
        if reasoning_payload is not None:
            extra_body["reasoning"] = reasoning_payload
        if extra_body:
            params["extra_body"] = extra_body

    @staticmethod
    def _extract_visible_content(message: Any) -> str:
        content = getattr(message, "content", "") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"reasoning", "reasoning_content", "thinking"}:
                    continue
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text:
                    parts.append(str(text))
            return "".join(parts)
        return str(content)

    def _sanitize_usage(self, usage: Optional[dict]) -> Optional[dict]:
        if usage is None or not self._strip_reasoning_output:
            return usage
        sanitized = dict(usage)
        for key in ("reasoning_tokens", "reasoning", "reasoning_content"):
            sanitized.pop(key, None)
        for nested_key in ("completion_tokens_details", "prompt_tokens_details"):
            nested = sanitized.get(nested_key)
            if isinstance(nested, dict):
                nested_copy = dict(nested)
                for key in list(nested_copy.keys()):
                    if "reasoning" in str(key).lower():
                        nested_copy.pop(key, None)
                sanitized[nested_key] = nested_copy
        return sanitized

    async def _wait_for_global_rate_limit_window(self, base_url: str) -> None:
        while True:
            until = self._GLOBAL_RATE_LIMIT_UNTIL.get(base_url, 0.0)
            now = time.time()
            if until <= now:
                return
            await asyncio.sleep(min(5.0, max(0.1, until - now)))

    @staticmethod
    def _parse_retry_after(value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            seconds = float(text)
            return max(0.0, seconds)
        except ValueError:
            pass
        try:
            dt = email.utils.parsedate_to_datetime(text)
            return max(0.0, dt.timestamp() - time.time())
        except Exception:
            return None

    def _compute_rate_limit_delay(self, error: Exception, attempt: int) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        for key in (
            "retry-after",
            "x-ratelimit-reset",
            "ratelimit-reset",
            "x-ratelimit-reset-requests",
        ):
            delay = self._parse_retry_after(headers.get(key))
            if delay is not None and delay > 0:
                return min(self.RATE_LIMIT_MAX_DELAY, delay + random.uniform(0, 1))
        fallback = min(self.RATE_LIMIT_MAX_DELAY, self.BASE_DELAY * (2 ** min(attempt, 8)))
        return fallback + random.uniform(0, 1)

    async def _wait_for_rate_limit_reset(
        self,
        *,
        error: Exception,
        attempt: int,
        request_id: str | None,
        model: str,
        base_url: str,
    ) -> None:
        delay = self._compute_rate_limit_delay(error, attempt)
        self._GLOBAL_RATE_LIMIT_UNTIL[base_url] = max(
            self._GLOBAL_RATE_LIMIT_UNTIL.get(base_url, 0.0),
            time.time() + delay,
        )
        self._set_last_failure_metadata(
            {
                "failure_stage": "chat_with_tools",
                "failure_type": "rate_limit",
                "model": model,
                "base_url": base_url,
                "request_id": request_id,
                "retry_delay_s": delay,
                "attempt": attempt + 1,
            }
        )
        log("LLM", f"Rate limited; waiting {delay:.1f}s before retry")
        await asyncio.sleep(delay)

    @staticmethod
    def _server_root_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized[:-3]
        return normalized

    @staticmethod
    def _should_bypass_proxy(base_url: str) -> bool:
        try:
            hostname = (urlparse(base_url).hostname or "").strip()
            if not hostname:
                return False
            if hostname in {"api.aicodemirror.com"}:
                return True
            if hostname in {"localhost", "127.0.0.1"}:
                return True
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return False

    def _build_httpx_client(
        self,
        *,
        base_url: str,
        timeout: httpx.Timeout,
    ) -> httpx.AsyncClient:
        # Local SGLang endpoints should bypass the host proxy, otherwise
        # requests to the colocated router/servers can get sent to localhost:10812.
        return httpx.AsyncClient(
            timeout=timeout,
            trust_env=not self._should_bypass_proxy(base_url),
        )

    async def _abort_request(
        self,
        lease: Optional[_ServerLease],
        request_id: Optional[str],
        reason: str,
    ) -> None:
        if lease is None or not request_id:
            return

        abort_url = f"{self._server_root_url(lease.base_url)}/abort_request"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {lease.api_key}",
            "X-API-Key": lease.api_key,
        }
        try:
            async with self._build_httpx_client(
                base_url=lease.base_url,
                timeout=httpx.Timeout(10.0, connect=5.0),
            ) as client:
                response = await client.post(
                    abort_url,
                    json={
                        "rid": request_id,
                        "abort_all": False,
                        "abort_message": reason[:200],
                    },
                    headers=headers,
                )
                if response.status_code == 200:
                    log("LLM", f"Abort requested for {request_id} on {lease.server_id}")
                else:
                    log(
                        "LLM",
                        f"Abort request for {request_id} on {lease.server_id} returned {response.status_code}",
                    )
        except Exception as abort_error:
            log("LLM", f"Abort request for {request_id} on {lease.server_id} failed: {abort_error}")

    async def chat(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        timeout_s: int = None,
    ) -> Tuple[str, Optional[dict]]:
        actual_timeout = timeout_s if timeout_s is not None else self._default_timeout

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        last_error = None
        attempt = 0
        rate_limit_attempt = 0
        max_completion_tokens_override = None
        while True:
            lease = None
            request_id = None
            request_start = time.time()
            try:
                lease = await self._acquire_lease()
                request_id = self._generate_request_id("chat")
                content, usage = await asyncio.wait_for(
                    self._make_request(
                        base_url=lease.base_url,
                        api_key=lease.api_key,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        seed=seed,
                        timeout_s=actual_timeout,
                        request_id=request_id,
                        max_completion_tokens_override=max_completion_tokens_override,
                    ),
                    timeout=actual_timeout,
                )
                await self._release_lease(lease, success=True, latency_s=time.time() - request_start)
                return content, usage

            except asyncio.TimeoutError:
                await self._abort_request(
                    lease,
                    request_id,
                    f"timeout after {actual_timeout}s (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                last_error = TimeoutError(f"LLM request timed out after {actual_timeout}s")
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        f"Strict-serial LLM request timed out after {actual_timeout}s",
                        last_error,
                        attempt + 1,
                    )
                log("LLM", f"Total timeout ({actual_timeout}s) exceeded, attempt {attempt + 1}/{self._max_retries}")
                await self._backoff(attempt)
                continue

            except asyncio.CancelledError:
                await self._abort_request(
                    lease,
                    request_id,
                    f"cancelled (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                raise

            except openai.RateLimitError as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"rate limit (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                last_error = e
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        "Strict-serial LLM request hit rate limit",
                        e,
                        attempt + 1,
                    )
                log("LLM", f"Rate limit hit, attempt {attempt + 1}/{self._max_retries}")
                await self._backoff(attempt)

            except openai.BadRequestError as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"bad request (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                error_msg = str(e).lower()
                has_context_details = self._extract_context_length_details(str(e)) is not None
                if (
                    "is longer than the model" in error_msg
                    or "context_length_exceeded" in error_msg
                    or "maximum context length" in error_msg
                    or has_context_details
                ):
                    reduced_cap = self._compute_reduced_completion_cap(
                        error=e,
                        current_cap=(
                            max_completion_tokens_override
                            if max_completion_tokens_override is not None
                            else self._max_completion_tokens
                        ),
                    )
                    if reduced_cap is not None:
                        current_cap = (
                            max_completion_tokens_override
                            if max_completion_tokens_override is not None
                            else self._max_completion_tokens
                        )
                        if current_cap is not None and reduced_cap >= current_cap:
                            raise LLMFatalError(
                                f"Token limit exceeded without room to reduce completion cap: {e}",
                                original_error=e,
                                attempts=attempt + 1,
                            )
                        max_completion_tokens_override = reduced_cap
                        log("LLM", f"Retrying with reduced max_completion_tokens={reduced_cap} after context error")
                        continue
                    log("LLM", f"Token limit exceeded - fatal error: {e}", force=True)
                    raise LLMFatalError(
                        f"Token limit exceeded: {e}",
                        original_error=e,
                        attempts=attempt + 1,
                    )
                raise

            except openai.APIStatusError as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"api status {e.status_code} (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                if e.status_code in self.RETRY_STATUS_CODES:
                    last_error = e
                    if self._strict_serial:
                        self._raise_strict_serial_error(
                            f"Strict-serial LLM API error {e.status_code}",
                            e,
                            attempt + 1,
                        )
                    log("LLM", f"API error {e.status_code}, attempt {attempt + 1}/{self._max_retries}")
                    await self._backoff(attempt)
                else:
                    raise

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"connection error (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                last_error = e
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        f"Strict-serial LLM connection error: {e}",
                        e,
                        attempt + 1,
                    )
                log("LLM", f"Connection error, attempt {attempt + 1}/{self._max_retries}: {e}")
                await self._backoff(attempt)

            except Exception as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"client error: {type(e).__name__} (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                error_msg = str(e).lower()
                if (
                    "is longer than the model" in error_msg
                    or "context_length_exceeded" in error_msg
                    or "maximum context length" in error_msg
                    or self._extract_context_length_details(str(e)) is not None
                ):
                    log("LLM", f"Token limit exceeded - fatal error: {e}", force=True)
                    raise LLMFatalError(
                        f"Token limit exceeded: {e}",
                        original_error=e,
                        attempts=attempt + 1,
                    )
                last_error = e
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        f"Strict-serial LLM error: {e}",
                        e,
                        attempt + 1,
                    )
                log("LLM", f"Error, attempt {attempt + 1}/{self._max_retries}: {e}")
                await self._backoff(attempt)

        raise last_error or Exception("LLM request failed after all retries")

    async def chat_with_tools(
        self,
        system: str,
        user: str,
        model: str,
        tools: Optional[List[dict]] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        timeout_s: int = None,
    ) -> LLMResponse:
        actual_timeout = timeout_s if timeout_s is not None else self._default_timeout

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        self._set_last_failure_metadata(None)

        last_error = None
        attempt = 0
        rate_limit_attempt = 0
        max_completion_tokens_override = None
        while True:
            lease = None
            request_id = None
            request_start = time.time()
            try:
                lease = await self._acquire_lease()
                await self._wait_for_global_rate_limit_window(lease.base_url)
                request_id = self._generate_request_id("tools")
                response = await asyncio.wait_for(
                    self._make_request_with_tools(
                        base_url=lease.base_url,
                        api_key=lease.api_key,
                        messages=messages,
                        model=model,
                        tools=tools,
                        temperature=temperature,
                        top_p=None,
                        seed=seed,
                        timeout_s=actual_timeout,
                        request_id=request_id,
                        max_completion_tokens_override=max_completion_tokens_override,
                    ),
                    timeout=actual_timeout,
                )
                await self._release_lease(lease, success=True, latency_s=time.time() - request_start)
                return response

            except asyncio.TimeoutError:
                await self._abort_request(
                    lease,
                    request_id,
                    f"timeout after {actual_timeout}s (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                last_error = TimeoutError(f"LLM request timed out after {actual_timeout}s")
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        f"Strict-serial LLM request timed out after {actual_timeout}s",
                        last_error,
                        attempt + 1,
                    )
                log("LLM", f"Total timeout ({actual_timeout}s) exceeded, attempt {attempt + 1}/{self._max_retries}")
                attempt += 1
                if attempt >= self._max_retries:
                    raise last_error or Exception("LLM request failed after all retries")
                await self._backoff(attempt)
                continue

            except asyncio.CancelledError:
                await self._abort_request(
                    lease,
                    request_id,
                    f"cancelled (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                raise

            except openai.RateLimitError as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"rate limit (attempt {rate_limit_attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                last_error = e
                await self._wait_for_rate_limit_reset(
                    error=e,
                    attempt=rate_limit_attempt,
                    request_id=request_id,
                    model=model,
                    base_url=lease.base_url if lease is not None else self._base_url,
                )
                rate_limit_attempt += 1
                continue

            except openai.BadRequestError as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"bad request (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                error_msg = str(e).lower()
                has_context_details = self._extract_context_length_details(str(e)) is not None
                if (
                    "is longer than the model" in error_msg
                    or "context_length_exceeded" in error_msg
                    or "maximum context length" in error_msg
                    or has_context_details
                ):
                    reduced_cap = self._compute_reduced_completion_cap(
                        error=e,
                        current_cap=(
                            max_completion_tokens_override
                            if max_completion_tokens_override is not None
                            else self._max_completion_tokens
                        ),
                    )
                    if reduced_cap is not None:
                        current_cap = (
                            max_completion_tokens_override
                            if max_completion_tokens_override is not None
                            else self._max_completion_tokens
                        )
                        if current_cap is not None and reduced_cap >= current_cap:
                            raise LLMFatalError(
                                f"Token limit exceeded without room to reduce completion cap: {e}",
                                original_error=e,
                                attempts=attempt + 1,
                            )
                        max_completion_tokens_override = reduced_cap
                        log("LLM", f"Retrying with reduced max_completion_tokens={reduced_cap} after context error")
                        continue
                    raise LLMFatalError(f"Token limit exceeded: {e}", original_error=e, attempts=attempt + 1)
                raise

            except openai.APIStatusError as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"api status {e.status_code} (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                if e.status_code in self.RETRY_STATUS_CODES:
                    last_error = e
                    if e.status_code == 429:
                        await self._wait_for_rate_limit_reset(
                            error=e,
                            attempt=rate_limit_attempt,
                            request_id=request_id,
                            model=model,
                            base_url=lease.base_url if lease is not None else self._base_url,
                        )
                        rate_limit_attempt += 1
                        continue
                    if self._strict_serial:
                        self._raise_strict_serial_error(
                            f"Strict-serial LLM API error {e.status_code}",
                            e,
                            attempt + 1,
                        )
                    log("LLM", f"API error {e.status_code}, attempt {attempt + 1}/{self._max_retries}")
                    attempt += 1
                    if attempt >= self._max_retries:
                        raise last_error or Exception("LLM request failed after all retries")
                    await self._backoff(attempt)
                else:
                    raise

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"connection error (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                last_error = e
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        f"Strict-serial LLM connection error: {e}",
                        e,
                        attempt + 1,
                    )
                log("LLM", f"Connection error, attempt {attempt + 1}/{self._max_retries}: {e}")
                attempt += 1
                if attempt >= self._max_retries:
                    raise last_error or Exception("LLM request failed after all retries")
                await self._backoff(attempt)

            except Exception as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"client error: {type(e).__name__} (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                error_msg = str(e).lower()
                if (
                    "is longer than the model" in error_msg
                    or "context_length_exceeded" in error_msg
                    or "maximum context length" in error_msg
                    or self._extract_context_length_details(str(e)) is not None
                ):
                    raise LLMFatalError(f"Token limit exceeded: {e}", original_error=e, attempts=attempt + 1)
                last_error = e
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        f"Strict-serial LLM error: {e}",
                        e,
                        attempt + 1,
                    )
                log("LLM", f"Error, attempt {attempt + 1}/{self._max_retries}: {e}")
                attempt += 1
                if attempt >= self._max_retries:
                    raise last_error or Exception("LLM request failed after all retries")
                await self._backoff(attempt)

    async def chat_with_tools_recovery(
        self,
        model: str,
        messages: Optional[list] = None,
        system: str = "",
        user: str = "",
        assistant_prefix: str = "",
        tools: Optional[List[dict]] = None,
        seed: Optional[int] = None,
        timeout_s: int = None,
        max_new_tokens: Optional[int] = None,
    ) -> LLMResponse:
        actual_timeout = timeout_s if timeout_s is not None else self._default_timeout

        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            remediation = (
                "The previous assistant output had invalid tool-call formatting. "
                "Emit exactly one valid tool call now. No explanation, no prose, no markdown, no XML."
            )
            if assistant_prefix:
                messages.append({"role": "assistant", "content": assistant_prefix})
                remediation = (
                    "Continue from the assistant's last output and emit exactly one valid tool call now. "
                    "No explanation, no prose, no markdown, no XML."
                )
            messages.append(
                {
                    "role": "user",
                    "content": remediation,
                }
            )
        self._set_last_failure_metadata(None)

        last_error = None
        for attempt in range(self._max_retries):
            lease = None
            request_id = None
            request_start = time.time()
            try:
                lease = await self._acquire_lease()
                await self._wait_for_global_rate_limit_window(lease.base_url)
                request_id = self._generate_request_id("format-recovery")
                response = await asyncio.wait_for(
                    self._make_request_with_tools(
                        base_url=lease.base_url,
                        api_key=lease.api_key,
                        messages=messages,
                        model=model,
                        tools=tools,
                        temperature=self._format_recovery_temperature,
                        top_p=self._format_recovery_top_p,
                        seed=seed,
                        timeout_s=actual_timeout,
                        request_id=request_id,
                        max_completion_tokens_override=max_new_tokens,
                    ),
                    timeout=actual_timeout,
                )
                await self._release_lease(lease, success=True, latency_s=time.time() - request_start)
                return response
            except Exception as e:
                await self._abort_request(
                    lease,
                    request_id,
                    f"format recovery error: {type(e).__name__} (attempt {attempt + 1})",
                )
                if lease is not None:
                    await self._release_lease(lease, success=False, latency_s=time.time() - request_start)
                last_error = e
                if isinstance(e, openai.RateLimitError) or (
                    isinstance(e, openai.APIStatusError) and getattr(e, "status_code", None) == 429
                ):
                    await self._wait_for_rate_limit_reset(
                        error=e,
                        attempt=rate_limit_attempt,
                        request_id=request_id,
                        model=model,
                        base_url=lease.base_url if lease is not None else self._base_url,
                    )
                    rate_limit_attempt += 1
                    continue
                if self._strict_serial:
                    self._raise_strict_serial_error(
                        f"Strict-serial format recovery error: {e}",
                        e,
                        attempt + 1,
                    )
                attempt += 1
                if attempt >= self._max_retries:
                    raise last_error or Exception("LLM format recovery failed after all retries")
                await self._backoff(attempt)
        raise last_error or Exception("LLM format recovery failed after all retries")

    async def _acquire_lease(self) -> _ServerLease:
        if self._router is not None:
            return await self._router.acquire(route_key=self._route_key)
        return _ServerLease(
            server_id="single",
            base_url=self._base_url,
            api_key=self._api_key,
            started_at=time.time(),
        )

    async def _release_lease(self, lease: _ServerLease, success: bool, latency_s: float):
        if self._router is not None:
            await self._router.release(lease, success=success, latency_s=latency_s)

    async def _make_request_with_tools(
        self,
        base_url: str,
        api_key: str,
        messages: list,
        model: str,
        tools: Optional[List[dict]],
        temperature: float,
        top_p: Optional[float],
        seed: Optional[int],
        timeout_s: int,
        request_id: str,
        max_completion_tokens_override: Optional[int] = None,
    ) -> LLMResponse:
        timeout_config = httpx.Timeout(
            connect=30.0,
            read=timeout_s,
            write=30.0,
            pool=30.0,
        )

        http_client = self._build_httpx_client(base_url=base_url, timeout=timeout_config)
        client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_config,
            max_retries=0,
            http_client=http_client,
        )

        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if top_p is not None:
                params["top_p"] = top_p
            if tools:
                params["tools"] = tools
            if seed is not None:
                params["seed"] = seed
            effective_max_completion_tokens = (
                max_completion_tokens_override
                if max_completion_tokens_override is not None
                else self._max_completion_tokens
            )
            if effective_max_completion_tokens is not None:
                # Pass both names for compatibility across OpenAI-compatible servers.
                params["max_tokens"] = effective_max_completion_tokens
                params["max_completion_tokens"] = effective_max_completion_tokens
            params["extra_body"] = {"request_id": request_id}
            self._apply_reasoning_controls(params, base_url=base_url, model=model)

            start_time = time.time()
            response = await client.chat.completions.create(**params)
            elapsed = time.time() - start_time

            if is_verbose():
                log("LLM", f"Tool call response in {elapsed:.1f}s ({request_id})")

            choice = response.choices[0] if response.choices else None
            if not choice:
                self._set_last_failure_metadata(
                    {
                        "failure_stage": "chat_with_tools",
                        "failure_type": "no_choices",
                        "model": model,
                        "base_url": base_url,
                        "request_id": request_id,
                        "attempt": 1,
                    }
                )
                raise ValueError("LLM returned no choices")

            content = self._extract_visible_content(choice.message)
            parsed_tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    parsed_tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            function={"name": tc.function.name, "arguments": tc.function.arguments},
                        )
                    )

            usage = self._sanitize_usage(response.usage.model_dump() if response.usage else None)

            if not content and not parsed_tool_calls:
                self._set_last_failure_metadata(
                    {
                        "failure_stage": "chat_with_tools",
                        "failure_type": "empty_response",
                        "model": model,
                        "base_url": base_url,
                        "request_id": getattr(response, "id", request_id),
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "usage": usage,
                        "choice_message_preview": {
                            "content": content[:200],
                            "tool_call_count": len(parsed_tool_calls),
                        },
                        "attempt": 1,
                    }
                )
                raise ValueError("LLM returned empty response (no content, no tool_calls)")

            return LLMResponse(
                content=content.strip(),
                tool_calls=parsed_tool_calls,
                usage=usage,
                request_id=getattr(response, "id", request_id),
            )
        finally:
            await client.close()

    async def _make_request(
        self,
        base_url: str,
        api_key: str,
        messages: list,
        model: str,
        temperature: float,
        seed: Optional[int],
        timeout_s: int,
        request_id: str,
        max_completion_tokens_override: Optional[int] = None,
    ) -> Tuple[str, Optional[dict]]:
        timeout_config = httpx.Timeout(
            connect=30.0,
            read=timeout_s,
            write=30.0,
            pool=30.0,
        )

        http_client = self._build_httpx_client(base_url=base_url, timeout=timeout_config)
        client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_config,
            max_retries=0,
            http_client=http_client,
        )

        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if seed is not None:
                params["seed"] = seed
            effective_max_completion_tokens = (
                max_completion_tokens_override
                if max_completion_tokens_override is not None
                else self._max_completion_tokens
            )
            if effective_max_completion_tokens is not None:
                # Pass both names for compatibility across OpenAI-compatible servers.
                params["max_tokens"] = effective_max_completion_tokens
                params["max_completion_tokens"] = effective_max_completion_tokens
            params["extra_body"] = {"request_id": request_id}
            self._apply_reasoning_controls(params, base_url=base_url, model=model)

            start_time = time.time()
            stream = await client.chat.completions.create(**params)

            content_parts = []
            usage = None
            chunk_count = 0
            last_progress = 0.0

            async for chunk in stream:
                chunk_count += 1
                if chunk_count > self.MAX_CHUNKS:
                    log("LLM", f"Chunk limit exceeded ({self.MAX_CHUNKS}), aborting {request_id}")
                    await self._abort_request(
                        _ServerLease(
                            server_id="stream-local",
                            base_url=base_url,
                            api_key=api_key,
                            started_at=start_time,
                        ),
                        request_id,
                        f"chunk limit exceeded ({self.MAX_CHUNKS})",
                    )
                    break

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    delta_content = self._extract_visible_content(delta)
                    if delta_content:
                        content_parts.append(delta_content)
                if chunk.usage:
                    usage = self._sanitize_usage(chunk.usage.model_dump())

                elapsed = time.time() - start_time
                if is_verbose() and elapsed - last_progress >= 1.0:
                    last_progress = elapsed
                    progress("LLM", elapsed, timeout_s, f"chunks:{chunk_count}")

            if is_verbose() and last_progress > 0:
                progress_done("LLM", f"Done in {time.time() - start_time:.1f}s, {chunk_count} chunks")

            content = "".join(content_parts)
            if not content:
                raise ValueError(f"LLM returned empty response after {chunk_count} chunks")

            return content.strip(), usage
        finally:
            await client.close()

    async def _backoff(self, attempt: int):
        delay = min(
            self.BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
            self.MAX_DELAY,
        )
        await asyncio.sleep(delay)
