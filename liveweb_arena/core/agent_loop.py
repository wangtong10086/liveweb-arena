"""Agent loop for browser-based task execution"""

import asyncio
import os
from collections import Counter
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse

from .browser import BrowserSession
from .cache import CacheFatalError
from .models import BrowserAction, BrowserObservation, CompositeTask, TrajectoryStep
from .agent_protocol import AgentProtocol
from .runtime_profiles import is_fast_collect_profile, normalize_runtime_profile
from ..utils.llm_client import LLMClient, LLMFatalError
from ..utils.logger import log


class BrowserFatalError(Exception):
    """
    Raised when browser navigation fails after maximum retries.

    This indicates persistent network or site accessibility issues
    that should terminate evaluation immediately.
    """

    def __init__(self, message: str, url: str = None, attempts: int = 0):
        super().__init__(message)
        self.url = url
        self.attempts = attempts


# Type for navigation callback: async (url: str) -> None
NavigationCallback = Callable[[str], Any]
# Type for step complete callback: async (step: TrajectoryStep) -> None
StepCompleteCallback = Callable[["TrajectoryStep"], Any]
# Type for observation callback: async (observation: BrowserObservation) -> None
ObservationCallback = Callable[[Any], Any]

# URL patterns that indicate browser/network errors (not AI's fault)
# Note: about:blank is NOT an error - it's the initial page where AI starts
ERROR_URL_PATTERNS = [
    "chrome-error://",
    "about:neterror",
]

LOCAL_RECOVERY_PREVIEW_LIMIT = 20


def is_error_page(url: str) -> bool:
    """Check if URL indicates a browser error (not AI's fault).

    Note: about:blank is NOT considered an error page - it's the starting point.
    Only actual error pages like chrome-error:// are treated specially.
    """
    if not url:
        return False
    return any(pattern in url.lower() for pattern in ERROR_URL_PATTERNS)


class AgentLoop:
    """
    Main agent loop that drives browser interaction via LLM.

    Uses AgentProtocol (function calling) for structured tool_calls interaction.
    The loop maintains trajectory state internally for partial recovery on timeout.
    """

    def __init__(
        self,
        session: BrowserSession,
        llm_client: LLMClient,
        protocol: AgentProtocol,
        max_steps: int = 30,
        runtime_profile: str | None = None,
        behavior_mode: str = "eval",
        enable_format_recovery: Optional[bool] = None,
        on_navigation: Optional[NavigationCallback] = None,
        on_step_complete: Optional[StepCompleteCallback] = None,
        on_observation: Optional[ObservationCallback] = None,
    ):
        self._session = session
        self._llm_client = llm_client
        self._protocol = protocol
        self._max_steps = max_steps
        self._runtime_profile = normalize_runtime_profile(runtime_profile or behavior_mode)
        self._behavior_mode = "collect" if is_fast_collect_profile(self._runtime_profile) else "eval"
        self._collect_mode = is_fast_collect_profile(self._runtime_profile)
        self._on_navigation = on_navigation
        self._on_step_complete = on_step_complete
        self._on_observation = on_observation
        self._failfast_action_failures = int(os.getenv("LIVEWEB_FAILFAST_ACTION_FAILURES", "5"))
        self._failfast_error_pages = int(os.getenv("LIVEWEB_FAILFAST_ERROR_PAGES", "10"))
        self._failfast_blank_observations = int(os.getenv("LIVEWEB_FAILFAST_BLANK_OBSERVATIONS", "4"))
        self._enable_format_recovery = (
            enable_format_recovery
            if enable_format_recovery is not None
            else (os.getenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1") == "1")
        )
        self._format_recovery_max_retries = int(os.getenv("LIVEWEB_FORMAT_RECOVERY_MAX_RETRIES", "4"))
        self._format_recovery_max_new_tokens = int(os.getenv("LIVEWEB_FORMAT_RECOVERY_MAX_NEW_TOKENS", "96"))
        self._format_recovery_empty_max_retries = int(
            os.getenv("LIVEWEB_FORMAT_RECOVERY_EMPTY_MAX_RETRIES", "2")
        )
        self._format_recovery_context_length = int(
            os.getenv("LIVEWEB_FORMAT_RECOVERY_CONTEXT_LENGTH", os.getenv("LIVEWEB_LLM_CONTEXT_LENGTH", "32768"))
        )
        self._format_recovery_token_margin = int(os.getenv("LIVEWEB_FORMAT_RECOVERY_TOKEN_MARGIN", "256"))
        self._enable_disallowed_domain_recovery = os.getenv("LIVEWEB_ENABLE_DISALLOWED_DOMAIN_RECOVERY", "1") == "1"
        self._disallowed_domain_recovery_max_retries = int(
            os.getenv("LIVEWEB_DISALLOWED_DOMAIN_RECOVERY_MAX_RETRIES", "4")
        )
        self._disallowed_domain_recovery_max_new_tokens = int(
            os.getenv("LIVEWEB_DISALLOWED_DOMAIN_RECOVERY_MAX_NEW_TOKENS", "96")
        )
        self._failfast_disallowed_domain_consecutive = int(
            os.getenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_CONSECUTIVE", "2")
        )
        self._failfast_disallowed_domain_total = int(
            os.getenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_TOTAL", "3")
        )
        self._empty_stop_recovery_max_retries = int(
            os.getenv("LIVEWEB_EMPTY_STOP_RECOVERY_MAX_RETRIES", "1")
        )
        self._invalid_ui_target_recovery_max_retries = int(
            os.getenv("LIVEWEB_INVALID_UI_TARGET_RECOVERY_MAX_RETRIES", "1")
        )
        self._taostats_list_action_recovery_max_retries = int(
            os.getenv("LIVEWEB_TAOSTATS_LIST_ACTION_RECOVERY_MAX_RETRIES", "1")
        )
        self._local_recovery_max_new_tokens = int(
            os.getenv("LIVEWEB_LOCAL_RECOVERY_MAX_NEW_TOKENS", "96")
        )
        self._natural_language_parse_recovery_max_retries = int(
            os.getenv("LIVEWEB_NATURAL_LANGUAGE_PARSE_RECOVERY_MAX_RETRIES", "1")
        )
        self._invalid_generated_url_recovery_max_retries = int(
            os.getenv("LIVEWEB_INVALID_GENERATED_URL_RECOVERY_MAX_RETRIES", "1")
        )

        # Internal state for partial recovery
        self._trajectory: List[TrajectoryStep] = []
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._final_answer = None
        self._format_recovery_attempts = 0
        self._format_recovery_successes = 0
        self._format_recovery_exhausted = 0
        self._format_failure_class_counts: Counter[str] = Counter()
        self._last_parse_failure_metadata: dict[str, Any] = {}
        self._last_llm_failure_metadata: dict[str, Any] = {}
        self._invalid_stop_payload = False
        self._last_stop_failure_class: str | None = None
        self._local_recovery_attempt_counts: Counter[str] = Counter()
        self._local_recovery_success_counts: Counter[str] = Counter()
        self._local_recovery_events_preview: list[dict[str, Any]] = []
        self._gemini_loop_same_action_threshold = int(
            os.getenv("LIVEWEB_GEMINI_LOOP_SAME_ACTION_THRESHOLD", "6")
        )
        self._gemini_loop_goto_oscillation_threshold = int(
            os.getenv("LIVEWEB_GEMINI_LOOP_GOTO_OSCILLATION_THRESHOLD", "4")
        )
        self._action_loop_detected = False
        self._last_action_loop_detail: dict[str, Any] = {}
        self._invalid_generated_url = False
        self._last_invalid_generated_url_detail: dict[str, Any] = {}
        self._gemini_loop_same_type_threshold = int(
            os.getenv("LIVEWEB_GEMINI_LOOP_SAME_TYPE_THRESHOLD", "4")
        )
        self._gemini_loop_search_bounce_threshold = int(
            os.getenv("LIVEWEB_GEMINI_LOOP_SEARCH_BOUNCE_THRESHOLD", "4")
        )

    def get_trajectory(self) -> List[TrajectoryStep]:
        """Get current trajectory (for partial recovery on timeout)"""
        return self._trajectory.copy()

    def get_usage(self) -> Optional[dict]:
        """Get current usage stats"""
        return self._total_usage.copy() if any(self._total_usage.values()) else None

    def get_runtime_profile(self) -> str:
        return self._runtime_profile

    def get_final_answer(self) -> Any:
        """Get final answer if available"""
        return self._final_answer

    def get_format_recovery_stats(self) -> dict:
        attempts = self._format_recovery_attempts
        successes = self._format_recovery_successes
        exhausted = self._format_recovery_exhausted
        return {
            "format_recovery_attempts": attempts,
            "format_recovery_successes": successes,
            "format_recovery_exhausted": exhausted,
            "format_recovery_success_rate": (successes / attempts) if attempts else 0.0,
            "format_failure_class_counts": dict(self._format_failure_class_counts),
            "format_failure_recoverable_rate": (
                sum(count for name, count in self._format_failure_class_counts.items() if name.startswith("recoverable_")) / total
                if (total := sum(self._format_failure_class_counts.values()))
                else 0.0
            ),
            "format_failure_terminal_rate": (
                sum(count for name, count in self._format_failure_class_counts.items() if name.startswith("terminal_")) / total
                if (total := sum(self._format_failure_class_counts.values()))
                else 0.0
            ),
        }

    def get_last_parse_failure_metadata(self) -> dict[str, Any]:
        return dict(self._last_parse_failure_metadata)

    def get_last_llm_failure_metadata(self) -> dict[str, Any]:
        return dict(self._last_llm_failure_metadata)

    def get_local_recovery_stats(self) -> dict[str, Any]:
        attempts = dict(self._local_recovery_attempt_counts)
        successes = dict(self._local_recovery_success_counts)
        total_attempts = sum(attempts.values())
        total_successes = sum(successes.values())
        if "empty_stop_recovery" in attempts and "invalid_stop_recovery" not in attempts:
            attempts["invalid_stop_recovery"] = attempts["empty_stop_recovery"]
        if "empty_stop_recovery" in successes and "invalid_stop_recovery" not in successes:
            successes["invalid_stop_recovery"] = successes["empty_stop_recovery"]
        stats: dict[str, Any] = {
            "local_recovery_attempts_total": total_attempts,
            "local_recovery_successes_total": total_successes,
            "local_recovery_success_rate": (total_successes / total_attempts) if total_attempts else 0.0,
            "local_recovery_events_preview": list(self._local_recovery_events_preview),
        }
        for key in sorted(set(attempts) | set(successes)):
            stats[f"{key}_attempts"] = attempts.get(key, 0)
            stats[f"{key}_successes"] = successes.get(key, 0)
        return stats

    def is_action_loop_detected(self) -> bool:
        return self._action_loop_detected

    def get_last_action_loop_detail(self) -> dict[str, Any]:
        return dict(self._last_action_loop_detail)

    def is_invalid_stop_payload(self) -> bool:
        return self._invalid_stop_payload

    def get_last_stop_failure_class(self) -> str | None:
        return self._last_stop_failure_class

    def is_invalid_generated_url(self) -> bool:
        return self._invalid_generated_url

    def get_last_invalid_generated_url_detail(self) -> dict[str, Any]:
        return dict(self._last_invalid_generated_url_detail)

    async def _call_llm(
        self, system_prompt: str, user_prompt: str, model: str,
        temperature: float, seed: Optional[int],
    ) -> Tuple[str, Optional[BrowserAction], Any]:
        """
        Call LLM with function calling protocol.

        Returns:
            Tuple of (raw_response, parsed_action_or_None, usage)
        """
        tools = self._protocol.get_tools()
        response = await self._llm_client.chat_with_tools(
            system=system_prompt,
            user=user_prompt,
            model=model,
            tools=tools,
            temperature=temperature,
            seed=seed,
        )
        raw_response = response.content
        if response.has_tool_calls:
            tc = response.tool_calls[0]
            raw_response = raw_response or f"[tool_call: {tc.function['name']}({tc.function['arguments']})]"
        action = self._protocol.parse_response(raw_response, response.tool_calls)
        return raw_response, action, response

    def _build_recovery_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        failure_class: str,
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for step in self._trajectory:
            messages.extend(self._protocol.serialize_step(step))
        messages.append({"role": "user", "content": user_prompt})
        remediation = (
            "The previous assistant output had invalid tool-call formatting. "
            "Emit exactly one valid tool call now. No explanation, no prose, no markdown, no XML."
        )
        if failure_class == "recoverable_empty":
            remediation = (
                "Your previous assistant message was empty. Emit exactly one valid tool call now. "
                "No explanation, no prose, no markdown, no XML."
            )
        messages.append(
            {
                "role": "user",
                "content": remediation,
            }
        )
        return messages

    def _build_disallowed_domain_recovery_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        blocked_url: str,
        allowed_domains: list[str],
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for step in self._trajectory:
            messages.extend(self._protocol.serialize_step(step))
        messages.append({"role": "user", "content": user_prompt})
        allowed_display = ", ".join(allowed_domains) if allowed_domains else "(unknown)"
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous action navigated to a disallowed domain.\n"
                    f"Blocked URL: {blocked_url}\n"
                    f"Allowed domains: {allowed_display}\n\n"
                    "Emit exactly one new valid NON-STOP tool call that stays on an allowed domain. "
                    "Do not reuse the blocked domain. No explanation, no prose, no markdown, no XML."
                ),
            }
        )
        return messages

    def _build_local_recovery_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        remediation: str,
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for step in self._trajectory:
            messages.extend(self._protocol.serialize_step(step))
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "user", "content": remediation})
        return messages

    @staticmethod
    def _get_navigation_metadata(session: Any) -> dict[str, Any] | None:
        getter = getattr(session, "get_last_navigation_metadata", None)
        if callable(getter):
            return getter()
        return None

    @staticmethod
    def _clear_navigation_metadata(session: Any) -> None:
        clearer = getattr(session, "clear_last_navigation_metadata", None)
        if callable(clearer):
            clearer()

    @staticmethod
    def _is_disallowed_domain_metadata(metadata: dict[str, Any] | None) -> bool:
        if not metadata:
            return False
        return str(metadata.get("classification_hint") or "") == "model_disallowed_domain"

    @staticmethod
    def _url_matches_allowed_domain(url: str, allowed_domain: str) -> bool:
        try:
            hostname = (urlparse(url).hostname or "").lower()
        except Exception:
            return False
        allowed_domain = (allowed_domain or "").lower()
        return bool(hostname) and (hostname == allowed_domain or hostname.endswith("." + allowed_domain))

    def _action_targets_allowed_domain(self, action: BrowserAction, allowed_domains: list[str]) -> bool:
        if action.action_type != "goto":
            return True
        url = str(action.params.get("url", "") or "")
        if not url or not allowed_domains:
            return True
        return any(self._url_matches_allowed_domain(url, domain) for domain in allowed_domains)

    @staticmethod
    def _action_is_same_page_recovery_safe(action: BrowserAction, *, allow_stop: bool = True) -> bool:
        if action.action_type in {"click", "click_role", "scroll", "view_more", "wait"}:
            return True
        if allow_stop and action.action_type == "stop":
            return True
        return False

    @staticmethod
    def _is_blank_answer_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return True
            if normalized in {
                "n/a",
                "na",
                "unknown",
                "none",
                "null",
                "need to check",
                "unable to determine",
                "not enough information",
            }:
                return True
        return False

    def _classify_stop_payload(
        self,
        *,
        task: CompositeTask,
        action: BrowserAction,
    ) -> tuple[str | None, list[str]]:
        if action.action_type != "stop":
            return None, []
        answers = action.params.get("answers")
        if answers is None and isinstance(action.params.get("final"), dict):
            answers = action.params.get("final", {}).get("answers")
        if not isinstance(answers, dict) or not answers:
            return "empty_answers", []
        required_tags = [str(getattr(st, "answer_tag", "") or "").strip() for st in task.subtasks]
        required_tags = [tag for tag in required_tags if tag]
        missing_tags = [tag for tag in required_tags if self._is_blank_answer_value(answers.get(tag))]
        if missing_tags:
            return "missing_answers", missing_tags
        return None, []

    def _record_local_recovery_event(
        self,
        *,
        kind: str,
        status: str,
        url: str | None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "kind": kind,
            "status": status,
            "url": url,
        }
        if detail:
            event.update(detail)
        if len(self._local_recovery_events_preview) < LOCAL_RECOVERY_PREVIEW_LIMIT:
            self._local_recovery_events_preview.append(event)

    @staticmethod
    def _is_gemini_model(model: str) -> bool:
        return "gemini" in str(model or "").lower()

    @staticmethod
    def _is_kimi_model(model: str) -> bool:
        model_lower = str(model or "").lower()
        return "kimi" in model_lower or "moonshotai/" in model_lower

    @staticmethod
    def _normalize_action_signature_value(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _action_signature(self, action: BrowserAction) -> tuple[str, str]:
        action_type = str(action.action_type or "")
        params = action.params or {}
        if action_type == "click":
            target = self._normalize_action_signature_value(params.get("selector"))
        elif action_type == "click_role":
            target = "|".join(
                [
                    self._normalize_action_signature_value(params.get("role")),
                    self._normalize_action_signature_value(params.get("name")),
                ]
            )
        elif action_type in {"type", "type_role"}:
            target = "|".join(
                [
                    self._normalize_action_signature_value(params.get("selector")),
                    self._normalize_action_signature_value(params.get("role")),
                    self._normalize_action_signature_value(params.get("name")),
                    self._normalize_action_signature_value(params.get("text")),
                ]
            )
        elif action_type == "goto":
            target = self._normalize_action_signature_value(params.get("url"))
        else:
            parts = [f"{k}={self._normalize_action_signature_value(v)}" for k, v in sorted(params.items())]
            target = "|".join(parts)
        return action_type, target

    @staticmethod
    def _normalize_action_url(url: Any) -> str:
        return str(url or "").strip()

    def _classify_invalid_generated_url(self, action: BrowserAction) -> dict[str, Any] | None:
        if action.action_type != "goto":
            return None
        raw_url = self._normalize_action_url(action.params.get("url"))
        if not raw_url:
            return {
                "kind": "missing_url",
                "url": raw_url,
                "reason": "missing URL in goto action",
            }
        try:
            parsed = urlparse(raw_url)
        except Exception as exc:
            return {
                "kind": "parse_error",
                "url": raw_url,
                "reason": f"URL parse failed: {exc}",
            }
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").strip()
        if scheme not in {"http", "https"}:
            return {
                "kind": "invalid_scheme",
                "url": raw_url,
                "reason": f"unsupported URL scheme: {scheme or '(missing)'}",
            }
        if not hostname:
            return {
                "kind": "missing_host",
                "url": raw_url,
                "reason": "URL is missing a valid host",
            }
        if any(char.isspace() for char in raw_url):
            return {
                "kind": "whitespace",
                "url": raw_url,
                "reason": "URL contains whitespace",
            }
        return None

    def _detect_action_loop(
        self,
        *,
        model: str,
        current_obs: BrowserObservation,
        action: BrowserAction,
    ) -> dict[str, Any] | None:
        if not self._collect_mode or not self._is_gemini_model(model):
            return None

        current_url = str(getattr(current_obs, "url", "") or "")
        signature = self._action_signature(action)
        tail_matches = 0
        for step in reversed(self._trajectory):
            if step.action is None:
                break
            if str(getattr(step.observation, "url", "") or "") != current_url:
                break
            if self._action_signature(step.action) != signature:
                break
            tail_matches += 1
        if tail_matches >= self._gemini_loop_same_action_threshold - 1:
            return {
                "kind": "same_action_repeat",
                "url": current_url,
                "action_type": signature[0],
                "action_target": signature[1],
                "repeat_count": tail_matches + 1,
            }

        if action.action_type in {"type", "type_role"}:
            same_type_matches = 0
            for step in reversed(self._trajectory):
                if step.action is None:
                    break
                if step.action.action_type not in {"type", "type_role"}:
                    break
                if str(getattr(step.observation, "url", "") or "") != current_url:
                    break
                if self._action_signature(step.action) != signature:
                    break
                same_type_matches += 1
            if same_type_matches >= self._gemini_loop_same_type_threshold - 1:
                return {
                    "kind": "same_type_repeat",
                    "url": current_url,
                    "action_type": signature[0],
                    "action_target": signature[1],
                    "repeat_count": same_type_matches + 1,
                }

        if action.action_type == "goto":
            goto_targets = [
                str(step.action.params.get("url", "") or "")
                for step in self._trajectory
                if step.action is not None and step.action.action_type == "goto"
            ]
            goto_targets.append(str(action.params.get("url", "") or ""))
            window = self._gemini_loop_goto_oscillation_threshold * 2
            recent_targets = goto_targets[-window:]
            unique_targets = []
            for target in recent_targets:
                if target and target not in unique_targets:
                    unique_targets.append(target)
            if len(recent_targets) == window and len(unique_targets) == 2:
                expected = [unique_targets[i % 2] for i in range(window)]
                if recent_targets == expected or recent_targets == list(reversed(expected)):
                    return {
                        "kind": "goto_oscillation",
                        "url": current_url,
                        "action_type": "goto",
                        "action_target": str(action.params.get("url", "") or ""),
                        "oscillation_urls": unique_targets,
                        "repeat_count": window,
                    }
            search_finance_markers = (
                "google.com/search",
                "google.com/finance",
                "finance.yahoo.com",
                "coinmarketcap.com",
                "marketwatch.com",
                "cnbc.com",
                "wsj.com",
            )
            search_finance_targets = [target for target in goto_targets if any(marker in target.lower() for marker in search_finance_markers)]
            if len(search_finance_targets) >= self._gemini_loop_search_bounce_threshold:
                recent_search_targets = search_finance_targets[-self._gemini_loop_search_bounce_threshold :]
                unique_recent = []
                for target in recent_search_targets:
                    if target and target not in unique_recent:
                        unique_recent.append(target)
                if len(unique_recent) <= 3 and len(recent_search_targets) > len(unique_recent):
                    return {
                        "kind": "search_finance_bounce",
                        "url": current_url,
                        "action_type": "goto",
                        "action_target": str(action.params.get("url", "") or ""),
                        "oscillation_urls": unique_recent,
                        "repeat_count": len(recent_search_targets),
                    }
        return None

    @staticmethod
    def _looks_like_missing_ui_target(raw_exception_message: str) -> bool:
        lower = (raw_exception_message or "").lower()
        return (
            "no element found with role" in lower
            or "strict mode violation" in lower
            or "waiting for locator" in lower and "resolved to 0 elements" in lower
        )

    def _classify_collect_recovery_kind(
        self,
        *,
        current_obs: BrowserObservation,
        navigation_metadata: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any]]:
        if not self._collect_mode or not navigation_metadata:
            return None, {}
        raw_exception_message = str(navigation_metadata.get("raw_exception_message") or "")
        raw_exception_type = str(navigation_metadata.get("raw_exception_type") or "")
        evidence = dict(navigation_metadata.get("evidence") or {})
        current_url = str(navigation_metadata.get("url") or current_obs.url or "")
        try:
            parsed = urlparse(current_url)
            hostname = (parsed.hostname or "").lower()
            path = parsed.path or "/"
        except Exception:
            hostname = ""
            path = "/"

        if evidence.get("ui_target_missing") or self._looks_like_missing_ui_target(raw_exception_message):
            return (
                "invalid_ui_target_recovery",
                {
                    "page_kind": evidence.get("page_kind"),
                    "interaction_kind": evidence.get("interaction_kind"),
                    "target_locator": evidence.get("target_locator") or evidence.get("selector"),
                    "raw_exception_type": raw_exception_type,
                    "raw_exception_message": raw_exception_message,
                },
            )

        lower = f"{raw_exception_type} {raw_exception_message}".lower()
        if (
            hostname.endswith("taostats.io")
            and path.startswith("/subnets")
            and str(navigation_metadata.get("navigation_stage") or "").startswith("action_")
            and ("timeout" in lower or "too many consecutive action failures" in lower)
        ):
            return (
                "taostats_list_action_recovery",
                {
                    "page_kind": "taostats_list",
                    "interaction_kind": evidence.get("interaction_kind") or "unknown",
                    "target_locator": evidence.get("target_locator") or evidence.get("selector"),
                    "raw_exception_type": raw_exception_type,
                    "raw_exception_message": raw_exception_message,
                },
            )
        return None, {}

    @staticmethod
    def _estimate_message_tokens(messages: list[dict]) -> int:
        total = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += max(1, len(content) // 4) + 8
            else:
                total += 16
        return total

    @staticmethod
    def _truncate_recovery_content(
        content: str,
        *,
        max_chars: int,
        keep_head_ratio: float = 0.7,
    ) -> str:
        if len(content) <= max_chars:
            return content
        if max_chars <= 64:
            return content[:max_chars]
        head_chars = max(32, int(max_chars * keep_head_ratio))
        tail_chars = max(16, max_chars - head_chars - 32)
        if head_chars + tail_chars + 32 > max_chars:
            tail_chars = max(16, max_chars - head_chars - 32)
        marker = "\n...[truncated for format recovery]...\n"
        return f"{content[:head_chars]}{marker}{content[-tail_chars:]}"

    def _slim_recovery_message(
        self,
        message: dict[str, Any],
        *,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        slimmed = dict(message)
        content = slimmed.get("content")
        if not isinstance(content, str):
            return slimmed

        role = str(slimmed.get("role") or "")
        max_chars = 1200
        if index == 0 and role == "system":
            max_chars = 4000
        elif index >= total - 1:
            max_chars = 800
        elif index >= total - 2:
            max_chars = 2500
        elif role == "assistant":
            max_chars = 900
        slimmed["content"] = self._truncate_recovery_content(content, max_chars=max_chars)
        return slimmed

    def _shrink_recovery_messages_to_budget(
        self,
        *,
        messages: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, Any]] | None:
        current = [dict(message) for message in messages]
        if not current:
            return current

        protected_indices: set[int] = set()
        if current and current[0].get("role") == "system":
            protected_indices.add(0)
        protected_indices.add(len(current) - 1)
        if len(current) >= 2:
            protected_indices.add(len(current) - 2)

        while self._estimate_message_tokens(current) > budget:
            largest_index = None
            largest_size = 0
            for idx, message in enumerate(current):
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                if len(content) > largest_size:
                    largest_index = idx
                    largest_size = len(content)
            if largest_index is None:
                break

            message = dict(current[largest_index])
            content = message.get("content", "")
            if not isinstance(content, str):
                break

            if len(content) > 256:
                reduced = max(192, int(len(content) * 0.65))
                message["content"] = self._truncate_recovery_content(content, max_chars=reduced)
                current[largest_index] = message
                continue

            removable_indices = [
                idx
                for idx in range(len(current))
                if idx not in protected_indices
            ]
            if not removable_indices:
                return None
            del current[removable_indices[0]]

            protected_indices = {
                idx if idx < removable_indices[0] else idx - 1
                for idx in protected_indices
                if idx != removable_indices[0]
            }

        if self._estimate_message_tokens(current) > budget:
            return None
        return current

    def _trim_recovery_messages(
        self,
        *,
        messages: list[dict],
        max_new_tokens: int,
    ) -> tuple[list[dict] | None, bool]:
        budget = self._format_recovery_context_length - max_new_tokens - self._format_recovery_token_margin
        if budget <= 0:
            return None, True
        overflowed = self._estimate_message_tokens(messages) > budget
        if not messages:
            return [], overflowed

        system_message = messages[0] if messages[0].get("role") == "system" else None
        tail_keep = 4 if len(messages) >= 5 else len(messages)
        trimmed_candidates: list[dict[str, Any]] = []
        if system_message is not None:
            trimmed_candidates.append(system_message)
        trimmed_candidates.extend(messages[-tail_keep:])

        deduped: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for message in trimmed_candidates:
            marker = id(message)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            deduped.append(message)

        slimmed = [
            self._slim_recovery_message(message, index=index, total=len(deduped))
            for index, message in enumerate(deduped)
        ]
        shrunk = self._shrink_recovery_messages_to_budget(messages=slimmed, budget=budget)
        if shrunk is None:
            return None, True
        return shrunk, overflowed

    @staticmethod
    def _classify_recovery_exception(exc: Exception) -> str | None:
        message = str(exc or "").lower()
        if "recoverable_context_overflow" in message:
            return "recoverable_context_overflow"
        if (
            "format recovery error" in message and "context length" in message
        ) or "strict-serial format recovery error" in message:
            return "format_recovery_overflow"
        if "requested token count exceeds the model's maximum context length" in message:
            return "llm_context_overflow"
        if "longer than the model's context length" in message:
            return "llm_context_overflow"
        return None

    async def _attempt_format_recovery(
        self,
        *,
        model: str,
        seed: Optional[int],
        raw_response: str,
        messages: list[dict],
        failure_class: str,
    ) -> Tuple[str, Optional[BrowserAction], Optional[dict], Optional[str]]:
        if not self._enable_format_recovery:
            return raw_response, None, None, None

        tools = self._protocol.get_tools()
        base_messages, overflowed = self._trim_recovery_messages(
            messages=list(messages),
            max_new_tokens=self._format_recovery_max_new_tokens,
        )
        if base_messages is None:
            log("Agent", "Skipping format recovery because trimmed recovery context still exceeds context budget", force=True)
            return raw_response, None, None, "recoverable_context_overflow"

        self._format_recovery_attempts += 1
        recovery_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        max_retries = self._format_recovery_max_retries
        if failure_class == "recoverable_empty":
            max_retries = min(max_retries, self._format_recovery_empty_max_retries)

        for retry_idx in range(max_retries):
            try:
                response = await self._llm_client.chat_with_tools_recovery(
                    messages=base_messages,
                    model=model,
                    tools=tools,
                    seed=seed,
                    max_new_tokens=self._format_recovery_max_new_tokens,
                )
            except Exception as exc:
                classified = self._classify_recovery_exception(exc)
                if classified is not None:
                    log(
                        "Agent",
                        f"Format recovery failed fast due to classified error={classified}: {exc}",
                        force=True,
                    )
                    return raw_response, None, recovery_usage, classified
                raise
            if response.usage:
                for key in recovery_usage:
                    recovery_usage[key] += int(response.usage.get(key, 0) or 0)
            recovered_raw = response.content
            if response.has_tool_calls:
                tc = response.tool_calls[0]
                recovered_raw = recovered_raw or f"[tool_call: {tc.function['name']}({tc.function['arguments']})]"
            action = self._protocol.parse_response(recovered_raw, response.tool_calls)
            if action is not None:
                self._format_recovery_successes += 1
                log(
                    "Agent",
                    f"Format recovery succeeded on retry {retry_idx + 1} with action={action.action_type}",
                    force=True,
                )
                return recovered_raw, action, recovery_usage, None

        self._format_recovery_exhausted += 1
        return raw_response, None, recovery_usage, None

    async def _attempt_disallowed_domain_recovery(
        self,
        *,
        model: str,
        seed: Optional[int],
        system_prompt: str,
        user_prompt: str,
        blocked_url: str,
        allowed_domains: list[str],
    ) -> Tuple[Optional[str], Optional[BrowserAction], Optional[dict]]:
        if not self._enable_disallowed_domain_recovery:
            return None, None, None

        tools = self._protocol.get_tools()
        base_messages, _overflowed = self._trim_recovery_messages(
            messages=self._build_disallowed_domain_recovery_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                blocked_url=blocked_url,
                allowed_domains=allowed_domains,
            ),
            max_new_tokens=self._disallowed_domain_recovery_max_new_tokens,
        )
        if base_messages is None:
            log("Agent", "Skipping disallowed-domain recovery because context budget is exceeded", force=True)
            return None, None, None

        recovery_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for retry_idx in range(self._disallowed_domain_recovery_max_retries):
            response = await self._llm_client.chat_with_tools_recovery(
                messages=base_messages,
                model=model,
                tools=tools,
                seed=seed,
                max_new_tokens=self._disallowed_domain_recovery_max_new_tokens,
            )
            if response.usage:
                for key in recovery_usage:
                    recovery_usage[key] += int(response.usage.get(key, 0) or 0)
            recovered_raw = response.content
            if response.has_tool_calls:
                tc = response.tool_calls[0]
                recovered_raw = recovered_raw or f"[tool_call: {tc.function['name']}({tc.function['arguments']})]"
            action = self._protocol.parse_response(recovered_raw, response.tool_calls)
            if action is None:
                continue
            if action.action_type == "stop":
                log("Agent", f"Disallowed-domain recovery retry {retry_idx + 1} returned stop; retrying", force=True)
                continue
            if not self._action_targets_allowed_domain(action, allowed_domains):
                log(
                    "Agent",
                    f"Disallowed-domain recovery retry {retry_idx + 1} still targeted blocked domain; retrying",
                    force=True,
                )
                continue
            log(
                "Agent",
                f"Disallowed-domain recovery succeeded on retry {retry_idx + 1} with action={action.action_type}",
                force=True,
            )
            return recovered_raw, action, recovery_usage
        return None, None, recovery_usage

    async def _attempt_local_recovery(
        self,
        *,
        kind: str,
        model: str,
        seed: Optional[int],
        system_prompt: str,
        user_prompt: str,
        remediation: str,
        validator,
        max_retries: int,
        url: str,
        detail: dict[str, Any] | None = None,
    ) -> Tuple[Optional[str], Optional[BrowserAction], Optional[dict]]:
        tools = self._protocol.get_tools()
        base_messages, _overflowed = self._trim_recovery_messages(
            messages=self._build_local_recovery_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                remediation=remediation,
            ),
            max_new_tokens=self._local_recovery_max_new_tokens,
        )
        if base_messages is None:
            self._record_local_recovery_event(
                kind=kind,
                status="skipped_context_overflow",
                url=url,
                detail=detail,
            )
            return None, None, None

        recovery_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for retry_idx in range(max_retries):
            self._local_recovery_attempt_counts[kind] += 1
            response = await self._llm_client.chat_with_tools_recovery(
                messages=base_messages,
                model=model,
                tools=tools,
                seed=seed,
                max_new_tokens=self._local_recovery_max_new_tokens,
            )
            if response.usage:
                for key in recovery_usage:
                    recovery_usage[key] += int(response.usage.get(key, 0) or 0)
            recovered_raw = response.content
            if response.has_tool_calls:
                tc = response.tool_calls[0]
                recovered_raw = recovered_raw or f"[tool_call: {tc.function['name']}({tc.function['arguments']})]"
            action = self._protocol.parse_response(recovered_raw, response.tool_calls)
            if action is None:
                continue
            if validator(action):
                self._local_recovery_success_counts[kind] += 1
                self._record_local_recovery_event(
                    kind=kind,
                    status="success",
                    url=url,
                    detail={"retry_index": retry_idx + 1, **(detail or {})},
                )
                return recovered_raw, action, recovery_usage
        self._record_local_recovery_event(
            kind=kind,
            status="exhausted",
            url=url,
            detail=detail,
        )
        return None, None, recovery_usage

    async def run(
        self,
        task: CompositeTask,
        model: str,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> Tuple[List[TrajectoryStep], Any, Optional[dict]]:
        """
        Run the agent loop until completion or max_steps.

        Args:
            task: Composite task to complete
            model: LLM model name
            temperature: LLM temperature
            seed: LLM seed for reproducibility

        Returns:
            Tuple of (trajectory, final_answer, usage)
            - trajectory: List of TrajectoryStep
            - final_answer: The final answer dict from stop action, or None
            - usage: Aggregated LLM usage dict
        """
        # Reset internal state
        self._trajectory = []
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._final_answer = None
        self._max_steps_reached = False
        self._parse_failed = False
        self._format_recovery_attempts = 0
        self._format_recovery_successes = 0
        self._format_recovery_exhausted = 0
        self._format_failure_class_counts = Counter()
        self._last_parse_failure_metadata = {}
        self._last_llm_failure_metadata = {}
        self._invalid_stop_payload = False
        self._last_stop_failure_class = None
        self._local_recovery_attempt_counts = Counter()
        self._local_recovery_success_counts = Counter()
        self._local_recovery_events_preview = []
        self._action_loop_detected = False
        self._last_action_loop_detail = {}
        self._invalid_generated_url = False
        self._last_invalid_generated_url_detail = {}

        system_prompt = self._protocol.build_system_prompt(task)
        log("Agent", f"Starting loop, max_steps={self._max_steps}, protocol=function_calling")

        obs = await self._session.goto("about:blank")
        consecutive_errors = 0
        consecutive_error_pages = 0
        consecutive_action_failures = 0
        consecutive_blank_observations = 0
        max_error_page_retries = self._failfast_error_pages
        consecutive_disallowed_domain_hits = 0
        total_disallowed_domain_hits = 0

        effective_step = 0  # Count all steps including error pages (AI sees them)
        iteration = 0  # Total iterations (safety limit)
        last_goto_url = None  # Track last navigation URL for error context

        while effective_step < self._max_steps:
            iteration += 1
            # Safety limit to prevent infinite loops
            if iteration > self._max_steps * 3:
                log("Agent", "Too many iterations, breaking loop", force=True)
                break

            # Check if we're on an error page - let AI see it and decide what to do
            if is_error_page(obs.url):
                consecutive_error_pages += 1
                log("Agent", f"Error page detected (visible to AI): {obs.url[:50]}")

                # Safety limit: if AI keeps landing on error pages, eventually stop
                if consecutive_error_pages >= max_error_page_retries:
                    log("Agent", f"Too many consecutive error pages ({consecutive_error_pages}), AI unable to navigate", force=True)
                    raise BrowserFatalError(
                        f"AI unable to navigate after {consecutive_error_pages} consecutive error pages",
                        url=last_goto_url,
                        attempts=consecutive_error_pages,
                    )
                # Error pages count as a step - AI will see it and can take corrective action
            else:
                # Reset error page counter on valid page
                consecutive_error_pages = 0

            obs_text = getattr(obs, "accessibility_tree", "") or ""
            if obs.url == "about:blank" or len(obs_text.strip()) < 50:
                consecutive_blank_observations += 1
            else:
                consecutive_blank_observations = 0

            if consecutive_blank_observations >= self._failfast_blank_observations:
                raise BrowserFatalError(
                    f"Too many blank observations ({consecutive_blank_observations})",
                    url=last_goto_url or obs.url,
                    attempts=consecutive_blank_observations,
                )

            effective_step += 1
            log("")  # Blank line between steps
            log("Agent", f"Step {effective_step}/{self._max_steps}, url={obs.url[:50]}")

            # Fire observation callback for real-time GT collection (before action)
            if self._on_observation:
                try:
                    await self._on_observation(obs)
                except CacheFatalError:
                    raise
                except Exception as e:
                    log("Agent", f"Observation callback error: {e}")
                    # Record GT collection failure so it's visible in results
                    from .gt_collector import get_current_gt_collector
                    gt = get_current_gt_collector()
                    if gt:
                        gt.record_observation_error(obs.url, str(e))

            # Pre-save observation so it's not lost if LLM call times out
            current_obs = obs
            step_num = effective_step - 1  # 0-indexed step number for trajectory
            user_prompt = self._protocol.build_step_prompt(
                current_obs, self._trajectory, effective_step, self._max_steps
            )

            try:
                raw_response, action, usage = await self._call_llm(
                    system_prompt, user_prompt, model, temperature, seed,
                )
                if usage and usage.usage:
                    for key in self._total_usage:
                        self._total_usage[key] += usage.usage.get(key, 0)
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                self._last_llm_failure_metadata = self._llm_client.get_last_failure_metadata()
                max_consecutive = 3
                log("Agent", f"LLM error ({consecutive_errors}/{max_consecutive}): {type(e).__name__}: {e}", force=True)

                if consecutive_errors >= max_consecutive:
                    raise LLMFatalError(
                        f"LLM errors exhausted after {consecutive_errors} consecutive failures: {type(e).__name__}: {e}",
                        original_error=e,
                        attempts=consecutive_errors,
                    )

                # Brief wait before retry
                await asyncio.sleep(1)
                continue

            # Parse failed - terminate immediately
            if action is None:
                failure_class = self._protocol.classify_format_failure(raw_response, usage.tool_calls if usage else None)
                parse_debug = self._protocol.debug_parse_metadata(raw_response, usage.tool_calls if usage else None)
                self._format_failure_class_counts[failure_class] += 1
                self._last_parse_failure_metadata = {
                    "format_failure_class": failure_class,
                    "raw_response_preview": (raw_response or "")[:400],
                    **parse_debug,
                }
                if failure_class.startswith("recoverable_"):
                    if self._enable_format_recovery:
                        log("Agent", f"Parse failed, attempting format recovery: {failure_class}", force=True)
                        raw_response, action, usage, failure_override = await self._attempt_format_recovery(
                            model=model,
                            seed=seed,
                            raw_response=raw_response,
                            messages=self._build_recovery_messages(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                failure_class=failure_class,
                            ),
                            failure_class=failure_class,
                        )
                        if failure_override is not None:
                            if self._format_failure_class_counts[failure_class] > 0:
                                self._format_failure_class_counts[failure_class] -= 1
                            self._format_failure_class_counts[failure_override] += 1
                        if usage:
                            for key in self._total_usage:
                                self._total_usage[key] += usage.get(key, 0)
                    else:
                        log("Agent", f"Parse failed with format recovery disabled: {failure_class}", force=True)
                elif (
                    self._collect_mode
                    and self._is_kimi_model(model)
                    and failure_class == "terminal_natural_language"
                    and str(parse_debug.get("protocol_parser_branch") or "") == "natural_language"
                ):
                    remediation = (
                        "Your previous assistant message used prose instead of a tool call.\n"
                        "Emit exactly one valid tool call or one valid stop call now. "
                        "No prose, no explanation, no markdown."
                    )
                    recovered_raw, recovered_action, recovered_usage = await self._attempt_local_recovery(
                        kind="natural_language_parse_recovery",
                        model=model,
                        seed=seed,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        remediation=remediation,
                        validator=lambda candidate: candidate is not None,
                        max_retries=self._natural_language_parse_recovery_max_retries,
                        url=current_obs.url,
                        detail={"failure_class": failure_class, "parser_branch": parse_debug.get("protocol_parser_branch")},
                    )
                    if recovered_usage:
                        for key in self._total_usage:
                            self._total_usage[key] += recovered_usage.get(key, 0)
                    if recovered_action is not None:
                        raw_response = recovered_raw or raw_response
                        action = recovered_action

                if action is None:
                    log("Agent", f"PARSE FAILED: {raw_response[:200]!r}", force=True)

                    step = TrajectoryStep(
                        step_num=step_num,
                        observation=current_obs,
                        action=None,
                        action_result="Parse failed - model output not valid JSON",
                        prompt=user_prompt,
                        raw_response=raw_response,
                    )
                    self._trajectory.append(step)
                    self._parse_failed = True
                    break

            if action.action_type == "stop":
                stop_failure_class = None
                stop_missing_tags: list[str] = []
                if self._collect_mode:
                    stop_failure_class, stop_missing_tags = self._classify_stop_payload(task=task, action=action)
                if stop_failure_class is not None:
                    self._last_stop_failure_class = stop_failure_class
                    remediation = (
                        "Your previous stop action had empty or incomplete answers.\n"
                        f"Required answer tags: {', '.join(stop_missing_tags) if stop_missing_tags else 'all required tags'}\n\n"
                        "Emit exactly one valid stop tool call now. "
                        "You must provide a short final string for each required answer key. "
                        "Do not explore further. Do not emit any non-stop action. "
                        "Do not include explanations."
                    )
                    recovered_raw, recovered_action, recovered_usage = await self._attempt_local_recovery(
                        kind="empty_stop_recovery",
                        model=model,
                        seed=seed,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        remediation=remediation,
                        validator=lambda candidate: (
                            candidate.action_type == "stop"
                            and self._classify_stop_payload(task=task, action=candidate)[0] is None
                        ),
                        max_retries=self._empty_stop_recovery_max_retries,
                        url=current_obs.url,
                        detail={"stop_failure_class": stop_failure_class, "missing_tags": stop_missing_tags},
                    )
                    if recovered_usage:
                        for key in self._total_usage:
                            self._total_usage[key] += recovered_usage.get(key, 0)
                    if recovered_action is None:
                        log("Agent", f"Invalid stop payload after recovery: {stop_failure_class}", force=True)
                        step = TrajectoryStep(
                            step_num=step_num,
                            observation=current_obs,
                            action=action,
                            action_result=f"Invalid stop payload: {stop_failure_class}",
                            prompt=user_prompt,
                            raw_response=raw_response,
                        )
                        self._trajectory.append(step)
                        self._invalid_stop_payload = True
                        break
                    raw_response = recovered_raw or raw_response
                    action = recovered_action
                final_params = action.params.get("final", {})
                self._final_answer = final_params if final_params else action.params
                log("Agent", f"Completed: {self._final_answer}")

                step = TrajectoryStep(
                    step_num=step_num,
                    observation=current_obs,
                    action=action,
                    action_result="Task completed",
                    prompt=user_prompt,
                    raw_response=raw_response,
                )
                self._trajectory.append(step)

                # Fire step complete callback for final step
                if self._on_step_complete:
                    try:
                        await self._on_step_complete(step)
                    except Exception as e:
                        log("Agent", f"Step complete callback error: {e}")
                break
            else:
                log("Agent", f"Action: {action.action_type}")
                old_url = obs.url if obs else None
                recovered_before_execute_kind = None
                invalid_generated_url_detail = self._classify_invalid_generated_url(action)
                if self._collect_mode and invalid_generated_url_detail is not None:
                    remediation = (
                        "Your previous goto action used an invalid URL.\n"
                        f"Invalid URL: {invalid_generated_url_detail.get('url') or '(missing)'}\n"
                        f"Reason: {invalid_generated_url_detail.get('reason') or 'invalid generated URL'}\n\n"
                        "Emit exactly one new valid NON-STOP tool call now. "
                        "If you use goto, it must be a fully qualified http(s) URL with a valid host. "
                        "No explanation."
                    )
                    recovered_raw, recovered_action, recovered_usage = await self._attempt_local_recovery(
                        kind="invalid_generated_url_recovery",
                        model=model,
                        seed=seed,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        remediation=remediation,
                        validator=lambda candidate: (
                            candidate.action_type != "stop"
                            and self._classify_invalid_generated_url(candidate) is None
                        ),
                        max_retries=self._invalid_generated_url_recovery_max_retries,
                        url=current_obs.url,
                        detail=invalid_generated_url_detail,
                    )
                    if recovered_usage:
                        for key in self._total_usage:
                            self._total_usage[key] += recovered_usage.get(key, 0)
                    if recovered_action is not None:
                        raw_response = recovered_raw or raw_response
                        action = recovered_action
                        recovered_before_execute_kind = "invalid_generated_url_recovery"
                    else:
                        self._invalid_generated_url = True
                        self._last_invalid_generated_url_detail = dict(invalid_generated_url_detail)
                        step = TrajectoryStep(
                            step_num=step_num,
                            observation=current_obs,
                            action=action,
                            action_result=f"Invalid generated URL: {invalid_generated_url_detail.get('reason')}",
                            prompt=user_prompt,
                            raw_response=raw_response,
                        )
                        self._trajectory.append(step)
                        break
                loop_detail = self._detect_action_loop(model=model, current_obs=current_obs, action=action)
                if loop_detail is not None:
                    self._action_loop_detected = True
                    self._last_action_loop_detail = loop_detail
                    detail_text = (
                        f"{loop_detail.get('kind')} type={loop_detail.get('action_type')} "
                        f"target={loop_detail.get('action_target')} repeat={loop_detail.get('repeat_count')}"
                    )
                    log("Agent", f"Detected repetitive action loop: {detail_text}", force=True)
                    step = TrajectoryStep(
                        step_num=step_num,
                        observation=current_obs,
                        action=action,
                        action_result=f"Aborted: repetitive_action_loop ({detail_text})",
                        prompt=user_prompt,
                        raw_response=raw_response,
                    )
                    self._trajectory.append(step)
                    break

                # Execute action - browser handles navigation errors internally
                # and returns error pages as valid observations
                try:
                    recovered_after_disallowed = False
                    recovered_after_local_error = None
                    disallowed_recovery_attempts = 0
                    while True:
                        try:
                            obs = await self._session.execute_action(action)
                            action_result = "Success"
                            if recovered_before_execute_kind:
                                action_result = f"Success (recovered after {recovered_before_execute_kind})"
                            consecutive_action_failures = 0
                        except BrowserFatalError:
                            raise
                        except Exception as e:
                            navigation_metadata = self._get_navigation_metadata(self._session)
                            recovery_kind, recovery_detail = self._classify_collect_recovery_kind(
                                current_obs=current_obs,
                                navigation_metadata=navigation_metadata,
                            )
                            remediation = None
                            validator = None
                            max_retries = 0
                            if recovery_kind == "invalid_ui_target_recovery":
                                remediation = (
                                    "Your previous UI target was missing on the current page.\n"
                                    f"Current URL: {current_obs.url}\n"
                                    f"Failed target: {recovery_detail.get('target_locator') or '(unknown)'}\n\n"
                                    "Emit exactly one safer SAME-PAGE action now. "
                                    "Do not use goto. Do not repeat the missing target. "
                                    "Allowed actions: click, click_role, scroll, view_more, wait, stop. "
                                    "No explanation."
                                )
                                validator = lambda candidate: self._action_is_same_page_recovery_safe(candidate, allow_stop=True)
                                max_retries = self._invalid_ui_target_recovery_max_retries
                            elif recovery_kind == "taostats_list_action_recovery":
                                remediation = (
                                    "You are already on the taostats subnet list page.\n"
                                    f"Current URL: {current_obs.url}\n"
                                    f"Failed interaction kind: {recovery_detail.get('interaction_kind') or 'unknown'}\n"
                                    f"Failed target: {recovery_detail.get('target_locator') or '(unknown)'}\n\n"
                                    "Emit exactly one more conservative SAME-PAGE action now. "
                                    "Do not use goto. Do not repeat the same failed selector or control. "
                                    "Allowed actions: click, click_role, scroll, view_more, wait, stop. "
                                    "If the page already contains enough data, you may stop."
                                )
                                validator = lambda candidate: self._action_is_same_page_recovery_safe(candidate, allow_stop=True)
                                max_retries = self._taostats_list_action_recovery_max_retries

                            if remediation and validator and max_retries > 0:
                                recovered_raw, recovered_action, recovered_usage = await self._attempt_local_recovery(
                                    kind=recovery_kind,
                                    model=model,
                                    seed=seed,
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                    remediation=remediation,
                                    validator=validator,
                                    max_retries=max_retries,
                                    url=current_obs.url,
                                    detail=recovery_detail,
                                )
                                if recovered_usage:
                                    for key in self._total_usage:
                                        self._total_usage[key] += recovered_usage.get(key, 0)
                                self._clear_navigation_metadata(self._session)
                                if recovered_action is not None:
                                    raw_response = recovered_raw or raw_response
                                    action = recovered_action
                                    recovered_after_local_error = recovery_kind
                                    continue

                            action_result = f"Failed: {e}"
                            consecutive_action_failures += 1
                            if consecutive_action_failures >= self._failfast_action_failures:
                                raise BrowserFatalError(
                                    f"Too many consecutive action failures ({consecutive_action_failures})",
                                    url=last_goto_url or current_obs.url,
                                    attempts=consecutive_action_failures,
                                )
                            break

                        if action.action_type == "goto":
                            last_goto_url = action.params.get("url", "")

                        navigation_metadata = self._get_navigation_metadata(self._session)
                        if self._is_disallowed_domain_metadata(navigation_metadata):
                            blocked_url = (
                                str((navigation_metadata.get("evidence") or {}).get("blocked_url") or "")
                                or str(navigation_metadata.get("url") or "")
                                or str(action.params.get("url", "") or "")
                            )
                            allowed_domains = list((navigation_metadata.get("evidence") or {}).get("allowed_domains") or [])
                            consecutive_disallowed_domain_hits += 1
                            total_disallowed_domain_hits += 1
                            disallowed_recovery_attempts += 1
                            log(
                                "Agent",
                                (
                                    "Disallowed domain hit; attempting local recovery "
                                    f"(consecutive={consecutive_disallowed_domain_hits}, total={total_disallowed_domain_hits}, "
                                    f"local_retry={disallowed_recovery_attempts}/{self._disallowed_domain_recovery_max_retries})"
                                ),
                                force=True,
                            )
                            recovered_raw, recovered_action, recovered_usage = await self._attempt_disallowed_domain_recovery(
                                model=model,
                                seed=seed,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                blocked_url=blocked_url,
                                allowed_domains=allowed_domains,
                            )
                            if recovered_usage:
                                for key in self._total_usage:
                                    self._total_usage[key] += recovered_usage.get(key, 0)
                            self._clear_navigation_metadata(self._session)
                            if recovered_action is not None:
                                raw_response = recovered_raw or raw_response
                                action = recovered_action
                                recovered_after_disallowed = True
                                continue
                            if (
                                consecutive_disallowed_domain_hits >= self._failfast_disallowed_domain_consecutive
                                or total_disallowed_domain_hits >= self._failfast_disallowed_domain_total
                            ):
                                raise BrowserFatalError(
                                    (
                                        "Too many disallowed-domain navigations "
                                        f"(consecutive={consecutive_disallowed_domain_hits}, total={total_disallowed_domain_hits})"
                                    ),
                                    url=blocked_url or last_goto_url or current_obs.url,
                                    attempts=consecutive_disallowed_domain_hits,
                                )
                            raise BrowserFatalError(
                                "disallowed-domain recovery exhausted before reaching an allowed page",
                                url=blocked_url or last_goto_url or current_obs.url,
                                attempts=disallowed_recovery_attempts,
                            )

                        consecutive_disallowed_domain_hits = 0
                        if recovered_after_disallowed:
                            action_result = "Success (recovered after disallowed-domain retry)"
                        elif recovered_after_local_error:
                            action_result = f"Success (recovered after {recovered_after_local_error})"

                        # Fire navigation callback if URL changed
                        if self._on_navigation and obs.url != old_url:
                            try:
                                await self._on_navigation(obs.url)
                            except CacheFatalError:
                                raise  # Cache failure = browser can't load = terminate immediately
                            except Exception as e:
                                log("Agent", f"Navigation callback error: {e}")
                        break
                except BrowserFatalError:
                    raise

            step = TrajectoryStep(
                step_num=step_num,
                observation=current_obs,
                action=action,
                action_result=action_result,
                prompt=user_prompt,
                raw_response=raw_response,
            )
            self._trajectory.append(step)

            # Fire step complete callback (after action executed)
            if self._on_step_complete:
                try:
                    await self._on_step_complete(step)
                except Exception as e:
                    log("Agent", f"Step complete callback error: {e}")

        # Check if max steps reached without completion
        if self._final_answer is None and effective_step >= self._max_steps:
            self._max_steps_reached = True
            log("Agent", f"Max steps ({self._max_steps}) reached without completion", force=True)

        log("Agent", f"Finished with {len(self._trajectory)} steps")
        return self._trajectory, self._final_answer, self.get_usage()

    def is_max_steps_reached(self) -> bool:
        """Check if max steps was reached without completion"""
        return self._max_steps_reached

    def is_parse_failed(self) -> bool:
        """Check if evaluation terminated due to parse failure"""
        return self._parse_failed
