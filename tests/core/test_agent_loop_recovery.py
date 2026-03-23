import pytest

from liveweb_arena.core.agent_loop import AgentLoop, BrowserFatalError
from liveweb_arena.core.agent_protocol import FunctionCallingProtocol
from liveweb_arena.core.models import BrowserObservation, CompositeTask
from liveweb_arena.utils.llm_client import LLMResponse, ToolCall


class _FakeSession:
    async def goto(self, url: str):
        return BrowserObservation(url=url, title="Blank", accessibility_tree="root")

    async def execute_action(self, action):
        return BrowserObservation(url="https://example.com", title="Done", accessibility_tree="root")

    def get_last_navigation_metadata(self):
        return None


class _FakeSessionWithBlockedSequence:
    def __init__(self, observations, metadatas):
        self._observations = list(observations)
        self._metadatas = list(metadatas)
        self._last_navigation_metadata = None

    async def goto(self, url: str):
        self._last_navigation_metadata = None
        return BrowserObservation(url=url, title="Blank", accessibility_tree="root")

    async def execute_action(self, action):
        self._last_navigation_metadata = self._metadatas.pop(0)
        return self._observations.pop(0)

    def get_last_navigation_metadata(self):
        return self._last_navigation_metadata

    def clear_last_navigation_metadata(self):
        self._last_navigation_metadata = None


class _FakeSessionWithActionSequence:
    def __init__(self, outcomes, metadatas):
        self._outcomes = list(outcomes)
        self._metadatas = list(metadatas)
        self._last_navigation_metadata = None

    async def goto(self, url: str):
        self._last_navigation_metadata = None
        return BrowserObservation(
            url=url,
            title="Blank",
            accessibility_tree="root content with enough accessible text to avoid blank observation failfast",
        )

    async def execute_action(self, action):
        self._last_navigation_metadata = self._metadatas.pop(0)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def get_last_navigation_metadata(self):
        return self._last_navigation_metadata

    def clear_last_navigation_metadata(self):
        self._last_navigation_metadata = None


class _FakeSubTask:
    plugin_name = "coingecko"
    answer_tag = "a1"
    expected_steps = 1


class _FakeSubTask2:
    plugin_name = "taostats"
    answer_tag = "a2"
    expected_steps = 1


class _FakeLLMClient:
    def __init__(self, initial_response, recovery_responses=None):
        self.initial_response = initial_response
        self.recovery_responses = list(recovery_responses or [])
        self.recovery_calls = 0
        self.recovery_messages = []

    async def chat_with_tools(self, **kwargs):
        return self.initial_response

    async def chat_with_tools_recovery(self, **kwargs):
        self.recovery_calls += 1
        self.recovery_messages.append(kwargs.get("messages"))
        return self.recovery_responses.pop(0)


class _QueuedLLMClient:
    def __init__(self, responses, recovery_responses=None):
        self._responses = list(responses)
        self._recovery_responses = list(recovery_responses or [])
        self.recovery_calls = 0
        self.recovery_messages = []

    async def chat_with_tools(self, **kwargs):
        return self._responses.pop(0)

    async def chat_with_tools_recovery(self, **kwargs):
        self.recovery_calls += 1
        self.recovery_messages.append(kwargs.get("messages"))
        if not self._recovery_responses:
            raise AssertionError("Recovery should not be called in this test")
        return self._recovery_responses.pop(0)

    def get_last_failure_metadata(self):
        return {}


def _task():
    return CompositeTask(
        subtasks=[_FakeSubTask()],
        combined_intent="Find the answer",
        plugin_hints={},
        seed=1,
    )


def _task_two_answers():
    return CompositeTask(
        subtasks=[_FakeSubTask(), _FakeSubTask2()],
        combined_intent="Find the answers",
        plugin_hints={},
        seed=1,
    )


@pytest.mark.anyio
async def test_agent_loop_recovers_from_truncated_tool_json(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_RETRIES", "2")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_NEW_TOKENS", "64")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\",\"arguments\":{\"url\":\"https://example.com\"}"),
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert loop.is_parse_failed() is False
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 1
    assert stats["format_recovery_successes"] == 1
    assert llm_client.recovery_calls == 1
    assert trajectory[-1].action.action_type == "stop"
    messages = llm_client.recovery_messages[0]
    assistant_messages = [item for item in messages if item["role"] == "assistant"]
    assert assistant_messages == []
    assert messages[-1]["role"] == "user"
    assert "invalid tool-call formatting" in messages[-1]["content"].lower()


@pytest.mark.anyio
async def test_agent_loop_does_not_recover_terminal_natural_language(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="I will inspect the page and then answer in plain English."),
        recovery_responses=[],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 0
    assert llm_client.recovery_calls == 0
    assert trajectory[-1].action is None


@pytest.mark.anyio
async def test_agent_loop_recovers_terminal_natural_language_for_kimi_in_collect_mode(monkeypatch):
    monkeypatch.setenv("LIVEWEB_NATURAL_LANGUAGE_PARSE_RECOVERY_MAX_RETRIES", "1")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="I found the page and will answer next."),
        recovery_responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"},
                    )
                ]
            ),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(
        task=_task(), model="moonshotai/kimi-k2.5", temperature=0.0, seed=1
    )

    assert final_answer == {"answers": {"a1": "42"}}
    assert loop.is_parse_failed() is False
    stats = loop.get_local_recovery_stats()
    assert stats["natural_language_parse_recovery_attempts"] == 1
    assert stats["natural_language_parse_recovery_successes"] == 1
    assert llm_client.recovery_calls == 1
    assert len(trajectory) == 1
    assert trajectory[0].action.action_type == "stop"


@pytest.mark.anyio
async def test_agent_loop_marks_parse_failed_after_recovery_exhausted(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_RETRIES", "2")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        recovery_responses=[
            LLMResponse(content="<tool_call>{\"name\":\"goto\""),
            LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 1
    assert stats["format_recovery_successes"] == 0


def test_trim_recovery_messages_truncates_large_recent_observation(monkeypatch):
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_CONTEXT_LENGTH", "1024")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TOKEN_MARGIN", "128")

    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=_FakeLLMClient(initial_response=LLMResponse(content="")),
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )
    giant_observation = "Observation:\n" + ("A" * 12000) + "\nTail marker"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "{\"name\":\"goto\"}"},
        {"role": "user", "content": giant_observation},
        {"role": "user", "content": "Emit exactly one valid tool call now."},
    ]

    trimmed, overflowed = loop._trim_recovery_messages(messages=messages, max_new_tokens=64)

    assert overflowed is True
    assert trimmed is not None
    assert loop._estimate_message_tokens(trimmed) <= (1024 - 64 - 128)
    assert trimmed[-2]["role"] == "user"
    assert "Tail marker" in trimmed[-2]["content"]
    assert len(trimmed[-2]["content"]) < len(giant_observation)
    assert "[truncated for format recovery]" in trimmed[-2]["content"]


@pytest.mark.anyio
async def test_attempt_format_recovery_skips_when_budget_is_non_positive(monkeypatch):
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_CONTEXT_LENGTH", "128")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TOKEN_MARGIN", "128")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        recovery_responses=[],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    raw_response, action, usage, failure_override = await loop._attempt_format_recovery(
        model="qwen",
        seed=1,
        raw_response="<tool_call>{\"name\":\"goto\"",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Find the answer"},
            {"role": "user", "content": "Emit one tool call"},
        ],
        failure_class="recoverable_partial_json",
    )

    assert action is None
    assert usage is None
    assert failure_override == "recoverable_context_overflow"
    stats = loop.get_local_recovery_stats()
    assert llm_client.recovery_calls == 0
    assert stats.get("format_recovery_exhausted", 0) == 0


@pytest.mark.anyio
async def test_agent_loop_limits_empty_recovery_retries(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_RETRIES", "4")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_EMPTY_MAX_RETRIES", "2")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content=""),
        recovery_responses=[
            LLMResponse(content=""),
            LLMResponse(content=""),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    stats = loop.get_format_recovery_stats()
    assert stats["format_recovery_attempts"] == 1
    assert stats["format_recovery_exhausted"] == 1
    assert llm_client.recovery_calls == 2


@pytest.mark.anyio
async def test_agent_loop_skips_recovery_when_context_budget_exceeded(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_CONTEXT_LENGTH", "64")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_MAX_NEW_TOKENS", "32")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TOKEN_MARGIN", "16")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        recovery_responses=[],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
    )
    long_obs = BrowserObservation(url="https://example.com", title="Long", accessibility_tree="x" * 2000)
    loop._trajectory = []
    messages = loop._build_recovery_messages(
        system_prompt="sys",
        user_prompt=f"obs {long_obs.accessibility_tree}",
        failure_class="recoverable_truncated_tool_json",
    )

    raw, action, usage, override = await loop._attempt_format_recovery(
        model="qwen",
        seed=1,
        raw_response="<tool_call>{\"name\":\"goto\"",
        messages=messages,
        failure_class="recoverable_truncated_tool_json",
    )

    assert action is None
    assert usage is None
    assert override == "recoverable_context_overflow"
    assert llm_client.recovery_calls == 0


@pytest.mark.anyio
async def test_agent_loop_explicit_disable_recovery_overrides_env(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_FORMAT_RECOVERY", "1")

    llm_client = _FakeLLMClient(
        initial_response=LLMResponse(content="<tool_call>{\"name\":\"goto\""),
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
        enable_format_recovery=False,
    )

    trajectory, final_answer, usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_parse_failed() is True
    assert llm_client.recovery_calls == 0


@pytest.mark.anyio
async def test_agent_loop_failfasts_after_consecutive_disallowed_domain_hits(monkeypatch):
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_CONSECUTIVE", "2")
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_TOTAL", "3")
    monkeypatch.setenv("LIVEWEB_DISALLOWED_DOMAIN_RECOVERY_MAX_RETRIES", "1")

    blocked_obs = BrowserObservation(
        url="https://finance.yahoo.com/quote/V/",
        title="Domain not allowed",
        accessibility_tree="Domain not allowed. Please return to an allowed site and continue the task.",
    )
    session = _FakeSessionWithBlockedSequence(
        observations=[blocked_obs, blocked_obs],
        metadatas=[
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {
                    "blocked_url": "https://finance.yahoo.com/quote/V/",
                    "allowed_domains": ["coingecko.com"],
                },
            },
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {
                    "blocked_url": "https://finance.yahoo.com/quote/V/",
                    "allowed_domains": ["coingecko.com"],
                },
            },
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=5,
    )

    with pytest.raises(BrowserFatalError) as exc_info:
        await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert "disallowed-domain" in str(exc_info.value)
    assert exc_info.value.url == "https://finance.yahoo.com/quote/V/"
    assert len(loop.get_trajectory()) == 0
    assert llm_client.recovery_calls == 1


@pytest.mark.anyio
async def test_agent_loop_resets_disallowed_domain_consecutive_counter_after_recovery(monkeypatch):
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_CONSECUTIVE", "2")
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_TOTAL", "3")
    monkeypatch.setenv("LIVEWEB_DISALLOWED_DOMAIN_RECOVERY_MAX_RETRIES", "1")

    blocked_obs = BrowserObservation(
        url="https://finance.yahoo.com/quote/V/",
        title="Domain not allowed",
        accessibility_tree="Domain not allowed. Please return to an allowed site and continue the task.",
    )
    good_obs = BrowserObservation(
        url="https://example.com",
        title="Example",
        accessibility_tree="root content with enough accessible text to avoid blank observation failfast",
    )
    session = _FakeSessionWithBlockedSequence(
        observations=[blocked_obs, good_obs, blocked_obs, good_obs],
        metadatas=[
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {
                    "blocked_url": "https://finance.yahoo.com/quote/V/",
                    "allowed_domains": ["example.com"],
                },
            },
            None,
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {
                    "blocked_url": "https://finance.yahoo.com/quote/V/",
                    "allowed_domains": ["example.com"],
                },
            },
            None,
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_2", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_3", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "goto", "arguments": "{\"url\":\"https://example.com\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_r2", function={"name": "goto", "arguments": "{\"url\":\"https://example.com\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=5,
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert len(trajectory) == 3
    assert llm_client.recovery_calls == 2
    assert trajectory[0].action.params["url"] == "https://example.com"
    assert trajectory[0].action_result == "Success (recovered after disallowed-domain retry)"
    assert trajectory[1].action.params["url"] == "https://example.com"


@pytest.mark.anyio
async def test_agent_loop_recovers_disallowed_domain_without_appending_blocked_step(monkeypatch):
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_CONSECUTIVE", "2")
    monkeypatch.setenv("LIVEWEB_FAILFAST_DISALLOWED_DOMAIN_TOTAL", "3")
    monkeypatch.setenv("LIVEWEB_DISALLOWED_DOMAIN_RECOVERY_MAX_RETRIES", "1")

    blocked_obs = BrowserObservation(
        url="https://finance.yahoo.com/quote/V/",
        title="Domain not allowed",
        accessibility_tree="Domain not allowed. Please return to an allowed site and continue the task.",
    )
    good_obs = BrowserObservation(
        url="https://www.coingecko.com/en/coins/optimism",
        title="Optimism",
        accessibility_tree="Circulating Supply 2,117,847,344 OP",
    )
    session = _FakeSessionWithBlockedSequence(
        observations=[blocked_obs, good_obs],
        metadatas=[
            {
                "classification_hint": "model_disallowed_domain",
                "url": "https://finance.yahoo.com/quote/V/",
                "evidence": {
                    "blocked_url": "https://finance.yahoo.com/quote/V/",
                    "allowed_domains": ["coingecko.com", "taostats.io"],
                },
            },
            None,
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/V/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_2", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "goto", "arguments": "{\"url\":\"https://www.coingecko.com/en/coins/optimism\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=4,
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert len(trajectory) == 2
    assert trajectory[0].action.params["url"] == "https://www.coingecko.com/en/coins/optimism"
    assert "recovered after disallowed-domain retry" in trajectory[0].action_result.lower()
    assert llm_client.recovery_calls == 1
    messages = llm_client.recovery_messages[0]
    assert "Blocked URL: https://finance.yahoo.com/quote/V/" in messages[-1]["content"]
    assert "Allowed domains: coingecko.com, taostats.io" in messages[-1]["content"]


@pytest.mark.anyio
async def test_agent_loop_recovers_empty_stop_without_appending_bad_stop():
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "stop", "arguments": "{\"answers\":{}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\",\"a2\":\"Trishool\"}}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task_two_answers(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42", "a2": "Trishool"}}
    assert len(trajectory) == 1
    assert trajectory[0].action.action_type == "stop"
    assert "recovered" not in trajectory[0].action_result.lower()
    local_stats = loop.get_local_recovery_stats()
    assert local_stats["empty_stop_recovery_attempts"] == 1
    assert local_stats["empty_stop_recovery_successes"] == 1


@pytest.mark.anyio
async def test_agent_loop_marks_invalid_stop_payload_when_recovery_exhausted():
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "stop", "arguments": "{\"answers\":{}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "stop", "arguments": "{\"answers\":{}}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task_two_answers(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_invalid_stop_payload() is True
    assert loop.get_last_stop_failure_class() == "empty_answers"
    assert len(trajectory) == 1
    assert trajectory[0].action.action_type == "stop"
    assert "invalid stop payload" in trajectory[0].action_result.lower()


@pytest.mark.anyio
async def test_agent_loop_recovers_invalid_ui_target_with_same_page_action():
    taostats_obs = BrowserObservation(
        url="https://taostats.io/subnets",
        title="Subnets",
        accessibility_tree="Subnet list root content with enough text for recovery decisions",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[
            RuntimeError("No element found with role='button' name='Rows: 25'"),
            taostats_obs,
        ],
        metadatas=[
            {
                "url": "https://taostats.io/subnets",
                "navigation_stage": "action_click_role",
                "raw_exception_type": "RuntimeError",
                "raw_exception_message": "No element found with role='button' name='Rows: 25'",
                "evidence": {
                    "ui_target_missing": True,
                    "page_kind": "taostats_list",
                    "interaction_kind": "show_all",
                    "target_locator": "role=button[name='Rows: 25']",
                },
            },
            None,
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "click_role", "arguments": "{\"role\":\"button\",\"name\":\"Rows: 25\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_2", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "scroll", "arguments": "{\"direction\":\"down\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=3,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert trajectory[0].action.action_type == "scroll"
    assert "invalid_ui_target_recovery" in trajectory[0].action_result
    stats = loop.get_local_recovery_stats()
    assert stats["invalid_ui_target_recovery_attempts"] == 1
    assert stats["invalid_ui_target_recovery_successes"] == 1


@pytest.mark.anyio
async def test_agent_loop_rejects_invalid_ui_target_recovery_that_uses_goto():
    taostats_obs = BrowserObservation(
        url="https://taostats.io/subnets",
        title="Subnets",
        accessibility_tree="Subnet list root content with enough text for recovery decisions",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[
            RuntimeError("No element found with role='button' name='Rows: 25'"),
        ],
        metadatas=[
            {
                "url": "https://taostats.io/subnets",
                "navigation_stage": "action_click_role",
                "raw_exception_type": "RuntimeError",
                "raw_exception_message": "No element found with role='button' name='Rows: 25'",
                "evidence": {
                    "ui_target_missing": True,
                    "page_kind": "taostats_list",
                    "interaction_kind": "show_all",
                    "target_locator": "role=button[name='Rows: 25']",
                },
            },
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "click_role", "arguments": "{\"role\":\"button\",\"name\":\"Rows: 25\"}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=1,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert len(trajectory) == 1
    assert trajectory[0].action.action_type == "click_role"
    assert trajectory[0].action_result.startswith("Failed:")


@pytest.mark.anyio
async def test_agent_loop_recovers_taostats_list_timeout_with_same_page_action():
    taostats_obs = BrowserObservation(
        url="https://taostats.io/subnets",
        title="Subnets",
        accessibility_tree="Subnet list root content with enough text for recovery decisions",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[
            RuntimeError("Page.click: Timeout 5000ms exceeded."),
            taostats_obs,
        ],
        metadatas=[
            {
                "url": "https://taostats.io/subnets",
                "navigation_stage": "action_click",
                "raw_exception_type": "TimeoutError",
                "raw_exception_message": "Page.click: Timeout 5000ms exceeded.",
                "evidence": {
                    "page_kind": "taostats_list",
                    "interaction_kind": "sort",
                    "target_locator": "thead th:has-text(\"24H\")",
                },
            },
            None,
        ],
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "click", "arguments": "{\"selector\":\"thead th:has-text(\\\"24H\\\")\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_2", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "scroll", "arguments": "{\"direction\":\"down\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=3,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert trajectory[0].action.action_type == "scroll"
    assert "taostats_list_action_recovery" in trajectory[0].action_result
    stats = loop.get_local_recovery_stats()
    assert stats["taostats_list_action_recovery_attempts"] == 1
    assert stats["taostats_list_action_recovery_successes"] == 1


@pytest.mark.anyio
async def test_agent_loop_collect_only_recoveries_are_disabled_in_eval_mode():
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "stop", "arguments": "{\"answers\":{}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=1,
        behavior_mode="eval",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {}}
    assert llm_client.recovery_calls == 0
    assert len(trajectory) == 1


@pytest.mark.anyio
async def test_agent_loop_recovers_invalid_generated_url_without_appending_bad_step():
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "goto", "arguments": "{\"url\":\"https://:\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="call_2", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "goto", "arguments": "{\"url\":\"https://example.com\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=3,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer == {"answers": {"a1": "42"}}
    assert loop.is_invalid_generated_url() is False
    assert len(trajectory) == 2
    assert trajectory[0].action.params["url"] == "https://example.com"
    assert "invalid_generated_url_recovery" in trajectory[0].action_result
    stats = loop.get_local_recovery_stats()
    assert stats["invalid_generated_url_recovery_attempts"] == 1
    assert stats["invalid_generated_url_recovery_successes"] == 1


@pytest.mark.anyio
async def test_agent_loop_marks_invalid_generated_url_when_recovery_exhausted():
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_1", function={"name": "goto", "arguments": "{\"url\":\"https://:\"}"})]),
        ],
        recovery_responses=[
            LLMResponse(tool_calls=[ToolCall(id="call_r1", function={"name": "goto", "arguments": "{\"url\":\"https://:\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=_FakeSession(),
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=2,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(task=_task(), model="qwen", temperature=0.0, seed=1)

    assert final_answer is None
    assert loop.is_invalid_generated_url() is True
    assert loop.get_last_invalid_generated_url_detail()["kind"] == "missing_host"
    assert len(trajectory) == 1
    assert "Invalid generated URL" in trajectory[0].action_result


@pytest.mark.anyio
async def test_agent_loop_detects_repetitive_same_action_loop_for_gemini(monkeypatch):
    monkeypatch.setenv("LIVEWEB_GEMINI_LOOP_SAME_ACTION_THRESHOLD", "6")

    repeated_obs = BrowserObservation(
        url="https://taostats.io/subnets/",
        title="Subnets",
        accessibility_tree="Subnet table with enough accessible text to support repeated interaction.",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[repeated_obs] * 6,
        metadatas=[None] * 6,
    )
    click_args = "{\"selector\":\"th:nth-child(3)\"}"
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="goto_1", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnets/\"}"})]),
            *[
                LLMResponse(tool_calls=[ToolCall(id=f"click_{idx}", function={"name": "click", "arguments": click_args})])
                for idx in range(6)
            ],
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=10,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(
        task=_task(), model="google/gemini-3-flash-preview", temperature=0.0, seed=1
    )

    assert final_answer is None
    assert loop.is_action_loop_detected() is True
    assert loop.get_last_action_loop_detail()["kind"] == "same_action_repeat"
    assert trajectory[-1].action_result.startswith("Aborted: repetitive_action_loop")
    assert "th:nth-child(3)" in trajectory[-1].action_result


@pytest.mark.anyio
async def test_agent_loop_detects_repetitive_same_type_loop_for_gemini(monkeypatch):
    monkeypatch.setenv("LIVEWEB_GEMINI_LOOP_SAME_TYPE_THRESHOLD", "4")

    search_obs = BrowserObservation(
        url="https://taostats.io/subnets/",
        title="Subnets",
        accessibility_tree="Subnet list page with search box and enough accessible text.",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[search_obs] * 5,
        metadatas=[None] * 5,
    )
    type_args = "{\"selector\":\"input[placeholder='Search']\",\"text\":\"SN23\"}"
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="goto_1", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnets/\"}"})]),
            *[
                LLMResponse(tool_calls=[ToolCall(id=f"type_{idx}", function={"name": "type", "arguments": type_args})])
                for idx in range(4)
            ],
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=8,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(
        task=_task(), model="google/gemini-3-flash-preview", temperature=0.0, seed=1
    )

    assert final_answer is None
    assert loop.is_action_loop_detected() is True
    assert loop.get_last_action_loop_detail()["kind"] == "same_type_repeat"


@pytest.mark.anyio
async def test_agent_loop_detects_goto_oscillation_for_gemini(monkeypatch):
    monkeypatch.setenv("LIVEWEB_GEMINI_LOOP_GOTO_OSCILLATION_THRESHOLD", "4")

    subnet_list_obs = BrowserObservation(
        url="https://taostats.io/subnets/",
        title="Subnets",
        accessibility_tree="Subnet list with enough accessible text to support navigation.",
    )
    subnet_detail_obs = BrowserObservation(
        url="https://taostats.io/subnet/23/",
        title="Subnet 23",
        accessibility_tree="Subnet detail with enough accessible text to support navigation.",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[
            subnet_list_obs,
            subnet_detail_obs,
            subnet_list_obs,
            subnet_detail_obs,
            subnet_list_obs,
            subnet_detail_obs,
            subnet_list_obs,
        ],
        metadatas=[None] * 7,
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="goto_1", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnets/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_2", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnet/23/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_3", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnets/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_4", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnet/23/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_5", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnets/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_6", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnet/23/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_7", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnets/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_8", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnet/23/\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=12,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(
        task=_task(), model="google/gemini-3-flash-preview", temperature=0.0, seed=1
    )

    assert final_answer is None
    assert loop.is_action_loop_detected() is True
    detail = loop.get_last_action_loop_detail()
    assert detail["kind"] == "goto_oscillation"
    assert set(detail["oscillation_urls"]) == {
        "https://taostats.io/subnets/",
        "https://taostats.io/subnet/23/",
    }
    assert trajectory[-1].action_result.startswith("Aborted: repetitive_action_loop")


@pytest.mark.anyio
async def test_agent_loop_detects_search_finance_bounce_for_gemini(monkeypatch):
    monkeypatch.setenv("LIVEWEB_GEMINI_LOOP_SEARCH_BOUNCE_THRESHOLD", "4")

    google_obs = BrowserObservation(
        url="https://www.google.com/search?q=dogecoin",
        title="Google Search",
        accessibility_tree="Search results page with enough text to support navigation.",
    )
    yahoo_obs = BrowserObservation(
        url="https://finance.yahoo.com/quote/DOGE-USD/",
        title="Yahoo Finance",
        accessibility_tree="Finance quote page with enough text to support navigation.",
    )
    cmc_obs = BrowserObservation(
        url="https://coinmarketcap.com/currencies/dogecoin/",
        title="CoinMarketCap",
        accessibility_tree="Coin page with enough text to support navigation.",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[google_obs, yahoo_obs, google_obs, cmc_obs],
        metadatas=[None] * 4,
    )
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="goto_1", function={"name": "goto", "arguments": "{\"url\":\"https://www.google.com/search?q=dogecoin\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_2", function={"name": "goto", "arguments": "{\"url\":\"https://finance.yahoo.com/quote/DOGE-USD/\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_3", function={"name": "goto", "arguments": "{\"url\":\"https://www.google.com/search?q=dogecoin\"}"})]),
            LLMResponse(tool_calls=[ToolCall(id="goto_4", function={"name": "goto", "arguments": "{\"url\":\"https://coinmarketcap.com/currencies/dogecoin/\"}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=6,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(
        task=_task(), model="google/gemini-3-flash-preview", temperature=0.0, seed=1
    )

    assert final_answer is None
    assert loop.is_action_loop_detected() is True
    assert loop.get_last_action_loop_detail()["kind"] == "search_finance_bounce"


@pytest.mark.anyio
async def test_agent_loop_does_not_apply_loop_guard_to_non_gemini(monkeypatch):
    monkeypatch.setenv("LIVEWEB_GEMINI_LOOP_SAME_ACTION_THRESHOLD", "6")

    repeated_obs = BrowserObservation(
        url="https://taostats.io/subnets/",
        title="Subnets",
        accessibility_tree="Subnet table with enough accessible text to support repeated interaction.",
    )
    session = _FakeSessionWithActionSequence(
        outcomes=[repeated_obs] * 7,
        metadatas=[None] * 7,
    )
    click_args = "{\"selector\":\"th:nth-child(3)\"}"
    llm_client = _QueuedLLMClient(
        responses=[
            LLMResponse(tool_calls=[ToolCall(id="goto_1", function={"name": "goto", "arguments": "{\"url\":\"https://taostats.io/subnets/\"}"})]),
            *[
                LLMResponse(tool_calls=[ToolCall(id=f"click_{idx}", function={"name": "click", "arguments": click_args})])
                for idx in range(6)
            ],
            LLMResponse(tool_calls=[ToolCall(id="stop_1", function={"name": "stop", "arguments": "{\"answers\":{\"a1\":\"42\"}}"})]),
        ],
    )
    loop = AgentLoop(
        session=session,
        llm_client=llm_client,
        protocol=FunctionCallingProtocol(),
        max_steps=12,
        behavior_mode="collect",
    )

    trajectory, final_answer, _usage = await loop.run(
        task=_task(), model="mimo_v2_pro_openrouter", temperature=0.0, seed=1
    )

    assert final_answer == {"answers": {"a1": "42"}}
    assert loop.is_action_loop_detected() is False
    assert trajectory[-1].action.action_type == "stop"
