import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from liveweb_arena.utils.llm_client import LLMClient, LLMResponse


class _FakeResponseUsage:
    def __init__(self, payload=None):
        self._payload = payload or {"prompt_tokens": 10, "completion_tokens": 2}

    def model_dump(self):
        return dict(self._payload)


class _FakeToolCall:
    def __init__(self, name: str, arguments: str):
        self.id = "call_1"
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeChoice:
    def __init__(self, content: str = "", tool_calls=None, reasoning_content=None):
        self.message = SimpleNamespace(content=content, tool_calls=tool_calls or [], reasoning_content=reasoning_content)
        self.finish_reason = "stop"


class _FakeChatCompletions:
    def __init__(self, recorder, *, sleep_s: float = 0.0):
        self._recorder = recorder
        self._sleep_s = sleep_s

    async def create(self, **kwargs):
        self._recorder.append(kwargs)
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        return SimpleNamespace(
            id=kwargs["extra_body"]["request_id"],
            choices=[_FakeChoice(tool_calls=[_FakeToolCall("goto", '{"url":"https://example.com"}')])],
            usage=_FakeResponseUsage(),
        )


class _FakeAsyncOpenAI:
    def __init__(self, recorder, *, sleep_s: float = 0.0, **kwargs):
        self._recorder = recorder
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(recorder, sleep_s=sleep_s))

    async def close(self):
        return None


class _FakeHTTPXResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


class _FakeAsyncHTTPClient:
    def __init__(self, recorder, *args, **kwargs):
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers=None):
        self._recorder.append((url, json, headers or {}))
        return _FakeHTTPXResponse()


class _FakeBuiltHTTPClient:
    def __init__(self, recorder, *args, **kwargs):
        recorder.append(kwargs)


class _FakeRateLimitError(Exception):
    def __init__(self, message: str, *, response):
        super().__init__(message)
        self.response = response


class _RateLimitThenSuccessCompletions:
    def __init__(self, recorder, failures_before_success: int = 2):
        self._recorder = recorder
        self._remaining = failures_before_success

    async def create(self, **kwargs):
        self._recorder.append(kwargs)
        if self._remaining > 0:
            self._remaining -= 1
            response = httpx.Response(
                429,
                headers={"retry-after": "0"},
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            )
            raise _FakeRateLimitError("rate limited", response=response)
        return SimpleNamespace(
            id=kwargs["extra_body"]["request_id"],
            choices=[_FakeChoice(tool_calls=[_FakeToolCall("goto", '{"url":"https://example.com"}')])],
            usage=_FakeResponseUsage(),
        )


class _RateLimitThenSuccessOpenAI:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)

    async def close(self):
        return None


class _ContextLimitThenSuccessCompletions:
    def __init__(self, recorder):
        self._recorder = recorder
        self._calls = 0

    async def create(self, **kwargs):
        self._recorder.append(kwargs)
        self._calls += 1
        if self._calls == 1:
            raise openai.BadRequestError(
                "Requested token count exceeds the model's maximum context length of 32768 tokens. "
                "You requested a total of 34078 tokens: 1310 tokens from the input messages and 32768 tokens "
                "for the completion. Please reduce the number of tokens in the input messages or the completion "
                "to fit within the limit."
            )
        return SimpleNamespace(
            id=kwargs["extra_body"]["request_id"],
            choices=[_FakeChoice(tool_calls=[_FakeToolCall("goto", '{"url":"https://example.com"}')])],
            usage=_FakeResponseUsage(),
        )


class _ContextLimitThenSuccessOpenAI:
    def __init__(self, recorder):
        self.chat = SimpleNamespace(completions=_ContextLimitThenSuccessCompletions(recorder))

    async def close(self):
        return None


@pytest.mark.anyio
async def test_chat_with_tools_sends_request_id_in_extra_body(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    response = await client.chat_with_tools(
        system="system",
        user="user",
        model="qwen",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    assert isinstance(response, LLMResponse)
    assert requests
    request_payload = requests[0]
    assert "extra_body" in request_payload
    assert request_payload["extra_body"]["request_id"].startswith("liveweb-tools-")
    assert response.request_id == request_payload["extra_body"]["request_id"]


@pytest.mark.anyio
async def test_chat_with_tools_sends_reasoning_controls_via_extra_body(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)
    monkeypatch.setenv("LIVEWEB_ENABLE_THINKING", "0")
    monkeypatch.setenv("LIVEWEB_SEPARATE_REASONING", "1")

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    await client.chat_with_tools(
        system="system",
        user="user",
        model="qwen",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    payload = requests[0]
    assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["extra_body"]["separate_reasoning"] is True


@pytest.mark.anyio
async def test_chat_with_tools_uses_openrouter_reasoning_none_when_thinking_disabled(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)
    monkeypatch.setenv("LIVEWEB_ENABLE_THINKING", "0")

    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="or")
    await client.chat_with_tools(
        system="system",
        user="user",
        model="minimax/minimax-m2.7",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    payload = requests[0]
    assert payload["extra_body"]["reasoning"] == {"effort": "none", "exclude": True}


@pytest.mark.anyio
async def test_chat_with_tools_uses_kimi_reasoning_enabled_false_when_thinking_disabled(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)
    monkeypatch.setenv("LIVEWEB_ENABLE_THINKING", "0")

    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="or")
    await client.chat_with_tools(
        system="system",
        user="user",
        model="moonshotai/kimi-k2.5",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    payload = requests[0]
    assert payload["extra_body"]["reasoning"] == {"enabled": False}


@pytest.mark.anyio
async def test_chat_with_tools_uses_low_reasoning_effort_when_enabled(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)

    client = LLMClient(
        base_url="https://api.aicodemirror.com/api/codex/backend-api/codex/v1",
        api_key="k",
        enable_thinking=True,
        separate_reasoning=True,
        reasoning_effort="low",
        strip_reasoning_output=True,
    )
    await client.chat_with_tools(
        system="system",
        user="user",
        model="gpt-5.4",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    payload = requests[0]
    assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["extra_body"]["separate_reasoning"] is True
    assert payload["extra_body"]["reasoning"] == {"effort": "low"}


@pytest.mark.anyio
async def test_chat_with_tools_strips_reasoning_output_from_content_and_usage(monkeypatch):
    requests = []

    class _ReasoningOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                id=kwargs["extra_body"]["request_id"],
                choices=[
                    _FakeChoice(
                        content=[
                            {"type": "reasoning", "text": "hidden chain of thought"},
                            {"type": "output_text", "text": "Visible answer"},
                        ],
                        tool_calls=[],
                        reasoning_content="hidden chain of thought",
                    )
                ],
                usage=_FakeResponseUsage(
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "completion_tokens_details": {"reasoning_tokens": 9, "accepted_prediction_tokens": 0},
                    }
                ),
            )

        async def close(self):
            return None

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _ReasoningOpenAI)

    client = LLMClient(
        base_url="https://api.aicodemirror.com/api/codex/backend-api/codex/v1",
        api_key="k",
        enable_thinking=True,
        separate_reasoning=True,
        reasoning_effort="low",
        strip_reasoning_output=True,
    )
    response = await client.chat_with_tools(
        system="system",
        user="user",
        model="gpt-5.4",
        tools=None,
        temperature=0.0,
        timeout_s=5,
    )

    assert response.content == "Visible answer"
    assert response.usage["completion_tokens_details"] == {"accepted_prediction_tokens": 0}


@pytest.mark.anyio
async def test_chat_with_tools_recovery_uses_stochastic_small_request(monkeypatch):
    requests = []

    def _factory(**kwargs):
        return _FakeAsyncOpenAI(requests, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _factory)
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TEMPERATURE", "0.35")
    monkeypatch.setenv("LIVEWEB_FORMAT_RECOVERY_TOP_P", "0.95")

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    await client.chat_with_tools_recovery(
        system="system",
        user="user prompt",
        assistant_prefix="<tool_call>{\"name\":\"goto\"",
        model="qwen",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        max_new_tokens=96,
    )

    payload = requests[0]
    assert payload["temperature"] == 0.35
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 96
    assert payload["max_completion_tokens"] == 96
    assert payload["messages"][2] == {"role": "assistant", "content": "<tool_call>{\"name\":\"goto\""}
    assert "Continue from the assistant's last output" in payload["messages"][3]["content"]


@pytest.mark.anyio
async def test_chat_with_tools_timeout_triggers_abort(monkeypatch):
    requests = []
    aborts = []

    def _openai_factory(**kwargs):
        return _FakeAsyncOpenAI(requests, sleep_s=0.2, **kwargs)

    def _httpx_factory(*args, **kwargs):
        return _FakeAsyncHTTPClient(aborts, *args, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", _openai_factory)
    monkeypatch.setattr("liveweb_arena.utils.llm_client.httpx.AsyncClient", _httpx_factory)

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local", strict_serial=True)

    with pytest.raises(Exception):
        await client.chat_with_tools(
            system="system",
            user="user",
            model="qwen",
            tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
            temperature=0.0,
            timeout_s=0.01,
        )

    assert requests
    assert aborts
    abort_url, abort_payload, abort_headers = aborts[0]
    assert abort_url == "http://127.0.0.1:31050/abort_request"
    assert abort_payload["rid"] == requests[0]["extra_body"]["request_id"]
    assert abort_payload["abort_all"] is False
    assert abort_headers["Authorization"] == "Bearer local"
    assert abort_headers["X-API-Key"] == "local"


@pytest.mark.parametrize(
    ("base_url", "should_bypass"),
    [
        ("http://localhost:31050/v1", True),
        ("http://127.0.0.1:31050/v1", True),
        ("http://10.0.0.8:31050/v1", True),
        ("http://192.168.1.10:31050/v1", True),
        ("http://172.16.0.5:31050/v1", True),
        ("http://169.254.0.20:31050/v1", True),
        ("http://[::1]:31050/v1", True),
        ("https://api.aicodemirror.com/api/codex/backend-api/codex/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://example.com/v1", False),
    ],
)
def test_should_bypass_proxy_for_local_and_private_addresses(base_url, should_bypass):
    assert LLMClient._should_bypass_proxy(base_url) is should_bypass


def test_build_httpx_client_disables_trust_env_for_local_routes(monkeypatch):
    built_clients = []

    def _httpx_factory(*args, **kwargs):
        return _FakeBuiltHTTPClient(built_clients, *args, **kwargs)

    monkeypatch.setattr("liveweb_arena.utils.llm_client.httpx.AsyncClient", _httpx_factory)

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    client._build_httpx_client(
        base_url="http://127.0.0.1:31050/v1",
        timeout=object(),
    )
    client._build_httpx_client(
        base_url="https://api.openai.com/v1",
        timeout=object(),
    )
    client._build_httpx_client(
        base_url="https://api.aicodemirror.com/api/codex/backend-api/codex/v1",
        timeout=object(),
    )

    assert built_clients[0]["trust_env"] is False
    assert built_clients[1]["trust_env"] is True
    assert built_clients[2]["trust_env"] is False


@pytest.mark.anyio
async def test_chat_with_tools_records_empty_response_audit(monkeypatch):
    requests = []

    class _EmptyChatCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                id=kwargs["extra_body"]["request_id"],
                choices=[_FakeChoice(content="", tool_calls=[])],
                usage=_FakeResponseUsage(),
            )

    class _EmptyAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_EmptyChatCompletions())

        async def close(self):
            return None

    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.AsyncOpenAI", lambda **kwargs: _EmptyAsyncOpenAI(**kwargs))

    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="k", strict_serial=True)
    with pytest.raises(Exception):
        await client.chat_with_tools(
            system="system",
            user="user",
            model="z-ai/glm-5",
            tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
            temperature=0.0,
            timeout_s=5,
        )

    audit = client.get_last_failure_metadata()
    assert audit["failure_type"] == "empty_response"
    assert audit["model"] == "z-ai/glm-5"
    assert audit["base_url"] == "https://openrouter.ai/api/v1"
    assert audit["usage"]["completion_tokens"] == 2


@pytest.mark.anyio
async def test_chat_with_tools_retries_on_rate_limit_even_in_strict_serial(monkeypatch):
    requests = []
    completions = _RateLimitThenSuccessCompletions(requests)

    monkeypatch.setattr(
        "liveweb_arena.utils.llm_client.openai.AsyncOpenAI",
        lambda **kwargs: _RateLimitThenSuccessOpenAI(completions),
    )
    monkeypatch.setattr("liveweb_arena.utils.llm_client.openai.RateLimitError", _FakeRateLimitError)

    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="k", strict_serial=True)
    response = await client.chat_with_tools(
        system="system",
        user="user",
        model="minimax/minimax-m2.7",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    assert isinstance(response, LLMResponse)
    assert len(requests) == 3


@pytest.mark.anyio
async def test_chat_with_tools_reduces_completion_cap_after_context_error(monkeypatch):
    requests = []
    completions = _ContextLimitThenSuccessCompletions(requests)
    monkeypatch.setenv("LIVEWEB_MAX_COMPLETION_TOKENS", "32768")
    monkeypatch.setattr(
        "liveweb_arena.utils.llm_client.openai.AsyncOpenAI",
        lambda **kwargs: _RateLimitThenSuccessOpenAI(completions),
    )

    client = LLMClient(base_url="http://127.0.0.1:31050/v1", api_key="local")
    async def _noop_abort(*args, **kwargs):
        return None
    monkeypatch.setattr(client, "_abort_request", _noop_abort)
    response = await client.chat_with_tools(
        system="system",
        user="user",
        model="qwen",
        tools=[{"type": "function", "function": {"name": "goto", "parameters": {"type": "object"}}}],
        temperature=0.0,
        timeout_s=5,
    )

    assert isinstance(response, LLMResponse)
    assert len(requests) == 2
    assert requests[0]["max_completion_tokens"] == 32768
    assert requests[1]["max_completion_tokens"] < 32768


@pytest.mark.anyio
async def test_global_rate_limit_window_delays_following_requests(monkeypatch):
    client = LLMClient(base_url="https://openrouter.ai/api/v1", api_key="k", strict_serial=True)
    client._GLOBAL_RATE_LIMIT_UNTIL[client._base_url] = asyncio.get_running_loop().time() + 0.2
    # convert monotonic-ish test window to wall time used by implementation
    import time as _time
    client._GLOBAL_RATE_LIMIT_UNTIL[client._base_url] = _time.time() + 0.2
    start = _time.time()
    await client._wait_for_global_rate_limit_window(client._base_url)
    elapsed = _time.time() - start
    assert elapsed >= 0.15
