"""Tests for provider-specific protocols and ordered fallback."""

import json

import pytest

from oi_agent.agent.llm import LLMError, chat
from oi_agent.config import LLMConfig


class FakeResponse:
    """Minimal response object for the HTTP client boundary."""

    def __init__(self, status_code: int, payload: dict):
        """Store a status code and JSON payload.

        Args:
            status_code: HTTP response status.
            payload: JSON response body.
        """
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        """Raise when the simulated provider rejects the request."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        """Return the simulated JSON body."""
        return self._payload


class FakeClient:
    """Async HTTP client returning scripted provider responses."""

    calls = []
    responses = []

    def __init__(self, *args, **kwargs):
        """Accept the same construction shape as httpx.AsyncClient."""

    async def __aenter__(self):
        """Enter the fake client."""
        return self

    async def __aexit__(self, *args):
        """Leave the fake client."""
        return False

    async def post(self, url, headers, json):
        """Record a request and return the next scripted response."""
        self.calls.append((url, headers, json))
        return self.responses.pop(0)


class FakeStreamResponse:
    """Minimal SSE response for visible-answer streaming tests."""

    def __init__(self, lines):
        """Store SSE lines.

        Args:
            lines: Iterable of raw SSE lines.
        """
        self.status_code = 200
        self._lines = lines

    def raise_for_status(self):
        """Accept the simulated successful response."""

    async def aiter_lines(self):
        """Yield the scripted SSE lines."""
        for line in self._lines:
            yield line


class FakeStreamContext:
    """Async context manager wrapping a fake stream response."""

    def __init__(self, response):
        """Store the response returned on enter.

        Args:
            response: Fake streaming response.
        """
        self.response = response

    async def __aenter__(self):
        """Return the streaming response."""
        return self.response

    async def __aexit__(self, *args):
        """Close the fake stream."""
        return False


class StreamingClient(FakeClient):
    """Async client exposing a scripted streaming response."""

    lines = []

    def stream(self, method, url, headers, json):
        """Record a stream request and return its fake context."""
        self.calls.append((url, headers, json))
        return FakeStreamContext(FakeStreamResponse(self.lines))


@pytest.mark.asyncio
async def test_zai_anthropic_failure_falls_back_to_openai_provider(monkeypatch):
    """A failed large Z.AI call uses the responding small-model provider."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        FakeResponse(429, {"error": "rate limited"}),
        FakeResponse(200, {
            "choices": [{"message": {"content": "small answer"}}],
        }),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("BIG_KEY", "zai-secret")
    monkeypatch.setenv("SMALL_KEY", "hetzner-secret")
    big = LLMConfig(
        "https://api.z.ai", "glm-4.6", "BIG_KEY", "anthropic"
    )
    small = LLMConfig(
        "https://inference.hetzner.com/api/v1", "Qwen3.8-27B", "SMALL_KEY"
    )

    text, provider = await chat([big, small], "system", "question")

    assert text == "small answer"
    assert provider.endswith("/Qwen3.8-27B")
    assert FakeClient.calls[0][0] == (
        "https://api.z.ai/api/anthropic/v1/messages"
    )
    assert FakeClient.calls[0][1]["x-api-key"] == "zai-secret"
    assert "Authorization" not in FakeClient.calls[0][1]
    assert FakeClient.calls[1][0] == (
        "https://inference.hetzner.com/api/v1/chat/completions"
    )
    assert FakeClient.calls[1][1]["Authorization"] == "Bearer hetzner-secret"
    assert FakeClient.calls[1][2]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_disable_thinking_sets_chat_template_kwarg(monkeypatch):
    """disable_thinking injects the vLLM/Qwen enable_thinking=false lever."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [FakeResponse(200, {
        "choices": [{"message": {"content": "answer"}}],
    })]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    await chat([LLMConfig("https://small.example/v1", "Qwen3.8-27B",
                          disable_thinking=True)], "system", "question")

    sent = FakeClient.calls[0][2]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_debug_wire_logs_request_and_response(monkeypatch, caplog):
    """OI_LLM_DEBUG_WIRE logs the sent payload and raw response verbatim."""
    from oi_agent.agent import llm

    caplog.set_level("WARNING")
    FakeClient.calls = []
    FakeClient.responses = [FakeResponse(200, {
        "choices": [{"message": {"content": "answer"}}],
    })]
    # FakeResponse needs a .text for the wire logger.
    FakeClient.responses[0].text = '{"choices":[{"message":{"content":"answer"}}]}'
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("OI_LLM_DEBUG_WIRE", "1")

    await chat([LLMConfig("https://small.example/v1", "Qwen3.8-27B",
                          disable_thinking=True)], "system", "question")

    assert "[llm][wire] endpoint=" in caplog.text
    assert "chat_template_kwargs" in caplog.text
    assert "raw_response=" in caplog.text


@pytest.mark.asyncio
async def test_thinking_kwarg_absent_by_default(monkeypatch):
    """Without disable_thinking the request stays a plain completion."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [FakeResponse(200, {
        "choices": [{"message": {"content": "answer"}}],
    })]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    await chat([LLMConfig("https://small.example/v1", "model")],
               "system", "question")

    assert "chat_template_kwargs" not in FakeClient.calls[0][2]


@pytest.mark.asyncio
async def test_anthropic_response_parses_content_blocks(monkeypatch):
    """The Anthropic response shape is converted to plain reply text."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [FakeResponse(200, {
        "content": [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ],
    })]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("ZAI_KEY", "zai-secret")

    text, _ = await chat([
        LLMConfig("https://api.z.ai", "glm-4.6", "ZAI_KEY", "anthropic")
    ], "system", "question")

    assert text == "part onepart two"
    assert FakeClient.calls[0][2]["max_tokens"] == 1024
    assert FakeClient.calls[0][2]["system"] == "system"


@pytest.mark.asyncio
async def test_empty_content_raises_without_leaking_reasoning(monkeypatch, caplog):
    """Empty content raises (honest error to Discord); reasoning stays in logs.

    Reasoning is the model's private scratchpad — it must never become the
    posted reply. It is logged as shape metadata only, never as text.
    """
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [FakeResponse(200, {
        "choices": [{
            "finish_reason": "length",
            "message": {"content": " ", "reasoning": "private chain"},
        }],
        "usage": {"completion_tokens": 64},
    })]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    with pytest.raises(LLMError):
        await chat([LLMConfig("https://small.example/v1", "model")],
                   "system", "question")

    assert "reasoning_present=True" in caplog.text
    assert "reasoning_len=13" in caplog.text
    assert "private chain" not in caplog.text


@pytest.mark.asyncio
async def test_answer_text_not_logged_by_default(monkeypatch, caplog):
    """Streaming alone must NOT write visible answer text to logs.

    Answers can quote private repository evidence; logging them requires the
    separate, explicitly named OI_LLM_LOG_ANSWERS diagnostic switch.
    """
    from oi_agent.agent import llm

    caplog.set_level("INFO")
    StreamingClient.calls = []
    StreamingClient.lines = [
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "secret evidence"}}],
        }),
        "data: [DONE]",
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", StreamingClient)
    monkeypatch.setenv("OI_LLM_STREAM_LOGS", "1")
    monkeypatch.delenv("OI_LLM_LOG_ANSWERS", raising=False)

    text, _ = await chat([
        LLMConfig("https://small.example/v1", "model")
    ], "system", "question")

    assert text == "secret evidence"  # still delivered to the caller
    assert "secret evidence" not in caplog.text  # never written to logs


@pytest.mark.asyncio
async def test_answer_logging_is_explicit_opt_in(monkeypatch, caplog):
    """OI_LLM_LOG_ANSWERS=1 is the only switch that logs answer text."""
    from oi_agent.agent import llm

    caplog.set_level("INFO")
    StreamingClient.calls = []
    StreamingClient.lines = [
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "visible "}}],
        }),
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "answer"}}],
        }),
        "data: [DONE]",
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", StreamingClient)
    monkeypatch.setenv("OI_LLM_STREAM_LOGS", "1")
    monkeypatch.setenv("OI_LLM_LOG_ANSWERS", "1")

    text, _ = await chat([
        LLMConfig("https://small.example/v1", "model")
    ], "system", "question")

    assert text == "visible answer"
    assert StreamingClient.calls[0][2]["stream"] is True
    assert "[answer] visible " in caplog.text
    assert "[answer] answer" in caplog.text


@pytest.mark.asyncio
async def test_reasoning_content_never_logged_metadata_only(monkeypatch,
                                                           caplog):
    """Privacy contract: hidden chain-of-thought NEVER reaches any log sink.

    The old behavior logged ``[thinking] <text>`` under OI_LLM_STREAM_LOGS;
    now only non-sensitive metadata (observed char count) may appear.
    """
    from oi_agent.agent import llm

    caplog.set_level("INFO")
    StreamingClient.calls = []
    StreamingClient.lines = [
        "data: " + json.dumps({
            "choices": [{"delta": {"reasoning_content": "let me think"}}],
        }),
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "final answer"}}],
        }),
        "data: [DONE]",
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", StreamingClient)
    monkeypatch.setenv("OI_LLM_STREAM_LOGS", "1")
    monkeypatch.delenv("OI_LLM_LOG_ANSWERS", raising=False)

    text, _ = await chat([
        LLMConfig("https://small.example/v1", "model")
    ], "system", "question")

    assert text == "final answer"
    assert "let me think" not in caplog.text      # content never logged
    assert "[thinking]" not in caplog.text        # old leak tag gone
    assert "content not logged" in caplog.text    # metadata line present
    assert "12 chars" in caplog.text              # len("let me think")


def _openai_tool_response(call_id: str = "call_1",
                          arguments: str = '{"patterns": ["ContractPosition"]}'
                          ) -> FakeResponse:
    """Build one openai-style assistant turn requesting a single tool call.

    Args:
        call_id: Tool-call id the provider assigns.
        arguments: Raw JSON arguments string exactly as providers send it.

    Returns:
        FakeResponse: Scripted tool-call turn.
    """
    return FakeResponse(200, {"choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "grep_repo", "arguments": arguments},
        }],
    }}]})


def _openai_text_response(text: str) -> FakeResponse:
    """Build one openai-style plain-text assistant turn.

    Args:
        text: Final visible answer.

    Returns:
        FakeResponse: Scripted text turn.
    """
    return FakeResponse(200, {
        "choices": [{"message": {"role": "assistant", "content": text}}],
    })


def _anthropic_tool_response() -> FakeResponse:
    """Build one anthropic-style turn mixing text and a tool_use block."""
    return FakeResponse(200, {"content": [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "tu_1", "name": "read_file",
         "input": {"path": "src/x.py"}},
    ]})


@pytest.mark.asyncio
async def test_openai_tool_loop_feeds_results_and_answers(monkeypatch):
    """A tool_call turn runs the executor and feeds results back as role=tool."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        _openai_tool_response(),
        _openai_text_response("found it"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    seen = []

    def execute(name, args):
        seen.append((name, args))
        return "src/x.py:1: class ContractPosition"

    cfg = LLMConfig("http://p1", "m1", "K")
    text, provenance = await llm.chat_agentic(
        [cfg], "sys", "question", tools=[{"fake": "schema"}],
        execute=execute, max_iterations=4, max_tokens=100,
    )

    assert text == "found it"
    assert seen == [("grep_repo", {"patterns": ["ContractPosition"]})]
    first, second = FakeClient.calls[0][2], FakeClient.calls[1][2]
    assert first["tools"] == [{"fake": "schema"}]
    tool_msgs = [m for m in second["messages"] if m.get("role") == "tool"]
    assert tool_msgs == [{
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "src/x.py:1: class ContractPosition",
    }]
    assert provenance == "http://p1/m1"


@pytest.mark.asyncio
async def test_anthropic_tool_loop_uses_one_tool_result_block(monkeypatch):
    """Anthropic results land in ONE user message of tool_result blocks."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        _anthropic_tool_response(),
        FakeResponse(200, {"content": [
            {"type": "text", "text": "done reading"},
        ]}),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("ZKEY", "z-secret")
    cfg = LLMConfig("https://api.z.ai", "glm-4.6", "ZKEY", "anthropic")

    text, _ = await llm.chat_agentic(
        [cfg], "sys", "question",
        tools=[{"type": "function", "function": {
            "name": "read_file", "description": "read one file",
            "parameters": {"type": "object"},
        }}],
        execute=lambda name, args: "file body", max_iterations=4,
        max_tokens=100,
    )

    assert text == "done reading"
    second = FakeClient.calls[1][2]
    # System prompt carries the base prompt plus the tool-round budget preamble.
    assert second["system"].startswith("sys")
    assert "TOOL BUDGET" in second["system"]
    assistant_turns = [m for m in second["messages"]
                       if m["role"] == "assistant"]
    assert FakeClient.calls[1][2]["tools"] == [{
        "name": "read_file",
        "description": "read one file",
        "input_schema": {"type": "object"},
    }]
    assert assistant_turns[-1]["content"][1]["type"] == "tool_use"
    result_msgs = [m for m in second["messages"] if m["role"] == "user"
                   and isinstance(m["content"], list)]
    # One user turn: the tool_result block followed by the budget countdown text.
    assert len(result_msgs) == 1
    blocks = result_msgs[0]["content"]
    assert blocks[0] == {"type": "tool_result", "tool_use_id": "tu_1",
                         "content": "file body"}
    assert blocks[1]["type"] == "text"
    assert "round 1 of 4" in blocks[1]["text"]


@pytest.mark.asyncio
async def test_iteration_cap_forces_final_no_tools_call(monkeypatch):
    """Exhausting the budget triggers exactly one tools-free final request."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    # Two DISTINCT calls so the loop guard (exact-repeat detector) does not
    # fire; this test exercises the plain cap-exhaustion path.
    FakeClient.responses = [
        _openai_tool_response(
            call_id="c1", arguments='{"patterns": ["alpha"]}'),
        _openai_tool_response(
            call_id="c2", arguments='{"patterns": ["beta"]}'),
        _openai_text_response("giving my best answer"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    rounds = []

    def execute(name, args):
        rounds.append(name)
        return "evidence"

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=execute, max_iterations=2, max_tokens=50,
    )

    assert text == "giving my best answer"
    assert len(rounds) == 2
    assert len(FakeClient.calls) == 3
    assert "tools" not in FakeClient.calls[2][2]


@pytest.mark.asyncio
async def test_exact_repeat_is_not_re_executed(monkeypatch):
    """An identical repeated call is skipped and answered with a pointer."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    # Round 1 asks for a call; round 2 asks for the SAME call; then answers.
    FakeClient.responses = [
        _openai_tool_response(call_id="c1"),
        _openai_tool_response(call_id="c2"),  # identical arguments
        _openai_text_response("answer"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    runs = []

    def execute(name, args):
        runs.append(args)
        return "the file body"

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=execute, max_iterations=8, max_tokens=50,
    )

    assert text == "answer"
    # Executor ran only ONCE despite two identical requests.
    assert len(runs) == 1
    # The second (latest) tool message fed back a Memory Pointer, not a body.
    tool_msgs = [m for m in FakeClient.calls[2][2]["messages"]
                 if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "already ran in round" in tool_msgs[-1]["content"]
    assert "the file body" not in tool_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_repeated_calls_break_loop_early(monkeypatch):
    """Two stale (all-repeat) rounds end the loop before the cap and answer."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    # Three identical tool rounds (round 1 novel, rounds 2 & 3 stale) trip the
    # guard after round 3; the 4th response is the forced-final text answer.
    # Without the guard this would run to max_iterations=20.
    FakeClient.responses = [
        _openai_tool_response(call_id="c0"),
        _openai_tool_response(call_id="c1"),
        _openai_tool_response(call_id="c2"),
        _openai_text_response("forced answer"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    runs = []

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=lambda n, a: runs.append(a) or "body",
        max_iterations=20, max_tokens=50,
    )

    assert text == "forced answer"
    # Broke early: round 1 (novel) + rounds 2 & 3 (stale) then forced final.
    # That is 3 tool rounds + 1 no-tools call = 4 requests, far below 20.
    assert len(FakeClient.calls) == 4
    assert "tools" not in FakeClient.calls[-1][2]
    # Executor ran once; the repeats never reached it.
    assert len(runs) == 1
    final_msg = FakeClient.calls[-1][2]["messages"][-1]
    assert "repeating" in final_msg["content"].lower()


@pytest.mark.asyncio
async def test_distinct_calls_do_not_trip_loop_guard(monkeypatch):
    """Genuinely different calls each execute and never trigger the guard."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        _openai_tool_response(
            call_id="c1", arguments='{"patterns": ["authentication_service"]}'),
        _openai_tool_response(
            call_id="c2", arguments='{"patterns": ["billing_repository"]}'),
        _openai_tool_response(
            call_id="c3", arguments='{"patterns": ["invoice_generator"]}'),
        _openai_text_response("answer from three sources"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    runs = []

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=lambda n, a: runs.append(a) or "body",
        max_iterations=8, max_tokens=50,
    )

    assert text == "answer from three sources"
    # All three distinct calls executed; no early break.
    assert len(runs) == 3
    assert "tools" in FakeClient.calls[0][2]
    assert "tools" in FakeClient.calls[1][2]


@pytest.mark.asyncio
async def test_budget_stated_up_front_and_counted_down(monkeypatch):
    """The model is told its total budget and gets a countdown per round."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        _openai_tool_response(),
        _openai_text_response("answer"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=lambda _n, _a: "evidence",
        max_iterations=7, max_tokens=50,
    )

    # First request: system prompt states the total budget.
    system_msg = next(m["content"] for m in FakeClient.calls[0][2]["messages"]
                      if m["role"] == "system")
    assert "7 tool-call round" in system_msg
    # Second request: the countdown note rode back with the tool results.
    note = FakeClient.calls[1][2]["messages"][-1]
    assert note["role"] == "user"
    assert "round 1 of 7" in note["content"]
    assert "6 remaining" in note["content"]


@pytest.mark.asyncio
async def test_final_turn_carries_answer_now_instruction(monkeypatch):
    """The forced no-tools call is preceded by an explicit final-turn order."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        _openai_tool_response(),
        _openai_text_response("final answer from evidence"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=lambda _n, _a: "evidence",
        max_iterations=1, max_tokens=50,
    )

    # max_iterations=1: one tool round, then the forced final no-tools call.
    assert text == "final answer from evidence"
    assert "tools" not in FakeClient.calls[1][2]
    final_user = FakeClient.calls[1][2]["messages"][-1]
    assert final_user["role"] == "user"
    assert "final turn" in final_user["content"].lower()
    assert "no tools" in final_user["content"].lower()


@pytest.mark.asyncio
async def test_plain_text_first_turn_skips_the_loop(monkeypatch):
    """A direct answer costs one request and never touches the executor."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [_openai_text_response("straight away")]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    def fail(_name, _args):
        raise AssertionError("executor must not run")

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=fail, max_iterations=3, max_tokens=50,
    )

    assert text == "straight away"
    assert len(FakeClient.calls) == 1


@pytest.mark.asyncio
async def test_executor_exception_becomes_error_string(monkeypatch):
    """A crashing callback feeds ERROR text back instead of killing the loop."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        _openai_tool_response(),
        _openai_text_response("ok then"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    def boom(_name, _args):
        raise RuntimeError("boom")

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}], execute=boom, max_iterations=4, max_tokens=50,
    )

    assert text == "ok then"
    tool_msg = next(m for m in FakeClient.calls[1][2]["messages"]
                    if m.get("role") == "tool")
    assert "ERROR" in tool_msg["content"]
    assert "boom" in tool_msg["content"]


@pytest.mark.asyncio
async def test_malformed_arguments_json_fed_back(monkeypatch):
    """Unparseable openai arguments become an error result, not a crash."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        _openai_tool_response(arguments="{not json"),
        _openai_text_response("recovered"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    text, _ = await llm.chat_agentic(
        [LLMConfig("http://p1", "m1")], "sys", "q",
        tools=[{"t": 1}],
        execute=lambda _n, _a: "unused", max_iterations=4, max_tokens=50,
    )

    assert text == "recovered"
    tool_msg = next(m for m in FakeClient.calls[1][2]["messages"]
                    if m.get("role") == "tool")
    assert tool_msg["content"].startswith("ERROR: invalid arguments JSON")


@pytest.mark.asyncio
async def test_mid_loop_failure_falls_through_to_next_provider(monkeypatch):
    """A provider dying mid-loop hands the whole session to the next one."""
    from oi_agent.agent import llm

    FakeClient.calls = []
    FakeClient.responses = [
        FakeResponse(500, {"error": "boom"}),
        _openai_tool_response(),
        _openai_text_response("from fallback"),
    ]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    big = LLMConfig("https://api.z.ai", "glm-4.6", "K", "anthropic")
    small = LLMConfig("http://fallback", "small")

    text, provenance = await llm.chat_agentic(
        [big, small], "sys", "q", tools=[{"t": 1}],
        execute=lambda _n, _a: "ev", max_iterations=4, max_tokens=50,
    )

    assert text == "from fallback"
    assert provenance == "http://fallback/small"


@pytest.mark.asyncio
async def test_agentic_failure_keeps_empty_exception_type(monkeypatch, caplog):
    """An empty exception message still identifies the provider failure type."""
    from oi_agent.agent import llm

    class EmptyFailureResponse:
        """Response whose status check mimics an empty-message timeout."""

        def raise_for_status(self):
            """Raise without a message, as several httpx exceptions do."""
            raise TimeoutError()

    FakeClient.calls = []
    FakeClient.responses = [EmptyFailureResponse()]
    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)

    with pytest.raises(LLMError, match="TimeoutError"):
        await llm.chat_agentic(
            [LLMConfig("http://p1", "m1")], "sys", "q",
            tools=[{"t": 1}], execute=lambda _n, _a: "unused",
            max_iterations=1, max_tokens=50,
        )

    assert "trying next provider: TimeoutError:" in caplog.text


def test_fingerprint_ignores_key_order():
    """Reordered argument keys produce the same exact-repeat fingerprint."""
    from oi_agent.agent.llm import _fingerprint

    a = _fingerprint("grep_repo", {"patterns": ["x"], "max_results": 5})
    b = _fingerprint("grep_repo", {"max_results": 5, "patterns": ["x"]})
    c = _fingerprint("read_file", {"patterns": ["x"], "max_results": 5})
    assert a == b       # key order is normalized
    assert a != c       # tool name is part of the key


def test_call_tracker_exact_repeat_and_novel():
    """The tracker flags an identical second call and passes distinct ones."""
    from oi_agent.agent.llm import _CallTracker

    t = _CallTracker()
    assert t.classify("grep_repo", {"patterns": ["a"]}, 1)[0] == "novel"
    kind, detail = t.classify("grep_repo", {"patterns": ["a"]}, 2)
    assert kind == "repeat"
    assert "round 1" in detail
    assert t.classify("read_file", {"path": "totally/other.py"}, 3)[0] \
        == "novel"


def test_call_tracker_near_duplicate():
    """Cosmetically different arguments are flagged as near-duplicates."""
    from oi_agent.agent.llm import _CallTracker

    t = _CallTracker()
    t.classify("grep_repo", {"patterns": ["get_position_summary"]}, 1)
    kind, detail = t.classify(
        "grep_repo", {"patterns": ["get_position_summarx"]}, 2)
    assert kind == "near"
    assert "round 1" in detail


def test_call_tracker_stale_round_break():
    """Two consecutive all-repeat rounds report the loop as stuck."""
    from oi_agent.agent.llm import _CallTracker

    t = _CallTracker()
    assert t.note_round(["novel"]) is False
    assert t.note_round(["repeat"]) is False   # stale_rounds == 1
    assert t.note_round(["repeat"]) is True    # stale_rounds == 2 -> break


def test_call_tracker_novel_resets_stale_counter():
    """A novel call between repeats resets the stale-round streak."""
    from oi_agent.agent.llm import _CallTracker

    t = _CallTracker()
    assert t.note_round(["repeat"]) is False
    assert t.note_round(["novel"]) is False    # reset
    assert t.note_round(["repeat"]) is False   # only 1 again, no break
