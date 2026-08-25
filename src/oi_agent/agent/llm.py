"""Small multi-protocol LLM client with ordered provider fallback.

OpenAI-compatible providers use `/chat/completions`; Z.AI can use the
Anthropic-compatible `/api/anthropic/v1/messages` protocol.
Each endpoint selects its own wire format, and failures fall through to the
next configured provider.
"""

import difflib
import hashlib
import json
import logging
import os
from collections.abc import Callable

import httpx

from ..config import LLMConfig

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(600.0)


class LLMError(Exception):
    """Raised when all configured providers fail."""


def _log_wire(endpoint: str, payload: dict, raw_response: str) -> None:
    """Log the exact outgoing request and raw response for one call.

    Diagnostic-only (gated by ``OI_LLM_DEBUG_WIRE``). System/user message
    bodies are truncated to keep logs readable; the goal is to see whether a
    request field like ``chat_template_kwargs`` was actually sent and how the
    provider replied verbatim. This does expose the full response body, so use
    it only against non-sensitive debugging traffic.

    Args:
        endpoint: Provider endpoint the request went to.
        payload: The JSON body that was sent.
        raw_response: The provider's raw response text.
    """
    sent = dict(payload)
    if isinstance(sent.get("messages"), list):
        sent["messages"] = [
            {**m, "content": str(m.get("content", ""))[:120] + "…"}
            for m in sent["messages"] if isinstance(m, dict)
        ]
    logger.warning("[llm][wire] endpoint=%s request=%s", endpoint,
                   json.dumps(sent, ensure_ascii=False))
    logger.warning("[llm][wire] raw_response=%s", raw_response[:4_000])


def _log_invalid_payload(
    endpoint: str,
    data,
    message=None,
    choice=None,
) -> None:
    """Log response shape metadata without exposing hidden reasoning text.

    Args:
        endpoint: Provider endpoint.
        data: Decoded provider payload.
        message: OpenAI-style message object, if present.
        choice: OpenAI-style choice object, if present.
    """
    content = message.get("content") if isinstance(message, dict) else None
    reasoning = None
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content", message.get("reasoning"))
    logger.warning(
        "[llm] unusable response endpoint=%s top_keys=%s message_keys=%s "
        "content_type=%s content_len=%s reasoning_present=%s reasoning_len=%s "
        "finish_reason=%s usage_keys=%s",
        endpoint,
        sorted(data) if isinstance(data, dict) else type(data).__name__,
        sorted(message) if isinstance(message, dict) else None,
        type(content).__name__,
        len(content) if isinstance(content, str) else None,
        bool(reasoning),
        len(reasoning) if isinstance(reasoning, str) else None,
        choice.get("finish_reason") if isinstance(choice, dict) else None,
        sorted(data.get("usage", {})) if isinstance(data, dict)
        and isinstance(data.get("usage"), dict) else None,
    )


async def _stream_text(response, api_style: str, model: str) -> str:
    """Parse provider SSE events and return only the visible answer text.

    Hidden reasoning content is NEVER logged — it is counted (one metadata
    line per stream) and discarded. Visible answer deltas are logged only
    when the operator explicitly sets ``OI_LLM_LOG_ANSWERS=1``, because
    answers can quote private repository evidence; they are a separate,
    clearly named diagnostic from the streaming switch.

    Args:
        response: OpenAI or Anthropic-compatible streaming response.
        api_style: Provider wire style.
        model: Model label for operational logs.

    Returns:
        Concatenated visible answer text (reasoning excluded).
    """
    chunks: list[str] = []
    log_answers = os.environ.get("OI_LLM_LOG_ANSWERS", "").lower() in {
        "1", "true", "yes", "on",
    }
    reasoning_chars = 0
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if api_style == "anthropic":
            event_delta = event.get("delta") or {}
            delta = event_delta.get("text", "")
            reasoning = event_delta.get("thinking", "")
        else:
            choices = event.get("choices") or [{}]
            event_delta = (choices[0].get("delta") or {}
                           if isinstance(choices[0], dict) else {})
            delta = event_delta.get("content", "")
            reasoning = event_delta.get(
                "reasoning_content", event_delta.get("reasoning", ""))
        if isinstance(reasoning, str) and reasoning:
            # Privacy contract: metadata about hidden reasoning is fine,
            # its CONTENT is never written to any log sink.
            reasoning_chars += len(reasoning)
        if isinstance(delta, str) and delta:
            chunks.append(delta)
            if log_answers:
                logger.info("[llm][%s][answer] %s", model, delta)
    if reasoning_chars:
        logger.info("[llm][%s] hidden reasoning observed (%d chars, "
                    "content not logged)", model, reasoning_chars)
    return "".join(chunks)



async def chat(
    configs: list[LLMConfig],
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> tuple[str, str]:
    """Send one completion with ordered fallback and optional answer streaming.

    Set ``OI_LLM_STREAM_LOGS=1`` to request SSE streaming (lower time-to-first
    token). Hidden reasoning fields are never logged and never used as answer
    text; visible answer deltas are logged only under ``OI_LLM_LOG_ANSWERS=1``.

    Args:
        configs: Ordered provider chain; first success wins.
        system: System prompt.
        user: User message.
        max_tokens: Completion budget.
        temperature: Sampling temperature.

    Returns:
        Tuple of (reply text, ``base_url/model`` that produced it).

    Raises:
        LLMError: When every configured provider fails or none is configured.
    """
    errors: list[str] = []
    stream_logs = os.environ.get("OI_LLM_STREAM_LOGS", "").lower() in {
        "1", "true", "yes", "on",
    }
    debug_wire = os.environ.get("OI_LLM_DEBUG_WIRE", "").lower() in {
        "1", "true", "yes", "on",
    }
    for cfg in configs:
        if not cfg.base_url or not cfg.model:
            errors.append(f"{cfg.model or cfg.base_url}: not configured")
            continue
        key = os.environ.get(cfg.api_key_env, "")
        endpoint = ""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                if cfg.api_style == "anthropic":
                    endpoint = (
                        f"{cfg.base_url.rstrip('/')}"
                        "/api/anthropic/v1/messages"
                    )
                    headers = {
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": cfg.model,
                        "max_tokens": max_tokens,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                        "temperature": temperature,
                    }
                    if stream_logs:
                        payload["stream"] = True
                        async with client.stream(
                            "POST", endpoint, headers=headers, json=payload
                        ) as resp:
                            resp.raise_for_status()
                            text = await _stream_text(
                                resp, cfg.api_style, cfg.model
                            )
                        data = {}
                    else:
                        resp = await client.post(
                            endpoint, headers=headers, json=payload
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        blocks = data.get("content", [])
                        text = "".join(
                            block.get("text", "")
                            for block in blocks
                            if isinstance(block, dict)
                        )
                else:
                    endpoint = (
                        f"{cfg.base_url.rstrip('/')}/chat/completions"
                    )
                    headers = (
                        {"Authorization": f"Bearer {key}"} if key else {}
                    )
                    payload = {
                        "model": cfg.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if cfg.disable_thinking:
                        # vLLM/Qwen lever: skip the hidden reasoning pass so the
                        # answer lands in `content` instead of being truncated
                        # inside the reasoning field.
                        payload["chat_template_kwargs"] = {
                            "enable_thinking": False
                        }
                    if stream_logs:
                        payload["stream"] = True
                        async with client.stream(
                            "POST", endpoint, headers=headers, json=payload
                        ) as resp:
                            resp.raise_for_status()
                            text = await _stream_text(
                                resp, cfg.api_style, cfg.model
                            )
                        data = {}
                    else:
                        resp = await client.post(
                            endpoint, headers=headers, json=payload
                        )
                        resp.raise_for_status()
                        if debug_wire:
                            _log_wire(endpoint, payload, resp.text)
                        data = resp.json()
                        choices = data.get("choices") or []
                        choice = choices[0] if choices else None
                        message = (
                            choice.get("message")
                            if isinstance(choice, dict)
                            else None
                        )
                        text = (
                            message.get("content")
                            if isinstance(message, dict)
                            else None
                        )
                        if not isinstance(text, str) or not text.strip():
                            _log_invalid_payload(
                                endpoint, data, message, choice
                            )
                if not isinstance(text, str) or not text.strip():
                    raise LLMError("provider returned non-text content")
                logger.info("[llm] %s/%s succeeded via %s",
                            cfg.base_url, cfg.model, cfg.api_style)
                return text, f"{cfg.base_url}/{cfg.model}"
        except Exception as exc:  # noqa: BLE001 - provider boundary
            label = endpoint or f"{cfg.base_url}/{cfg.model}"
            detail = f"{type(exc).__name__}: {exc}"
            logger.warning("[llm] %s failed; trying next provider: %s",
                           label, detail)
            errors.append(f"{label}: {detail}")
    raise LLMError("; ".join(errors) or "no providers configured")


def _parse_round(api_style: str, data: dict) -> tuple[str, list[dict], dict]:
    """Extract visible text and tool-call requests from one response.

    Normalizes both wire styles into one shape so the agentic loop stays
    provider-agnostic. A turn may legally carry empty text when it only
    requests tool calls.

    Args:
        api_style: ``"openai"`` or ``"anthropic"`` wire format selector.
        data: Parsed JSON response body.

    Returns:
        tuple[str, list[dict], dict]: Visible text ("" for tool-only turns),
        normalized calls (keys id/name/arguments/parse_error), and the raw
        assistant message to append back onto the running history.
    """
    if api_style == "anthropic":
        blocks = data.get("content", [])
        if not isinstance(blocks, list):
            blocks = []
        text = "".join(
            block.get("text", "") for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        calls = [
            {
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": (
                    block["input"] if isinstance(block.get("input"), dict)
                    else {}
                ),
                "parse_error": None,
            }
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        return text, calls, {"role": "assistant", "content": blocks}
    choices = data.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    message = message if isinstance(message, dict) else {}
    text = message.get("content")
    text = text if isinstance(text, str) else ""
    calls = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        call = {
            "id": raw.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": {},
            "parse_error": None,
        }
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            call["arguments"] = args
        except (ValueError, TypeError) as exc:
            call["parse_error"] = f"invalid arguments JSON: {exc}"
        calls.append(call)
    return text, calls, message


def _run_calls(
    calls: list[dict], execute: Callable[[str, dict], str],
    tracker: "_CallTracker", round_no: int,
) -> tuple[list[tuple[dict, str]], list[str]]:
    """Execute requested tool calls with exact/near-duplicate loop guarding.

    Exact repeats are NOT re-executed: a Memory Pointer reference is returned so
    the same large body never re-enters context (the source of window overflow).
    Near-duplicates run once but carry an advisory. Novel calls run normally.

    Args:
        calls: Normalized tool-call dicts from _parse_round.
        execute: Injected sync callback running one call by (name, arguments).
        tracker: Per-audit repeat detector.
        round_no: Current 1-based loop round (for pointer/advisory wording).

    Returns:
        tuple: ((call, textual result) pairs in request order, per-call
        classification strings for stale-round accounting).
    """
    pairs = []
    classes: list[str] = []
    for call in calls:
        if call["parse_error"]:
            result = f"ERROR: {call['parse_error']}"
            classes.append("novel")
            pairs.append((call, result))
            _log_tool(call, result)
            continue
        kind, detail = tracker.classify(
            call["name"], call["arguments"], round_no)
        classes.append(kind)
        if kind == "repeat":
            # Memory Pointer: skip execution, feed back the reference only.
            pairs.append((call, detail))
            logger.info("[llm][tool] %s REPEAT (round of first use noted); "
                        "skipped", call["name"] or "?")
            continue
        try:
            out = execute(call["name"], call["arguments"])
            result = out if isinstance(out, str) else str(out)
        except Exception as exc:  # noqa: BLE001 - callback boundary
            result = f"ERROR: {type(exc).__name__}: {exc}"
        if kind == "near" and detail:
            result = result + detail
        pairs.append((call, result))
        _log_tool(call, result)
    return pairs, classes


def _log_tool(call: dict, result: str) -> None:
    """Log one tool call's name, arguments, and result size.

    Tool name/arguments are non-sensitive (regex patterns, repo-relative
    paths); the result body may quote repo evidence, so only its length is
    logged. Always on: 24/7 audits need visible retrieval activity.

    Args:
        call: Normalized tool-call dict.
        result: Textual result fed back to the model.
    """
    logger.info(
        "[llm][tool] %s args=%s -> %d chars",
        call["name"] or "?",
        json.dumps(call["arguments"], ensure_ascii=False),
        len(result),
    )


# Loop-breaking thresholds. Exact repeats are refused outright; near-duplicates
# (cosmetic argument variation) get one advisory before counting as stale; two
# fully stale rounds end the loop early and force a final answer. These mirror
# production harness defaults (opencode-anti-loop, LangChain
# max_sequential_tool_calls) scaled down for a 2-tool read-only auditor.
_NEAR_DUP_RATIO = 0.9
_MAX_STALE_ROUNDS = 2

_REPEAT_POINTER = (
    "NOTE: this identical call already ran in round {round}; its result is "
    "unchanged and is omitted here to save context. Do not repeat it — use the "
    "evidence you already have or try a genuinely different query."
)
_NEAR_DUP_ADVISORY = (
    "\n\n[loop guard] This call closely resembles one from round {round}. "
    "Make sure you are gathering NEW evidence, not rephrasing the same query."
)
_STUCK_PREFACE = (
    "You are repeating tool calls without making progress, so tool access is "
    "now closed. Answer the question using the evidence already gathered; if it "
    "is insufficient, say so honestly. Do not request tools."
)


def _fingerprint(name: str, arguments: dict) -> str:
    """Hash a tool call into a stable exact-repeat key.

    Argument key order is normalized (``sort_keys``) so the same call written
    with reordered keys still collides — exactly the shape providers vary.

    Args:
        name: Tool name.
        arguments: Tool arguments dict.

    Returns:
        Hex digest identifying this (name, arguments) pair.
    """
    blob = name + "\x00" + json.dumps(
        arguments, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


def _arg_signature(name: str, arguments: dict) -> str:
    """Build the flat string used for fuzzy near-duplicate comparison.

    Args:
        name: Tool name (prefixed so cross-tool calls never fuzzy-match).
        arguments: Tool arguments dict.

    Returns:
        A normalized signature string.
    """
    return name + "|" + json.dumps(
        arguments, sort_keys=True, ensure_ascii=False, default=str).lower()


class _CallTracker:
    """Detects repeated / near-duplicate tool calls within one audit.

    Exact repeats are keyed by fingerprint and never re-executed (a Memory
    Pointer reference is returned instead, bounding context growth). Near
    duplicates are executed once but flagged. The caller uses ``note_round`` to
    decide when the model is stuck and the loop should break early.
    """

    def __init__(self) -> None:
        """Initialize empty repeat/near-duplicate state."""
        # fingerprint -> 1-based round it was first seen
        self._seen: dict[str, int] = {}
        # (signature, round) history for fuzzy comparison
        self._signatures: list[tuple[str, int]] = []
        self.stale_rounds = 0

    def classify(self, name: str, arguments: dict, round_no: int
                 ) -> tuple[str, str | None]:
        """Classify one call as exact-repeat, near-duplicate, or novel.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.
            round_no: Current 1-based loop round.

        Returns:
            tuple[str, str | None]: ("repeat"|"near"|"novel", detail). For a
            repeat, detail is the Memory Pointer string to return instead of
            executing. For a near-duplicate, detail is an advisory to append to
            the real result. For novel, detail is None.
        """
        fp = _fingerprint(name, arguments)
        if fp in self._seen:
            return "repeat", _REPEAT_POINTER.format(round=self._seen[fp])
        sig = _arg_signature(name, arguments)
        near_round = self._nearest(sig)
        self._seen[fp] = round_no
        self._signatures.append((sig, round_no))
        if near_round is not None:
            return "near", _NEAR_DUP_ADVISORY.format(round=near_round)
        return "novel", None

    def _nearest(self, sig: str) -> int | None:
        """Return the round of the closest prior signature above the ratio.

        Args:
            sig: Signature of the call being classified.

        Returns:
            The matching round number, or None if nothing is close enough.
        """
        matcher = difflib.SequenceMatcher(None, sig, "")
        for prev_sig, prev_round in reversed(self._signatures):
            matcher.set_seq2(prev_sig)
            if matcher.ratio() >= _NEAR_DUP_RATIO:
                return prev_round
        return None

    def note_round(self, classes: list[str]) -> bool:
        """Update the stale-round counter after a round and report stuck-ness.

        A round is stale when it produced no novel evidence — every call was an
        exact repeat. Two consecutive stale rounds means the model is looping.

        Args:
            classes: Per-call classifications from this round.

        Returns:
            True when the loop should break early and force a final answer.
        """
        if classes and all(c == "repeat" for c in classes):
            self.stale_rounds += 1
        else:
            self.stale_rounds = 0
        return self.stale_rounds >= _MAX_STALE_ROUNDS


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert openai function schemas into anthropic input_schema form.

    Fallback chains routinely mix wire styles (e.g. a z.ai anthropic
    endpoint plus an openai-compatible fallback), so callers pass ONE
    canonical openai-shaped list and each provider's dialect is derived
    here. Entries already in anthropic shape pass through untouched.

    Args:
        tools: Schemas as {"type": "function", "function": {...}} entries.

    Returns:
        list[dict]: [{"name", "description", "input_schema"}] entries.
    """
    out = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict):
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
        else:
            out.append(tool)
    return out


_BUDGET_FINAL = (
    "This is your final turn. You have no tools left and cannot gather more "
    "evidence. Answer the question now using only the evidence already "
    "gathered; if it is insufficient, say so honestly. Do not request tools."
)


def _budget_preamble(max_iterations: int) -> str:
    """Build the system-prompt sentence that states the tool-round budget.

    The model plans its retrieval far better when it knows the budget up front
    rather than discovering the limit only when the harness forces a final,
    tool-less turn.

    Args:
        max_iterations: Total tool-call rounds available before the forced
            final answer.

    Returns:
        A leading-newline sentence to append onto the system prompt.
    """
    return (
        f"\n\nTOOL BUDGET: you have {max_iterations} tool-call round(s) to "
        "gather evidence before you MUST answer. Each turn where you request "
        "tools spends one round. Budget them: search broadly first, read only "
        "the files that matter, and leave yourself a round to write the answer."
    )


def _budget_note(rounds_used: int, max_iterations: int) -> str:
    """Build the per-round countdown appended after each tool-result batch.

    Args:
        rounds_used: Rounds consumed so far (1-based, includes this turn).
        max_iterations: Total tool-call rounds available.

    Returns:
        A short status line the model reads to track its remaining budget.
    """
    remaining = max(0, max_iterations - rounds_used)
    if remaining == 0:
        return (
            f"Tool round {rounds_used} of {max_iterations} used; 0 remaining. "
            "Your next turn must be the final answer — no more tools."
        )
    return (
        f"Tool round {rounds_used} of {max_iterations} used; {remaining} "
        "remaining. Continue only if you still need evidence, otherwise answer."
    )


async def chat_agentic(
    configs: list[LLMConfig],
    system: str,
    user: str,
    *,
    tools: list[dict],
    execute: Callable[[str, dict], str],
    max_iterations: int,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> tuple[str, str]:
    """Run a bounded tool loop with ordered provider fallback.

    The model drives retrieval by requesting tool calls; every request runs
    through the injected ``execute`` callback and its string result feeds
    back until the model answers in plain text. Each assistant turn asking
    for at least one call consumes one iteration; once the cap is hit a
    single final request WITHOUT tools forces an answer from what was already
    gathered. Streaming is deliberately unused inside the loop: tool-call
    deltas would add parser complexity for no product value.

    Args:
        configs: Ordered provider chain; first success wins.
        system: System prompt.
        user: Initial user message.
        tools: OpenAI-shaped tool schemas ({"type": "function", ...}); the
            anthropic dialect is derived per provider, since fallback chains
            routinely mix wire styles.
        execute: Sync callback executing one tool call by (name, arguments).
        max_iterations: Cap on model-driven tool rounds before the forced
            final no-tools request.
        max_tokens: Completion budget per request.
        temperature: Sampling temperature.

    Returns:
        Tuple of (final reply text, ``base_url/model`` that produced it).

    Raises:
        LLMError: When every configured provider fails or none is configured.
    """
    errors: list[str] = []
    debug_wire = os.environ.get("OI_LLM_DEBUG_WIRE", "").lower() in {
        "1", "true", "yes", "on",
    }
    for cfg in configs:
        if not cfg.base_url or not cfg.model:
            errors.append(f"{cfg.model or cfg.base_url}: not configured")
            continue
        key = os.environ.get(cfg.api_key_env, "")
        endpoint = ""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                anthropic = cfg.api_style == "anthropic"
                # Budget-awareness: the model plans poorly when it does not know
                # its tool-round budget, then wastes the forced final turn. State
                # the budget up front, count down per round, and demand an answer
                # on the last turn. See _BUDGET_* helpers.
                budget_system = system + _budget_preamble(max_iterations)
                if anthropic:
                    endpoint = (
                        f"{cfg.base_url.rstrip('/')}"
                        "/api/anthropic/v1/messages"
                    )
                    headers = {
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }
                    messages: list[dict] = [{"role": "user", "content": user}]
                else:
                    endpoint = (
                        f"{cfg.base_url.rstrip('/')}/chat/completions"
                    )
                    headers = (
                        {"Authorization": f"Bearer {key}"} if key else {}
                    )
                    messages = [
                        {"role": "system", "content": budget_system},
                        {"role": "user", "content": user},
                    ]

                def payload_for(include_tools: bool) -> dict:
                    body = {
                        "model": cfg.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if anthropic:
                        body["system"] = budget_system
                    elif cfg.disable_thinking:
                        # Same vLLM/Qwen lever as chat(): keep answers in
                        # `content` instead of truncating into reasoning.
                        body["chat_template_kwargs"] = {
                            "enable_thinking": False
                        }
                    if include_tools:
                        body["tools"] = (
                            _anthropic_tools(tools) if anthropic else tools
                        )
                    return body

                async def post_round(include_tools: bool) -> dict:
                    sent = payload_for(include_tools)
                    resp = await client.post(
                        endpoint, headers=headers, json=sent
                    )
                    resp.raise_for_status()
                    if debug_wire and not anthropic:
                        _log_wire(endpoint, sent, resp.text)
                    return resp.json()

                async def force_final(preface: str) -> str:
                    """Send one no-tools request and return the forced answer.

                    Args:
                        preface: Instruction telling the model to stop using
                            tools and answer now.

                    Returns:
                        The model's final text.

                    Raises:
                        LLMError: If the model still requests tools.
                    """
                    messages.append({"role": "user", "content": preface})
                    final_data = await post_round(False)
                    final_text, final_calls, _ = _parse_round(
                        cfg.api_style, final_data)
                    if final_calls:
                        raise LLMError(
                            "model kept requesting tools after the cap")
                    return final_text

                tracker = _CallTracker()
                for index in range(max_iterations):
                    data = await post_round(True)
                    text, calls, assistant_message = _parse_round(
                        cfg.api_style, data
                    )
                    if not calls:
                        break
                    # Consume one iteration: run every requested call and
                    # append results on this provider's wire format.
                    messages.append(assistant_message)
                    pairs, classes = _run_calls(
                        calls, execute, tracker, index + 1)
                    # Countdown so the model tracks its own remaining budget;
                    # rounds used counts this just-consumed turn (1-based).
                    note = _budget_note(index + 1, max_iterations)
                    if anthropic:
                        content = [
                            {"type": "tool_result",
                             "tool_use_id": call["id"],
                             "content": result}
                            for call, result in pairs
                        ]
                        content.append({"type": "text", "text": note})
                        messages.append({"role": "user", "content": content})
                    else:
                        for call, result in pairs:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": result,
                            })
                        messages.append({"role": "user", "content": note})
                    if tracker.note_round(classes):
                        # Model is looping on repeated calls: close tool access
                        # early and force an answer instead of burning the whole
                        # budget (and overflowing context) on the same calls.
                        logger.warning(
                            "[llm] loop guard tripped after %d round(s); "
                            "forcing final answer", index + 1)
                        text = await force_final(_STUCK_PREFACE)
                        break
                else:
                    # Cap exhausted without a plain-text turn: one final
                    # no-tools request forces an answer from what was gathered.
                    # Tell the model explicitly so it stops seeking tools and
                    # commits to an answer instead of returning empty content.
                    text = await force_final(_BUDGET_FINAL)
                if not isinstance(text, str) or not text.strip():
                    raise LLMError("provider returned non-text content")
                logger.info("[llm] %s/%s succeeded via %s",
                            cfg.base_url, cfg.model, cfg.api_style)
                return text, f"{cfg.base_url}/{cfg.model}"
        except Exception as exc:  # noqa: BLE001 - provider boundary
            label = endpoint or f"{cfg.base_url}/{cfg.model}"
            detail = f"{type(exc).__name__}: {exc}"
            logger.warning("[llm] %s failed; trying next provider: %s",
                           label, detail)
            errors.append(f"{label}: {detail}")
    raise LLMError("; ".join(errors) or "no providers configured")
