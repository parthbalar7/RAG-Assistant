"""
Unified LLM client — routes to Anthropic or Ollama based on runtime config.

Supports multi-node Ollama: set RAG_OLLAMA_EXTRA_NODES=http://macbook:11434,http://other:11434
in .env to distribute calls across machines via round-robin with automatic failover.

Usage:
    from core.llm_client import chat, get_backend, set_backend, list_ollama_models

    # Non-streaming
    text = chat(messages, system="You are helpful.", stream=False)

    # Streaming — yields str tokens
    for token in chat(messages, system="...", stream=True):
        print(token, end="")
"""

import logging
import threading
import time
from collections.abc import Generator

from config import settings

logger = logging.getLogger(__name__)

# Runtime override — allows switching backend without restarting the server
_runtime: dict[str, str | None] = {
    "backend": None,  # None means "use settings.llm_backend"
    "model": None,  # None means "use settings.llm_model / settings.ollama_model"
}

# Cached Anthropic client
_anthropic_client = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_backend() -> str:
    return _runtime["backend"] or settings.llm_backend


def get_model() -> str:
    if _runtime["model"]:
        return _runtime["model"]
    backend = get_backend()
    return settings.llm_model if backend == "anthropic" else settings.ollama_model


def set_backend(backend: str, model: str | None = None):
    """Switch backend at runtime. backend must be 'anthropic' or 'ollama'."""
    if backend not in ("anthropic", "ollama"):
        raise ValueError(f"Unknown backend '{backend}'. Choose 'anthropic' or 'ollama'.")
    _runtime["backend"] = backend
    _runtime["model"] = model
    logger.info("LLM backend switched → %s | model: %s", backend, model or "(default)")


def get_memory_model() -> str:
    """Return the appropriate model for lightweight tasks (memory extraction)."""
    backend = get_backend()
    if backend == "ollama":
        return settings.ollama_memory_model
    return settings.memory_extraction_model


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------


def chat(
    messages: list[dict[str, str]],
    system: str = "",
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stream: bool = False,
    *,
    keep_alive: str | None = None,
    json_schema: dict | None = None,
) -> str | Generator[str, None, None]:
    """
    Unified chat call.

    Args:
        messages:    [{"role": "user"|"assistant", "content": str}, ...]
        system:      System prompt string.
        model:       Override model name (uses backend default if None).
        max_tokens:  Max output tokens (uses settings default if None).
        temperature: Sampling temperature (uses settings default if None).
        stream:      If True, returns a generator that yields str tokens.
        keep_alive:  Ollama model residency (e.g. "1h", "1m"); defaults to
                     settings.ollama_keep_alive. Ignored on Anthropic.
        json_schema: JSON schema dict for Ollama structured outputs (passed as
                     `format`). Ignored on Anthropic — Claude parses fine
                     unconstrained.

    Returns:
        str if stream=False, generator[str] if stream=True.
    """
    result = _dispatch(messages, system, model, max_tokens, temperature, stream, keep_alive, json_schema)
    if stream:
        return result
    text, _usage = result
    return text


def chat_with_usage(
    messages: list[dict[str, str]],
    system: str = "",
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    *,
    keep_alive: str | None = None,
    json_schema: dict | None = None,
) -> tuple:
    """Non-streaming chat that also reports token usage.

    keep_alive and json_schema apply to the Ollama backend only (json_schema
    maps to Ollama's `format` for schema-constrained decoding); the Anthropic
    backend ignores both.

    Returns:
        (text, {"input_tokens": int, "output_tokens": int})
    """
    return _dispatch(
        messages, system, model, max_tokens, temperature, stream=False, keep_alive=keep_alive, json_schema=json_schema
    )


def _dispatch(messages, system, model, max_tokens, temperature, stream, keep_alive=None, json_schema=None):
    backend = get_backend()
    model = model or get_model()
    max_tokens = max_tokens or settings.llm_max_tokens
    temperature = temperature if temperature is not None else settings.llm_temperature

    if backend == "anthropic":
        return _anthropic_chat(messages, system, model, max_tokens, temperature, stream)
    elif backend == "ollama":
        return _ollama_chat(messages, system, model, max_tokens, temperature, stream, keep_alive, json_schema)
    else:
        raise ValueError(f"Unknown LLM backend: '{backend}'")


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _anthropic_chat(messages, system, model, max_tokens, temperature, stream):
    client = _get_anthropic_client()
    # Structured system block with cache_control so repeated prefixes (agent
    # loop, per-session system prompts) hit the prompt cache at 0.1x price.
    if system:
        system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    if stream:

        def _gen():
            with client.messages.stream(**kwargs) as s:
                yield from s.text_stream

        return _gen()
    else:
        resp = client.messages.create(**kwargs)
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        }
        return resp.content[0].text, usage


# ---------------------------------------------------------------------------
# Ollama backend — multi-node round-robin with failover
# ---------------------------------------------------------------------------

# Node health tracking: {url: last_failure_timestamp}
_node_failures: dict[str, float] = {}
_node_lock = threading.Lock()
_node_index = 0  # round-robin counter
_NODE_COOLDOWN = 60  # seconds before retrying a failed node


def _get_ollama_nodes() -> list[str]:
    """Return all configured Ollama node URLs (primary + extras)."""
    nodes = [settings.ollama_base_url]
    if settings.ollama_extra_nodes.strip():
        for url in settings.ollama_extra_nodes.split(","):
            url = url.strip()
            if url and url not in nodes:
                nodes.append(url)
    return nodes


def _pick_node() -> str:
    """Round-robin pick from healthy nodes, falling back to primary if all are down."""
    global _node_index
    nodes = _get_ollama_nodes()
    if len(nodes) == 1:
        return nodes[0]

    now = time.time()
    with _node_lock:
        # Try up to len(nodes) times to find a healthy one
        for _ in range(len(nodes)):
            _node_index = (_node_index + 1) % len(nodes)
            candidate = nodes[_node_index]
            fail_time = _node_failures.get(candidate, 0)
            if now - fail_time > _NODE_COOLDOWN:
                return candidate
        # All nodes failed recently — try primary anyway
        return nodes[0]


def _mark_failed(url: str):
    with _node_lock:
        _node_failures[url] = time.time()
        logger.warning("Ollama node marked down for %ds: %s", _NODE_COOLDOWN, url)


def _mark_ok(url: str):
    with _node_lock:
        _node_failures.pop(url, None)


def _build_ollama_messages(messages, system):
    """Prepend system message if provided (Ollama uses it in the messages list)."""
    result = []
    if system:
        result.append({"role": "system", "content": system})
    result.extend(messages)
    return result


def _ollama_chat(messages, system, model, max_tokens, temperature, stream, keep_alive=None, json_schema=None):
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError("Ollama Python package not installed. Run: pip install ollama>=0.4.0") from e

    ollama_messages = _build_ollama_messages(messages, system)
    # num_ctx overrides Ollama's 4k default, which silently truncates long prompts
    options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": settings.ollama_num_ctx}
    chat_kwargs = {
        "model": model,
        "messages": ollama_messages,
        "options": options,
        "keep_alive": keep_alive or settings.ollama_keep_alive,
    }
    if json_schema:
        chat_kwargs["format"] = json_schema
    nodes = _get_ollama_nodes()
    max_attempts = len(nodes)

    # Try nodes with failover
    for attempt in range(max_attempts):
        node_url = _pick_node()
        client = ollama.Client(host=node_url)
        tag = node_url.split("//")[-1].split(":")[0]  # short label for logs

        if stream:
            # Establish the stream and pull the first chunk eagerly so a dead
            # node fails here, where we can still fail over to another node.
            try:
                logger.debug("Ollama stream → %s", tag)
                stream_iter = client.chat(stream=True, **chat_kwargs)
                first_chunk = next(stream_iter, None)
            except Exception as e:
                _mark_failed(node_url)
                if attempt < max_attempts - 1:
                    logger.warning("Ollama node %s failed, trying next: %s", tag, e)
                    continue
                logger.error("All Ollama nodes failed. Last error (%s): %s", tag, e)
                raise

            def _gen(url=node_url, it=stream_iter, first=first_chunk):
                try:
                    if first is not None:
                        token = first.message.content
                        if token:
                            yield token
                    for chunk in it:
                        token = chunk.message.content
                        if token:
                            yield token
                    _mark_ok(url)
                except Exception as e:
                    logger.error("Ollama stream error (%s): %s", url, e)
                    _mark_failed(url)
                    raise

            return _gen()
        else:
            try:
                logger.debug("Ollama chat → %s", tag)
                resp = client.chat(**chat_kwargs)
                _mark_ok(node_url)
                usage = {
                    "input_tokens": getattr(resp, "prompt_eval_count", 0) or 0,
                    "output_tokens": getattr(resp, "eval_count", 0) or 0,
                }
                return resp.message.content, usage
            except Exception as e:
                _mark_failed(node_url)
                if attempt < max_attempts - 1:
                    logger.warning("Ollama node %s failed, trying next: %s", tag, e)
                    continue
                logger.error("All Ollama nodes failed. Last error (%s): %s", tag, e)
                raise


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def list_ollama_models() -> list[str]:
    """Return model names from all reachable Ollama nodes (deduplicated)."""
    try:
        import ollama
    except ImportError:
        return []

    seen = set()
    models = []
    for url in _get_ollama_nodes():
        try:
            client = ollama.Client(host=url)
            result = client.list()
            for m in result.models:
                if m.model not in seen:
                    seen.add(m.model)
                    models.append(m.model)
        except Exception as e:
            logger.warning("Could not list models from %s: %s", url, e)
    return models


def ollama_reachable() -> bool:
    """True if at least one Ollama node is reachable."""
    try:
        import ollama
    except ImportError:
        return False
    for url in _get_ollama_nodes():
        try:
            ollama.Client(host=url).list()
            return True
        except Exception:
            continue
    return False


def get_node_status() -> list[dict]:
    """Return health status of all configured Ollama nodes."""
    try:
        import ollama
    except ImportError:
        return []

    now = time.time()
    statuses = []
    for url in _get_ollama_nodes():
        reachable = False
        models = []
        try:
            result = ollama.Client(host=url).list()
            reachable = True
            models = [m.model for m in result.models]
        except Exception:
            pass
        fail_time = _node_failures.get(url, 0)
        statuses.append(
            {
                "url": url,
                "reachable": reachable,
                "models": models,
                "cooldown": fail_time > 0 and (now - fail_time) < _NODE_COOLDOWN,
            }
        )
    return statuses
