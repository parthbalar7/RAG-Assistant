"""
Unified LLM client — routes to Anthropic or Ollama based on runtime config.

Usage:
    from core.llm_client import chat, get_backend, set_backend, list_ollama_models

    # Non-streaming
    text = chat(messages, system="You are helpful.", stream=False)

    # Streaming — yields str tokens
    for token in chat(messages, system="...", stream=True):
        print(token, end="")
"""

import logging
from typing import Generator, List, Dict, Optional, Union

from config import settings

logger = logging.getLogger(__name__)

# Runtime override — allows switching backend without restarting the server
_runtime: Dict[str, Optional[str]] = {
    "backend": None,   # None means "use settings.llm_backend"
    "model": None,     # None means "use settings.llm_model / settings.ollama_model"
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


def set_backend(backend: str, model: Optional[str] = None):
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
    messages: List[Dict[str, str]],
    system: str = "",
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stream: bool = False,
) -> Union[str, Generator[str, None, None]]:
    """
    Unified chat call.

    Args:
        messages:    [{"role": "user"|"assistant", "content": str}, ...]
        system:      System prompt string.
        model:       Override model name (uses backend default if None).
        max_tokens:  Max output tokens (uses settings default if None).
        temperature: Sampling temperature (uses settings default if None).
        stream:      If True, returns a generator that yields str tokens.

    Returns:
        str if stream=False, generator[str] if stream=True.
    """
    backend = get_backend()
    model = model or get_model()
    max_tokens = max_tokens or settings.llm_max_tokens
    temperature = temperature if temperature is not None else settings.llm_temperature

    if backend == "anthropic":
        return _anthropic_chat(messages, system, model, max_tokens, temperature, stream)
    elif backend == "ollama":
        return _ollama_chat(messages, system, model, max_tokens, temperature, stream)
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
                for text in s.text_stream:
                    yield text
        return _gen()
    else:
        resp = client.messages.create(**kwargs)
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _build_ollama_messages(messages, system):
    """Prepend system message if provided (Ollama uses it in the messages list)."""
    result = []
    if system:
        result.append({"role": "system", "content": system})
    result.extend(messages)
    return result


def _ollama_chat(messages, system, model, max_tokens, temperature, stream):
    try:
        import ollama
    except ImportError:
        raise RuntimeError(
            "Ollama Python package not installed. Run: pip install ollama>=0.4.0"
        )

    ollama_messages = _build_ollama_messages(messages, system)
    options = {"temperature": temperature, "num_predict": max_tokens}
    client = ollama.Client(host=settings.ollama_base_url)

    if stream:
        def _gen():
            try:
                for chunk in client.chat(
                    model=model,
                    messages=ollama_messages,
                    stream=True,
                    options=options,
                ):
                    token = chunk.message.content
                    if token:
                        yield token
            except Exception as e:
                logger.error("Ollama stream error: %s", e)
                raise
        return _gen()
    else:
        try:
            resp = client.chat(
                model=model,
                messages=ollama_messages,
                options=options,
            )
            return resp.message.content
        except Exception as e:
            logger.error("Ollama chat error: %s", e)
            raise


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def list_ollama_models() -> List[str]:
    """Return locally available Ollama model names. Empty list if Ollama is unreachable."""
    try:
        import ollama
        client = ollama.Client(host=settings.ollama_base_url)
        result = client.list()
        return [m.model for m in result.models]
    except Exception as e:
        logger.warning("Could not list Ollama models: %s", e)
        return []


def ollama_reachable() -> bool:
    """Quick connectivity check for Ollama server."""
    try:
        import ollama
        ollama.Client(host=settings.ollama_base_url).list()
        return True
    except Exception:
        return False
