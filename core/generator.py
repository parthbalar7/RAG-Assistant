"""
LLM generation layer — backend-agnostic (Anthropic or Ollama).
Token-optimized with memory-augmented context injection.
"""

import logging
from dataclasses import dataclass, field

from config import settings
from core import llm_client

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    source_file: str
    lines: str
    relevance: str


@dataclass
class RAGResponse:
    answer: str
    citations: list = field(default_factory=list)
    context_used: int = 0
    model: str = ""
    usage: dict = field(default_factory=dict)
    memories_used: int = 0


@dataclass
class Message:
    role: str
    content: str


SYSTEM_PROMPT = """You are a precise technical documentation assistant. Answer questions using ONLY the provided context chunks.

Rules:
1. Base every claim on the provided context. If context is insufficient, say so.
2. Cite sources as [file:start_line-end_line] for every claim.
3. Include code snippets with syntax highlighting when relevant.
4. Never invent APIs, functions, or behaviors not in the context.
5. End with a ## Sources section listing each source used."""

SYSTEM_PROMPT_WITH_MEMORY = """You are a precise technical documentation assistant with persistent memory.

Rules:
1. Base every claim on the provided context. If context is insufficient, say so.
2. Cite sources as [file:start_line-end_line] for every claim.
3. Include code snippets with syntax highlighting when relevant.
4. Never invent APIs, functions, or behaviors not in the context.
5. End with a ## Sources section listing each source used.
6. Use recalled memories from past conversations naturally without mentioning the memory system."""


def _format_context(hits):
    if not hits:
        return "No relevant context was found."
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        source = meta.get("source", "{}:{}-{}".format(
            meta.get("document_path", "?"),
            meta.get("start_line", "?"),
            meta.get("end_line", "?")))
        lang = meta.get("language", "text")
        score = hit.get("rerank_score", hit.get("score", 0))
        parts.append(
            "--- {} [{}] [{}] [{:.2f}] ---\n{}\n".format(
                i, source, lang, score, hit["content"])
        )
    return "\n".join(parts)


def _build_messages(query, hits, conversation_history, memory_context):
    """Build the messages list for the LLM call."""
    messages = []
    if conversation_history:
        for msg in conversation_history[-(settings.max_history_turns * 2):]:
            content = msg.content if len(msg.content) <= 600 else msg.content[:600] + "..."
            messages.append({"role": msg.role, "content": content})

    parts = []
    if memory_context and memory_context.formatted:
        parts.append(memory_context.formatted)
    parts.append("## Retrieved context\n\n{}".format(_format_context(hits)))
    parts.append("\n## Question\n\n{}".format(query))
    messages.append({"role": "user", "content": "\n".join(parts)})
    return messages


def generate(query, hits, conversation_history=None, model=None, memory_context=None):
    messages = _build_messages(query, hits, conversation_history, memory_context)
    has_mem = memory_context and memory_context.count > 0
    sys_prompt = SYSTEM_PROMPT_WITH_MEMORY if has_mem else SYSTEM_PROMPT

    answer = llm_client.chat(
        messages=messages,
        system=sys_prompt,
        model=model,
        stream=False,
    )

    used_model = model or llm_client.get_model()
    return RAGResponse(
        answer=answer,
        citations=_extract_citations(answer, hits),
        context_used=len(hits),
        model=used_model,
        usage={},
        memories_used=memory_context.count if memory_context else 0,
    )


def generate_stream(query, hits, conversation_history=None, model=None, memory_context=None):
    messages = _build_messages(query, hits, conversation_history, memory_context)
    has_mem = memory_context and memory_context.count > 0
    sys_prompt = SYSTEM_PROMPT_WITH_MEMORY if has_mem else SYSTEM_PROMPT

    yield from llm_client.chat(
        messages=messages,
        system=sys_prompt,
        model=model,
        stream=True,
    )


def _extract_citations(answer, hits):
    citations = []
    seen = set()
    for hit in hits:
        meta = hit["metadata"]
        doc_path = meta.get("document_path", "")
        if doc_path and doc_path in answer and doc_path not in seen:
            seen.add(doc_path)
            citations.append(Citation(
                source_file=doc_path,
                lines="{}-{}".format(meta.get("start_line", "?"),
                                     meta.get("end_line", "?")),
                relevance="Score: {:.3f}".format(
                    hit.get("rerank_score", hit.get("score", 0))),
            ))
    return citations


# Keep this for any legacy callers (compliance.py used to import it)
def get_client():
    return llm_client._get_anthropic_client()
