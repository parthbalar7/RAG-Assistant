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
5. End with a ## Sources section listing each source used.

Formatting (always follow):
- Use **Markdown** for all responses.
- Use ## headings to separate major sections or topics.
- Use bullet points or numbered lists for multi-item answers — never write long run-on paragraphs.
- Use **bold** for key terms and `code` for technical identifiers.
- Keep paragraphs short (2-3 sentences max).
- Use tables when comparing multiple items.
- Add blank lines between sections for readability."""

SYSTEM_PROMPT_WITH_MEMORY = """You are a precise technical documentation assistant with persistent memory.

Rules:
1. Base every claim on the provided context. If context is insufficient, say so.
2. Cite sources as [file:start_line-end_line] for every claim.
3. Include code snippets with syntax highlighting when relevant.
4. Never invent APIs, functions, or behaviors not in the context.
5. End with a ## Sources section listing each source used.
6. Use recalled memories from past conversations naturally without mentioning the memory system.

Formatting (always follow):
- Use **Markdown** for all responses.
- Use ## headings to separate major sections or topics.
- Use bullet points or numbered lists for multi-item answers — never write long run-on paragraphs.
- Use **bold** for key terms and `code` for technical identifiers.
- Keep paragraphs short (2-3 sentences max).
- Use tables when comparing multiple items.
- Add blank lines between sections for readability."""


# Direct/no-retrieval answers (empty store, learned router 'direct' route):
# the document-grounded prompts would force "context is insufficient" refusals
# and fabricated Sources sections when there are deliberately no chunks.
SYSTEM_PROMPT_CHAT = """You are a helpful technical assistant with persistent memory of past conversations.

Rules:
1. Answer conversationally from general knowledge, the conversation history, and any recalled memories.
2. Use recalled memories naturally without mentioning the memory system.
3. If the question clearly needs the user's indexed documents and none are available, say so briefly and suggest indexing them.
4. Use **Markdown**; keep paragraphs short; use `code` for technical identifiers.
5. Do NOT emit a Sources section — there are no retrieved documents."""


def _format_context(hits):
    if not hits:
        return "No relevant context was found."
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        source = meta.get(
            "source",
            "{}:{}-{}".format(meta.get("document_path", "?"), meta.get("start_line", "?"), meta.get("end_line", "?")),
        )
        lang = meta.get("language", "text")
        score = hit.get("rerank_score", hit.get("score", 0))
        parts.append("--- {} [{}] [{}] [{:.2f}] ---\n{}\n".format(i, source, lang, score, hit["content"]))
    return "\n".join(parts)


def _build_messages(query, hits, conversation_history, memory_context):
    """Build the messages list for the LLM call."""
    messages = []
    if conversation_history:
        for msg in conversation_history[-(settings.max_history_turns * 2) :]:
            content = msg.content if len(msg.content) <= 600 else msg.content[:600] + "..."
            messages.append({"role": msg.role, "content": content})

    parts = []
    if memory_context and memory_context.formatted:
        parts.append(memory_context.formatted)
    parts.append(f"## Retrieved context\n\n{_format_context(hits)}")
    parts.append(f"\n## Question\n\n{query}")
    messages.append({"role": "user", "content": "\n".join(parts)})
    return messages


def _estimate_tokens(text):
    try:
        # Lazy import — ingestion pulls in tiktoken/multimodal, not needed at module load
        from core.ingestion import count_tokens

        return count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _warn_if_context_overflow(messages, system):
    """Ollama silently truncates anything past num_ctx — no error, the model just never sees it."""
    if llm_client.get_backend() != "ollama":
        return
    total = _estimate_tokens(system) + sum(_estimate_tokens(m["content"]) for m in messages)
    # num_ctx covers prompt AND generated tokens — reserve num_predict headroom,
    # or generation triggers context shifting that evicts the earliest prompt.
    budget = settings.ollama_num_ctx - settings.llm_max_tokens
    if total > budget:
        logger.warning(
            "Assembled prompt is ~%d tokens but ollama_num_ctx=%d leaves only %d after reserving "
            "%d output tokens — Ollama will silently drop or shift the excess. "
            "Reduce top_k/history or raise RAG_OLLAMA_NUM_CTX.",
            total,
            settings.ollama_num_ctx,
            budget,
            settings.llm_max_tokens,
        )


def _pick_system_prompt(hits, memory_context):
    if not hits:
        return SYSTEM_PROMPT_CHAT
    has_mem = memory_context and memory_context.count > 0
    return SYSTEM_PROMPT_WITH_MEMORY if has_mem else SYSTEM_PROMPT


def generate(query, hits, conversation_history=None, model=None, memory_context=None):
    messages = _build_messages(query, hits, conversation_history, memory_context)
    sys_prompt = _pick_system_prompt(hits, memory_context)
    _warn_if_context_overflow(messages, sys_prompt)

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
    _warn_if_context_overflow(messages, sys_prompt)

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
            citations.append(
                Citation(
                    source_file=doc_path,
                    lines="{}-{}".format(meta.get("start_line", "?"), meta.get("end_line", "?")),
                    relevance="Score: {:.3f}".format(hit.get("rerank_score", hit.get("score", 0))),
                )
            )
    return citations
