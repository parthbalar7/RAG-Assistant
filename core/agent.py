"""
Agentic RAG: Claude decides when to search, follow up, and synthesize answers.
Tools: search, get_file, answer
"""

import json
import logging
import re
from dataclasses import dataclass, field

import anthropic
from config import settings

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are an expert code documentation assistant with access to a searchable knowledge base.

You have these tools to investigate the codebase:

1. **search** - Search the knowledge base with a query
   Use: {"tool": "search", "query": "your search query"}

2. **get_file** - Retrieve all indexed content from a specific file
   Use: {"tool": "get_file", "path": "path/to/file.py"}

3. **answer** - Provide your final answer to the user
   Use: {"tool": "answer", "response": "your detailed answer with citations"}

Process:
1. Analyze the user's question
2. Use search/get_file tools to gather relevant context (you can use multiple)
3. When you have enough information, use the answer tool
4. Always cite sources as [file:lines]
5. If you can't find enough info, say so honestly

Respond with ONE tool call at a time as JSON. Do NOT include any text outside the JSON."""


@dataclass
class AgentStep:
    step_num: int
    tool: str
    input_data: dict
    output: str
    token_count: int = 0


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    total_tokens: int = 0


def run_agent(query, store, retrieve_fn=None, conversation_history=None, max_steps=None, on_step=None):
    """Run the agentic RAG loop."""
    from core.retriever import retrieve

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    max_steps = max_steps or settings.agent_max_steps

    messages = []
    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": "User question: {}".format(query)})

    all_context = []
    steps = []
    total_tokens = 0

    for step_num in range(1, max_steps + 1):
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=1024,
                temperature=0.1,
                system=AGENT_SYSTEM_PROMPT,
                messages=messages,
            )

            total_tokens += response.usage.input_tokens + response.usage.output_tokens
            raw_text = response.content[0].text.strip()

            tool_call = _parse_tool_call(raw_text)
            if not tool_call:
                step = AgentStep(step_num, "answer", {}, raw_text, response.usage.output_tokens)
                steps.append(step)
                if on_step:
                    on_step(step)
                return AgentResult(answer=raw_text, steps=steps, sources=_dedupe_sources(all_context), total_tokens=total_tokens)

            tool_name = tool_call.get("tool", "")

            if tool_name == "answer":
                final_answer = tool_call.get("response", raw_text)
                step = AgentStep(step_num, "answer", tool_call, final_answer, response.usage.output_tokens)
                steps.append(step)
                if on_step:
                    on_step(step)
                return AgentResult(answer=final_answer, steps=steps, sources=_dedupe_sources(all_context), total_tokens=total_tokens)

            elif tool_name == "search":
                search_query = tool_call.get("query", query)
                lang_filter = tool_call.get("language")
                hits = retrieve(store=store, query=search_query, use_reranking=True, use_hybrid=True, language_filter=lang_filter)
                all_context.extend(hits)

                context_text = _format_hits(hits)
                step = AgentStep(step_num, "search", tool_call, "Found {} results".format(len(hits)), response.usage.output_tokens)
                steps.append(step)
                if on_step:
                    on_step(step)

                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": "Search results for '{}':\n{}".format(search_query, context_text)})

            elif tool_name == "get_file":
                file_path = tool_call.get("path", "")
                hits = store.vector_search(file_path, top_k=20)
                file_hits = [h for h in hits if h["metadata"].get("document_path", "") == file_path]
                all_context.extend(file_hits)

                context_text = _format_hits(file_hits) if file_hits else "No indexed content found for '{}'".format(file_path)
                step = AgentStep(step_num, "get_file", tool_call, "Found {} chunks from {}".format(len(file_hits), file_path), response.usage.output_tokens)
                steps.append(step)
                if on_step:
                    on_step(step)

                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": "Contents of {}:\n{}".format(file_path, context_text)})

            else:
                step = AgentStep(step_num, "unknown", tool_call, raw_text, response.usage.output_tokens)
                steps.append(step)
                break

        except Exception as e:
            logger.error("Agent step {} failed: {}".format(step_num, e))
            step = AgentStep(step_num, "error", {}, str(e))
            steps.append(step)
            break

    return AgentResult(
        answer="I was unable to find a complete answer within the allowed steps. Here's what I found:\n\n" +
               "\n".join("- Step {} ({}): {}".format(s.step_num, s.tool, s.output) for s in steps),
        steps=steps,
        sources=_dedupe_sources(all_context),
        total_tokens=total_tokens,
    )


def _parse_tool_call(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _format_hits(hits):
    parts = []
    for i, hit in enumerate(hits[:6], 1):
        meta = hit["metadata"]
        source = meta.get("source", meta.get("document_path", "unknown"))
        parts.append("--- [{}] ---\n{}\n".format(source, hit["content"][:800]))
    return "\n".join(parts) if parts else "No results found."


def _dedupe_sources(hits):
    seen = set()
    sources = []
    for hit in hits:
        meta = hit.get("metadata", {})
        key = "{}:{}".format(meta.get("document_path", ""), meta.get("start_line", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": meta.get("document_path", ""),
                "lines": "{}-{}".format(meta.get("start_line", "?"), meta.get("end_line", "?")),
                "language": meta.get("language", ""),
                "score": round(hit.get("rerank_score", hit.get("score", 0)), 4),
                "preview": hit["content"][:200],
            })
    return sources
