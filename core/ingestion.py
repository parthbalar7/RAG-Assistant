"""
Document ingestion pipeline with code-aware chunking + multimodal support.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from config import settings
from core.multimodal import extract_multimodal, is_multimodal_file

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    # Python
    ".py",
    ".pyi",
    ".pyw",
    # JavaScript / TypeScript
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".mjs",
    ".cjs",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
    # Java / Kotlin / Scala
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".groovy",
    ".gradle",
    # C / C++ / C#
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".csx",
    # Systems
    ".go",
    ".rs",
    ".swift",
    ".m",
    ".mm",
    # Scripting
    ".rb",
    ".php",
    ".pl",
    ".pm",
    ".lua",
    ".r",
    ".R",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".bat",
    ".cmd",
    ".ps1",
    # Config / Data
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".xml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".properties",  # Docs
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".adoc",
    ".tex",
    ".log",
    # SQL / DB
    ".sql",
    ".graphql",
    ".gql",
    # DevOps
    ".dockerfile",
    ".tf",
    ".hcl",
    # Multimodal
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    # Misc code
    ".dart",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".clj",
    ".lisp",
    ".elm",
    ".proto",
    ".thrift",
    ".avsc",
}

LANG_MAP = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".csx": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".m": "objc",
    ".mm": "objc",
    ".rb": "ruby",
    ".php": "php",
    ".pl": "perl",
    ".pm": "perl",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "bash",
    ".bat": "batch",
    ".cmd": "batch",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".env": "text",
    ".properties": "properties",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".adoc": "asciidoc",
    ".tex": "latex",
    ".log": "text",
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".dockerfile": "docker",
    ".tf": "hcl",
    ".hcl": "hcl",
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".clj": "clojure",
    ".lisp": "lisp",
    ".elm": "elm",
    ".proto": "protobuf",
    ".thrift": "thrift",
    ".avsc": "json",
}

SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    ".git",
    ".next",
    "dist",
    "build",
    ".tox",
    "env",
    ".idea",
    ".vscode",
    ".settings",
    ".gradle",
    "target",
    "bin",
    "obj",
    ".svn",
    ".hg",
    "vendor",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    "eggs",
    "*.egg-info",
}


@dataclass
class Document:
    content: str
    filepath: str
    language: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    content: str
    chunk_id: str
    document_path: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def display_source(self):
        return f"{self.document_path}:{self.start_line}-{self.end_line}"


_tokenizer = None
_tokenizer_unavailable = False


def count_tokens(text):
    global _tokenizer, _tokenizer_unavailable
    if _tokenizer_unavailable:
        return max(1, len(text) // 4)
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            _tokenizer_unavailable = True
            logger.warning("tiktoken encoding unavailable, using approximate token counts: %s", e)
            return max(1, len(text) // 4)
    return len(_tokenizer.encode(text, disallowed_special=()))


def _should_skip(filepath, root):
    """Check if a file should be skipped based on directory names.
    Fixed for Windows paths where parts[0] can be 'D:\\'."""
    try:
        rel = filepath.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = filepath.parts

    return any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1])


def load_documents(directory):
    """Recursively load all supported documents."""
    docs = []
    root = Path(directory)
    if not root.exists():
        logger.warning(f"Directory not found: {directory}")
        return docs

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if _should_skip(filepath, root):
            continue

        rel_path = str(filepath.relative_to(root)).replace("\\", "/")
        lang = LANG_MAP.get(filepath.suffix.lower(), "text")

        try:
            if is_multimodal_file(str(filepath)):
                extracted = extract_multimodal(str(filepath))
                if extracted and extracted.text.strip():
                    docs.append(
                        Document(
                            content=extracted.text,
                            filepath=rel_path,
                            language=lang,
                            metadata={
                                "multimodal": True,
                                "image_count": len(extracted.images),
                            },
                        )
                    )
                    logger.info(f"Loaded (multimodal): {rel_path} ({lang})")
                continue

            content = filepath.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue
            docs.append(Document(content=content, filepath=rel_path, language=lang))
            logger.info(f"Loaded: {rel_path} ({lang}, {count_tokens(content)} tokens)")

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")

    logger.info(f"Loaded {len(docs)} documents from {directory}")
    return docs


def load_single_file(filepath, base_dir=""):
    """Load a single file."""
    p = Path(filepath)
    if not p.is_file():
        return None
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None

    lang = LANG_MAP.get(p.suffix.lower(), "text")
    rel = str(p.relative_to(base_dir)).replace("\\", "/") if base_dir else p.name

    try:
        if is_multimodal_file(filepath):
            extracted = extract_multimodal(filepath)
            if extracted and extracted.text.strip():
                return Document(
                    content=extracted.text,
                    filepath=rel,
                    language=lang,
                    metadata={"multimodal": True},
                )
            return None

        content = p.read_text(encoding="utf-8", errors="replace")
        return Document(content=content, filepath=rel, language=lang)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None


# ── Chunking ──

# Emitted by core.multimodal PDF extraction; hard chunk boundary.
_PAGE_MARKER_RE = re.compile(r"^--- Page (\d+) ---$")
_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")
_BREADCRUMB_MAX_TOKENS = 25
# Floor so a pathological breadcrumb can never consume the whole chunk budget.
_MIN_CONTENT_BUDGET = 32
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
# Location-header comment marker per language (a "#" line inside JS/Java/C is a syntax error).
_COMMENT_PREFIX = {
    "javascript": "//",
    "typescript": "//",
    "go": "//",
    "rust": "//",
    "java": "//",
    "c": "//",
    "cpp": "//",
    "csharp": "//",
}


def _generate_chunk_id(path, start, end):
    raw = f"{path}:{start}:{end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── AST-aware code chunking (astchunk / tree-sitter) ──

_AST_LANGUAGES = {"python", "java", "csharp", "typescript"}
_ast_builders = {}
_ast_disabled = False
_ast_parse_warned = False


def _get_ast_builder(language):
    """Return a cached ASTChunkBuilder for the language, or None (regex fallback)."""
    global _ast_disabled
    if _ast_disabled or language not in _AST_LANGUAGES:
        return None
    if language in _ast_builders:
        return _ast_builders[language]
    try:
        from importlib.metadata import version

        # tree-sitter 0.26.0 wheels crash the whole process (access violation — not a
        # catchable exception) on Node.start_point with astchunk's grammar wheels;
        # only <0.26 is verified safe, so refuse to enable rather than risk a segfault.
        ts_ver = tuple(int(p) for p in version("tree-sitter").split(".")[:2])
        if ts_ver >= (0, 26):
            raise RuntimeError(f"tree-sitter {'.'.join(map(str, ts_ver))} segfaults on Node.start_point; need <0.26")
        from astchunk import ASTChunkBuilder

        _ast_builders[language] = ASTChunkBuilder(
            max_chunk_size=settings.chunk_size * 3, language=language, metadata_template="default"
        )
    except Exception as e:
        _ast_disabled = True
        logger.warning("astchunk unavailable, using regex code chunking: %s", e)
        return None
    return _ast_builders[language]


def _chunk_code_ast(doc, comment, budget):
    """Structure-aware chunking via astchunk. Returns None on any failure so
    _chunk_code can transparently fall back to the regex path."""
    global _ast_parse_warned
    builder = _get_ast_builder(doc.language)
    if builder is None:
        return None

    # astchunk budgets in non-whitespace characters; convert the token budget using
    # this file's own nws-chars-per-token ratio so chunks match the regex path's size.
    doc_tokens = count_tokens(doc.content)
    nws_chars = sum(1 for ch in doc.content if not ch.isspace())
    ratio = (nws_chars / doc_tokens) if doc_tokens else 3.0
    builder.max_chunk_size = max(int(budget * ratio), _MIN_CONTENT_BUDGET)

    try:
        windows = builder.chunkify(doc.content, repo_level_metadata={"filepath": doc.filepath})
    except Exception as e:
        if not _ast_parse_warned:
            _ast_parse_warned = True
            logger.warning("astchunk parse failed for %s, falling back to regex chunking: %s", doc.filepath, e)
        else:
            logger.debug("astchunk parse failed for %s: %s", doc.filepath, e)
        return None

    chunks = []
    for window in windows:
        content = (window.get("content") or "").rstrip()
        if not content or count_tokens(content) < settings.min_chunk_size:
            continue
        meta = window.get("metadata", {})
        # tree-sitter rows are 0-based inclusive; Chunk lines are 1-based inclusive.
        start0 = int(meta.get("start_line_no", 0))
        end0 = int(meta.get("end_line_no", start0))
        chunks.append(
            Chunk(
                content=f"{comment} {doc.filepath}:{start0 + 1}-{end0 + 1} ({doc.language})\n{content}",
                chunk_id=_generate_chunk_id(doc.filepath, start0, end0),
                document_path=doc.filepath,
                language=doc.language,
                start_line=start0 + 1,
                end_line=end0 + 1,
                chunk_type="code",
                metadata={"ast_chunk": True},
            )
        )
    return chunks or None


def _make_breadcrumb(path, heading_path):
    levels = [p for p in heading_path.split(" > ") if p] if heading_path else []
    levels = levels[-3:]
    while True:
        crumb = f"[{path} > {' > '.join(levels)}]" if levels else f"[{path}]"
        if count_tokens(crumb) <= _BREADCRUMB_MAX_TOKENS or len(levels) <= 2:
            return crumb
        levels = levels[1:]


def _chunk_code(doc):
    lines = doc.content.split("\n")
    chunks = []

    block_patterns = {
        "python": re.compile(r"^(class |def |async def |@)"),
        "javascript": re.compile(r"^(function |class |const |let |var |export |import )"),
        "typescript": re.compile(r"^(function |class |const |let |var |export |import |interface |type )"),
        "go": re.compile(r"^(func |type |var |const |package )"),
        "rust": re.compile(r"^(fn |pub fn |struct |enum |impl |mod |use )"),
        "java": re.compile(r"^(public |private |protected |class |interface |enum )"),
        "ruby": re.compile(r"^(def |class |module |require )"),
        "c": re.compile(r"^(int |void |char |float |double |struct |enum |typedef |#include )"),
        "cpp": re.compile(r"^(int |void |char |class |struct |enum |template |namespace |#include )"),
        "csharp": re.compile(r"^(public |private |protected |class |interface |enum |namespace |using )"),
    }

    pattern = block_patterns.get(doc.language)
    if not pattern:
        return _chunk_prose(doc)

    comment = _COMMENT_PREFIX.get(doc.language, "#")

    # chunk_size budget must cover the prepended location header; estimate with worst-case line numbers.
    header_tokens = count_tokens(f"{comment} {doc.filepath}:{len(lines)}-{len(lines)} ({doc.language})\n")
    budget = max(settings.chunk_size - header_tokens, _MIN_CONTENT_BUDGET)

    ast_chunks = _chunk_code_ast(doc, comment, budget)
    if ast_chunks is not None:
        return ast_chunks

    boundaries = [0]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and pattern.match(stripped) and i > 0 and i != boundaries[-1]:
            boundaries.append(i)
    boundaries.append(len(lines))

    buffer_start = boundaries[0]
    buffer_lines = []

    for i in range(len(boundaries) - 1):
        block = lines[boundaries[i] : boundaries[i + 1]]
        block_tokens = count_tokens("\n".join(block))

        if buffer_lines and count_tokens("\n".join(buffer_lines)) + block_tokens > budget:
            content = "\n".join(buffer_lines)
            if count_tokens(content) >= settings.min_chunk_size:
                start_line = buffer_start + 1
                end_line = boundaries[i]
                chunks.append(
                    Chunk(
                        content=f"{comment} {doc.filepath}:{start_line}-{end_line} ({doc.language})\n{content}",
                        chunk_id=_generate_chunk_id(doc.filepath, buffer_start, boundaries[i] - 1),
                        document_path=doc.filepath,
                        language=doc.language,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="code",
                    )
                )
            buffer_start = boundaries[i]
            buffer_lines = block
        else:
            buffer_lines.extend(block)

    if buffer_lines:
        content = "\n".join(buffer_lines)
        if count_tokens(content) >= settings.min_chunk_size:
            start_line = buffer_start + 1
            end_line = len(lines)
            chunks.append(
                Chunk(
                    content=f"{comment} {doc.filepath}:{start_line}-{end_line} ({doc.language})\n{content}",
                    chunk_id=_generate_chunk_id(doc.filepath, buffer_start, len(lines)),
                    document_path=doc.filepath,
                    language=doc.language,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="code",
                )
            )

    return chunks


def _chunk_prose(doc):
    lines = doc.content.split("\n")
    chunks = []

    # Each section: (start_line, lines, heading_path, page). Headings and PDF page
    # markers are hard boundaries — text is never merged across either.
    sections = []
    heading_stack = []
    current_start = 0
    current_lines = []
    current_heading_path = ""
    current_page = None
    active_page = None

    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
        if in_fence or _FENCE_RE.match(stripped):
            # "# comment" lines inside fenced code blocks are not headings
            current_lines.append(line)
            continue
        heading_match = _ATX_HEADING_RE.match(stripped)
        page_match = _PAGE_MARKER_RE.match(stripped)
        is_heading = stripped.startswith("#") or (i > 0 and lines[i - 1].strip() and re.match(r"^[=\-]{3,}$", stripped))

        if heading_match:
            level = len(heading_match.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_match.group(2).strip()))
        if page_match:
            active_page = int(page_match.group(1))

        if (is_heading or page_match) and current_lines:
            sections.append((current_start, current_lines, current_heading_path, current_page))
            current_start = i
            current_lines = [line]
            current_heading_path = " > ".join(title for _, title in heading_stack)
            current_page = active_page
        else:
            current_lines.append(line)
            if (is_heading or page_match) and len(current_lines) == 1:
                current_heading_path = " > ".join(title for _, title in heading_stack)
                current_page = active_page

    if current_lines:
        sections.append((current_start, current_lines, current_heading_path, current_page))

    # Merge sub-minimum sections forward so short PDF pages (title slides,
    # dividers) aren't silently dropped by the min_chunk_size filter below.
    merged_sections = []
    carry = None
    for sec in sections:
        if carry is not None:
            c_start, c_lines, c_hp, c_page = carry
            sec = (c_start, c_lines + sec[1], c_hp or sec[2], c_page if c_page is not None else sec[3])
            carry = None
        if count_tokens("\n".join(sec[1])) < settings.min_chunk_size:
            carry = sec
        else:
            merged_sections.append(sec)
    if carry is not None:
        if merged_sections:
            p_start, p_lines, p_hp, p_page = merged_sections[-1]
            merged_sections[-1] = (p_start, p_lines + carry[1], p_hp, p_page)
        else:
            merged_sections.append(carry)
    sections = merged_sections

    for start_line, section_lines, heading_path, page in sections:
        # Breadcrumb counts against chunk_size so the embedded text still fits the model window.
        crumb_line = _make_breadcrumb(doc.filepath, heading_path) + "\n"
        budget = max(settings.chunk_size - count_tokens(crumb_line), _MIN_CONTENT_BUDGET)
        meta = {"heading_path": heading_path}
        if page is not None:
            meta["page"] = page

        section_text = "\n".join(section_lines)
        token_count = count_tokens(section_text)

        if token_count <= budget:
            if token_count >= settings.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=crumb_line + section_text,
                        chunk_id=_generate_chunk_id(doc.filepath, start_line, start_line + len(section_lines)),
                        document_path=doc.filepath,
                        language=doc.language,
                        start_line=start_line + 1,
                        end_line=start_line + len(section_lines),
                        chunk_type="prose",
                        metadata=dict(meta),
                    )
                )
        else:
            window = []
            window_start = start_line
            for j, sl in enumerate(section_lines):
                window.append(sl)
                if count_tokens("\n".join(window)) >= budget:
                    content = "\n".join(window)
                    chunks.append(
                        Chunk(
                            content=crumb_line + content,
                            chunk_id=_generate_chunk_id(doc.filepath, window_start, start_line + j),
                            document_path=doc.filepath,
                            language=doc.language,
                            start_line=window_start + 1,
                            end_line=start_line + j + 1,
                            chunk_type="prose",
                            metadata=dict(meta),
                        )
                    )
                    overlap_tokens = 0
                    overlap_start = len(window)
                    for k in range(len(window) - 1, -1, -1):
                        overlap_tokens += count_tokens(window[k])
                        if overlap_tokens >= settings.chunk_overlap:
                            overlap_start = k
                            break
                    window = window[overlap_start:]
                    window_start = start_line + j - len(window) + 1

            if window and count_tokens("\n".join(window)) >= settings.min_chunk_size:
                content = "\n".join(window)
                chunks.append(
                    Chunk(
                        content=crumb_line + content,
                        chunk_id=_generate_chunk_id(doc.filepath, window_start, start_line + len(section_lines)),
                        document_path=doc.filepath,
                        language=doc.language,
                        start_line=window_start + 1,
                        end_line=start_line + len(section_lines),
                        chunk_type="prose",
                        metadata=dict(meta),
                    )
                )

    return chunks


CODE_LANGUAGES = {"python", "javascript", "typescript", "go", "rust", "java", "ruby", "bash", "c", "cpp", "csharp"}


def chunk_document(doc):
    if doc.language in CODE_LANGUAGES:
        chunks = _chunk_code(doc)
    else:
        chunks = _chunk_prose(doc)

    for chunk in chunks:
        chunk.metadata["language"] = doc.language
        chunk.metadata["source_file"] = doc.filepath
        chunk.metadata.update(doc.metadata)

    logger.info(f"Chunked {doc.filepath} -> {len(chunks)} chunks")
    return chunks


def ingest_directory(directory):
    docs = load_documents(directory)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks
