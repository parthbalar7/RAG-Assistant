"""
Compliance scanning: samples ChromaDB chunks and uses Claude to analyze
them against regulatory frameworks (HIPAA, PCI DSS, GDPR, SOC2, OWASP).
"""

import json
import logging
import time
from typing import Any, Dict, List

from config import settings

logger = logging.getLogger(__name__)

FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "HIPAA": {
        "name": "HIPAA",
        "full_name": "Health Insurance Portability and Accountability Act",
        "focus_areas": [
            "PHI/ePHI exposure in logs, comments, or hardcoded values",
            "Missing encryption for data at rest or in transit",
            "Lack of access controls and role-based authentication",
            "Missing audit logging for PHI access",
            "Inadequate data retention and disposal policies",
            "Insufficient de-identification of health data",
            "No breach notification logic",
            "Business Associate Agreement gaps",
        ],
    },
    "PCI_DSS": {
        "name": "PCI DSS",
        "full_name": "Payment Card Industry Data Security Standard",
        "focus_areas": [
            "Hardcoded or logged credit card numbers, CVVs, PANs",
            "Storing cardholder data unencrypted",
            "Missing TLS/HTTPS for payment data transmission",
            "Weak authentication for payment systems",
            "Missing input validation susceptible to injection",
            "Logging of sensitive payment data",
            "Lack of tokenization or masking",
            "Insecure key management",
        ],
    },
    "GDPR": {
        "name": "GDPR",
        "full_name": "General Data Protection Regulation",
        "focus_areas": [
            "Collection of personal data without consent mechanisms",
            "Missing data minimization (collecting more than needed)",
            "No right-to-erasure or data portability implementation",
            "Missing data breach notification logic (72-hour rule)",
            "Cross-border data transfer without safeguards",
            "Hardcoded or logged PII (names, emails, IPs, IDs)",
            "No privacy-by-design patterns",
            "Missing Data Protection Impact Assessment triggers",
        ],
    },
    "SOC2": {
        "name": "SOC 2",
        "full_name": "Service Organization Control 2",
        "focus_areas": [
            "Missing access controls and least-privilege principles",
            "Inadequate audit logging and monitoring",
            "Error handling that exposes system internals",
            "Hardcoded credentials or API keys",
            "Missing input validation",
            "Insufficient availability and redundancy patterns",
            "No change management or versioning evidence",
            "Vendor and third-party risk gaps",
        ],
    },
    "OWASP": {
        "name": "OWASP Top 10",
        "full_name": "OWASP Top 10 Web Application Security Risks",
        "focus_areas": [
            "A01 Broken Access Control — missing authorization checks",
            "A02 Cryptographic Failures — weak or missing encryption",
            "A03 Injection — SQL, command, LDAP injection vulnerabilities",
            "A04 Insecure Design — missing threat modeling patterns",
            "A05 Security Misconfiguration — debug mode, default creds",
            "A06 Vulnerable Components — outdated or risky dependencies",
            "A07 Auth Failures — weak passwords, no MFA, broken sessions",
            "A08 Data Integrity Failures — insecure deserialization",
            "A09 Logging Failures — insufficient security event logging",
            "A10 SSRF — server-side request forgery patterns",
        ],
    },
}

_PROMPT = """\
You are a {fw_name} ({fw_full}) compliance expert auditing source code.

Focus areas for {fw_name}:
{focus_areas}

Analyze the code/documentation chunks below for compliance violations.

{code_context}

Respond with ONLY a JSON object — no markdown fences, no extra text:
{{
  "summary": "<2-3 sentence overall compliance posture>",
  "risk_score": <integer 0-100, where 0=fully compliant and 100=critical risk>,
  "issues": [
    {{
      "severity": "critical|high|medium|low",
      "category": "<compliance area>",
      "title": "<short title under 80 chars>",
      "description": "<what the problem is and why it matters>",
      "file": "<filename or 'multiple' or 'unknown'>",
      "recommendation": "<specific actionable fix>"
    }}
  ],
  "compliant_areas": ["<area that looks good>"]
}}
Be specific and cite file names when visible. Only report what is evidenced in the chunks."""


def _sample_chunks(store, sample_size: int) -> List[Dict[str, Any]]:
    total = store.count if store else 0
    if total == 0:
        return []

    all_data = store.collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []
    items = [{"content": d or "", "metadata": m or {}} for d, m in zip(docs, metas)]

    if len(items) <= sample_size:
        return items

    # Stratified by file so every file gets at least one chunk
    by_file: Dict[str, list] = {}
    for it in items:
        fp = (it["metadata"] or {}).get("document_path", "unknown")
        by_file.setdefault(fp, []).append(it)

    files = sorted(by_file.keys())
    per_file = max(1, sample_size // max(1, len(files)))
    sampled = []
    for fp in files:
        chunk_list = by_file[fp]
        step = max(1, len(chunk_list) // per_file)
        sampled.extend(chunk_list[::step][:per_file])

    return sampled[:sample_size]


def run_compliance_scan(store, framework: str, sample_size: int = 30) -> Dict[str, Any]:
    if framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework '{framework}'. Choose from: {list(FRAMEWORKS.keys())}")

    start = time.perf_counter()
    fw = FRAMEWORKS[framework]
    chunks = _sample_chunks(store, sample_size)

    if not chunks:
        return {
            "framework": framework, "framework_full_name": fw["full_name"],
            "summary": "No documents indexed.", "risk_score": 0,
            "issues": [], "compliant_areas": [],
            "sampled_chunks": 0, "total_chunks": 0, "duration_ms": 0,
        }

    # Build code context
    parts = []
    for i, ch in enumerate(chunks, 1):
        path = (ch["metadata"] or {}).get("document_path", "unknown")
        content = (ch["content"] or "").strip()[:600]
        parts.append(f"[Chunk {i} | {path}]\n{content}")
    code_context = "\n\n".join(parts)

    focus_areas = "\n".join(f"  - {a}" for a in fw["focus_areas"])
    prompt = _PROMPT.format(
        fw_name=fw["name"], fw_full=fw["full_name"],
        focus_areas=focus_areas,
        code_context=code_context[:14000],
    )

    from core import llm_client
    raw = llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        system="You are a compliance expert. Return only valid JSON with no markdown fences.",
        max_tokens=2048,
        temperature=0.1,
        stream=False,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Compliance JSON parse error: %s | raw: %s", e, raw[:300])
        parsed = {"summary": "Parse error: " + raw[:200], "risk_score": 50, "issues": [], "compliant_areas": []}

    return {
        "framework": framework,
        "framework_full_name": fw["full_name"],
        "summary": parsed.get("summary", ""),
        "risk_score": max(0, min(100, int(parsed.get("risk_score", 50)))),
        "issues": parsed.get("issues", []),
        "compliant_areas": parsed.get("compliant_areas", []),
        "sampled_chunks": len(chunks),
        "total_chunks": store.count,
        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
    }
