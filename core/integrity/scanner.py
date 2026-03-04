"""
Knowledge Integrity & Risk Radar

This module scans the indexed knowledge base and produces:
- contradictions
- blind spots
- resilience gaps
- documentation drift
- overall health score + recommendations

Design goals:
- deterministic first (regex + heuristics)
- zero new heavy dependencies
- safe defaults for large corpora (sampling)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import settings


# ---------------------------
# Data models (plain dicts externally)
# ---------------------------

@dataclass
class Evidence:
    file: str
    lines: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {"file": self.file, "lines": self.lines, "snippet": self.snippet}


@dataclass
class Issue:
    type: str  # contradiction | blind_spot | resilience_gap | drift
    severity: str  # low | medium | high | critical
    title: str
    description: str
    evidence: List[Evidence]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "evidence": [e.to_dict() for e in self.evidence],
            "recommendation": self.recommendation,
        }


# ---------------------------
# Corpus helpers
# ---------------------------

def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _safe_snippet(text: str, max_len: int = 240) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return (t[: max_len - 1] + "…") if len(t) > max_len else t


def _load_corpus(store, max_chunks: int) -> List[Dict[str, Any]]:
    """
    Returns list of {"content": str, "metadata": dict}
    Sampling strategy:
      - If total <= max_chunks: return all
      - Else: sample evenly per file to keep coverage
    """
    total = store.count if store else 0
    if total == 0:
        return []

    all_data = store.collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []
    items = [{"content": d or "", "metadata": (m or {})} for d, m in zip(docs, metas)]

    if len(items) <= max_chunks:
        return items

    # group by file
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        p = (it["metadata"] or {}).get("document_path", "") or "unknown"
        by_file.setdefault(p, []).append(it)

    files = sorted(by_file.keys())
    per_file = max(1, max_chunks // max(1, len(files)))
    sampled: List[Dict[str, Any]] = []
    for fp in files:
        chunk_list = by_file[fp]
        # deterministic spread
        step = max(1, len(chunk_list) // per_file)
        sampled.extend(chunk_list[::step][:per_file])

    # If still over (due to rounding), trim
    return sampled[:max_chunks]


# ---------------------------
# Claim extraction (for contradictions)
# ---------------------------

_NUMVAL = re.compile(
    r"(?P<val>\b\d{1,6}(?:\.\d{1,3})?\b)\s*(?P<unit>ms|s|sec|secs|seconds|minutes|mins|min|hours|hrs|days|d|%|percent|kb|mb|gb|tb)?\b",
    re.IGNORECASE,
)

_KEYVAL = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9 _/\-]{2,60})\s*(?:=|:|is|are|should be|must be)\s*(?P<value>[^.\n]{1,80})",
    re.IGNORECASE,
)

_BOOL = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9 _/\-]{2,60})\s+(?P<polarity>must not|should not|must|should)\b",
    re.IGNORECASE,
)

_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with", "by", "at", "is", "are", "be", "as"}


def _norm_key(k: str) -> str:
    k = k.lower()
    k = re.sub(r"[^a-z0-9 ]+", " ", k)
    parts = [p for p in k.split() if p and p not in _STOP]
    return " ".join(parts[:8])


def _extract_claims(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (key, value) pairs.
    Values are normalized numeric+unit when present to increase detectability.
    """
    claims: List[Tuple[str, str]] = []

    for m in _KEYVAL.finditer(text):
        key = _norm_key(m.group("key"))
        val_raw = m.group("value").strip()
        if not key:
            continue
        # Normalize numeric
        num = _NUMVAL.search(val_raw)
        if num:
            unit = (num.group("unit") or "").lower()
            val = f"{num.group('val')}{unit}".strip()
        else:
            val = re.sub(r"\s+", " ", val_raw.lower())[:40]
        claims.append((key, val))

    for m in _BOOL.finditer(text):
        key = _norm_key(m.group("key"))
        pol = m.group("polarity").lower().replace(" ", "_")
        if key:
            claims.append((key, pol))

    # light dedupe
    seen = set()
    out = []
    for k, v in claims:
        if (k, v) not in seen:
            seen.add((k, v))
            out.append((k, v))
    return out


# ---------------------------
# Issue detectors
# ---------------------------

def detect_contradictions(corpus: List[Dict[str, Any]], max_examples: int = 3) -> List[Issue]:
    """
    Detects conflicting claims across documents.
    """
    buckets: Dict[str, Dict[str, List[Evidence]]] = {}  # key -> value -> evidences

    for it in corpus:
        text = it["content"] or ""
        meta = it["metadata"] or {}
        fp = meta.get("document_path", "") or "unknown"
        lines = f"{meta.get('start_line','?')}-{meta.get('end_line','?')}"
        for k, v in _extract_claims(text):
            ev = Evidence(file=fp, lines=lines, snippet=_safe_snippet(text))
            buckets.setdefault(k, {}).setdefault(v, []).append(ev)

    issues: List[Issue] = []
    for k, values in buckets.items():
        # Need at least 2 distinct values from 2+ evidences to be meaningful
        if len(values) < 2:
            continue

        # prefer numeric/boolean conflicts
        distinct = list(values.keys())
        # score severity
        sev = "medium"
        if any(v.endswith(("ms", "s", "sec", "seconds", "min", "mins", "hours", "days", "%")) or v.replace(".", "", 1).isdigit() for v in distinct):
            sev = "high"
        if any(v in ("must", "must_not", "should", "should_not") for v in distinct):
            sev = "high"

        evs: List[Evidence] = []
        for v in distinct[:2]:
            evs.extend(values[v][:1])
        title = f"Conflicting statements detected: “{k}”"
        description = f"Multiple values/policies found for the same topic ({', '.join(distinct[:4])}). This can cause incorrect guidance and operational risk."
        recommendation = "Consolidate this topic into a single source of truth (one policy/runbook section), and update or deprecate conflicting documents."
        issues.append(Issue(
            type="contradiction",
            severity=sev,
            title=title,
            description=description,
            evidence=evs[:max_examples],
            recommendation=recommendation,
        ))

    # cap
    return issues[: settings.integrity_max_issues]


def detect_blind_spots(corpus: List[Dict[str, Any]]) -> List[Issue]:
    """
    Detect missing essential topics across the knowledge base.
    """
    full_text = "\n".join((it.get("content") or "") for it in corpus).lower()

    # Essential topics (simple heuristic)
    required_topics = [
        ("disaster recovery", ["disaster recovery", "dr plan", "drp", "failover region", "region failover"]),
        ("backups & restore", ["backup", "restore", "retention", "snapshot"]),
        ("monitoring & alerting", ["monitoring", "metrics", "alert", "pager", "on-call", "escalation"]),
        ("rollbacks", ["rollback", "roll back", "revert", "feature flag", "canary"]),
        ("timeouts & retries", ["timeout", "retry", "backoff", "circuit breaker"]),
        ("rate limiting", ["rate limit", "throttle", "quota"]),
        ("security access control", ["access control", "rbac", "least privilege", "permission", "authz"]),
        ("encryption", ["encryption", "tls", "https", "at rest"]),
    ]

    issues: List[Issue] = []
    for name, needles in required_topics:
        if not any(n in full_text for n in needles):
            issues.append(Issue(
                type="blind_spot",
                severity="high" if name in ("disaster recovery", "backups & restore", "monitoring & alerting") else "medium",
                title=f"Missing documentation: {name}",
                description=f"The knowledge base does not appear to describe {name}. Teams may be unprepared during incidents or audits.",
                evidence=[],
                recommendation=f"Add a dedicated section/runbook covering {name} with owners, procedures, and validation steps.",
            ))

    return issues[: settings.integrity_max_issues]


def detect_resilience_gaps(corpus: List[Dict[str, Any]]) -> List[Issue]:
    """
    Detect likely resilience gaps based on dependency mentions without corresponding mitigation language.
    """
    # If a dependency/service is mentioned, expect resilience keywords nearby in at least some documents.
    dependencies = {
        "database": ["postgres", "mysql", "database", "rds"],
        "cache": ["redis", "memcached", "cache"],
        "queue": ["kafka", "rabbitmq", "sqs", "queue"],
        "object storage": ["s3", "gcs", "blob storage", "object storage"],
    }
    mitigations = ["replica", "replication", "cluster", "multi-az", "failover", "fallback", "degraded", "circuit breaker", "retry", "timeout", "rate limit"]

    # Build per-dependency presence
    text_all = "\n".join((it.get("content") or "") for it in corpus).lower()

    issues: List[Issue] = []
    for dep_name, needles in dependencies.items():
        if any(n in text_all for n in needles):
            # expect mitigations
            if not any(m in text_all for m in mitigations):
                issues.append(Issue(
                    type="resilience_gap",
                    severity="high" if dep_name in ("database", "cache") else "medium",
                    title=f"Resilience gap: {dep_name} mentioned without mitigation details",
                    description=f"The documents reference {dep_name} components, but do not clearly describe redundancy/failover/timeouts/retries. This increases incident risk.",
                    evidence=[],
                    recommendation=f"Document {dep_name} resilience strategy (timeouts, retries, failover, backup/restore, capacity limits) and link it from the main runbook.",
                ))

    return issues[: settings.integrity_max_issues]


def detect_drift(store, corpus: List[Dict[str, Any]], previous_fingerprints: Optional[Dict[str, str]] = None) -> Tuple[List[Issue], Dict[str, str]]:
    """
    Detect drift by comparing current per-file fingerprints to previous fingerprints.

    Fingerprint is SHA256 over a stable summary of each file (sampled chunks + counts).
    """
    previous_fingerprints = previous_fingerprints or {}

    # build per-file stable text
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for it in corpus:
        fp = (it.get("metadata") or {}).get("document_path", "") or "unknown"
        by_file.setdefault(fp, []).append(it)

    current: Dict[str, str] = {}
    for fp, items in by_file.items():
        # stable sort by start_line if possible
        def _k(x):
            m = x.get("metadata") or {}
            try:
                return int(m.get("start_line") or 0)
            except Exception:
                return 0
        items_sorted = sorted(items, key=_k)
        sample_text = "\n".join((x.get("content") or "")[:400] for x in items_sorted[:10])
        payload = f"{fp}|{len(items_sorted)}|{sample_text}"
        current[fp] = _hash_text(payload)

    issues: List[Issue] = []
    for fp, h in current.items():
        prev = previous_fingerprints.get(fp)
        if prev and prev != h:
            issues.append(Issue(
                type="drift",
                severity="medium",
                title=f"Documentation drift detected: {fp}",
                description="This document's content changed significantly since the last integrity scan. Ensure dependent runbooks/policies remain consistent.",
                evidence=[Evidence(file=fp, lines="—", snippet="Fingerprint changed since last scan")],
                recommendation="Review recent updates, validate downstream references, and re-run contradiction checks for related documents.",
            ))

    return issues[: settings.integrity_max_issues], current


# ---------------------------
# Scoring
# ---------------------------

def score_health(issues: List[Issue]) -> Dict[str, Any]:
    weights = {
        "critical": 18,
        "high": 12,
        "medium": 7,
        "low": 3,
    }
    by_type = {"contradiction": 0, "blind_spot": 0, "resilience_gap": 0, "drift": 0}
    penalty = 0
    for iss in issues:
        by_type[iss.type] = by_type.get(iss.type, 0) + 1
        penalty += weights.get(iss.severity, 7)

    score = max(0, min(100, 100 - penalty))
    band = "excellent" if score >= 90 else "good" if score >= 75 else "fair" if score >= 55 else "poor"
    return {"score": score, "band": band, "counts": by_type, "penalty": penalty}


def build_recommendations(issues: List[Issue], max_items: int = 6) -> List[str]:
    recs = []
    seen = set()
    for i in issues:
        r = i.recommendation
        if r and r not in seen:
            seen.add(r)
            recs.append(r)
        if len(recs) >= max_items:
            break
    if not recs:
        recs.append("No major risks detected. Keep docs current by running periodic integrity scans.")
    return recs


# ---------------------------
# Public API
# ---------------------------

def run_integrity_scan(store, previous_fingerprints: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    started = time.time()
    max_chunks = getattr(settings, "integrity_scan_max_chunks", 1200)
    corpus = _load_corpus(store, max_chunks=max_chunks)

    issues: List[Issue] = []
    issues.extend(detect_contradictions(corpus))
    issues.extend(detect_blind_spots(corpus))
    issues.extend(detect_resilience_gaps(corpus))

    drift_issues, fingerprints = detect_drift(store, corpus, previous_fingerprints=previous_fingerprints)
    issues.extend(drift_issues)

    # sort: severity then type
    sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    issues_sorted = sorted(issues, key=lambda x: (sev_rank.get(x.severity, 9), x.type, x.title))

    health = score_health(issues_sorted)
    recs = build_recommendations(issues_sorted)

    out = {
        "started_at": started,
        "finished_at": time.time(),
        "duration_ms": round((time.time() - started) * 1000, 1),
        "health": health,
        "issues": [i.to_dict() for i in issues_sorted[: settings.integrity_max_issues]],
        "recommendations": recs,
        "fingerprints": fingerprints,
        "sampled_chunks": len(corpus),
        "total_chunks": store.count if store else 0,
    }
    return out
