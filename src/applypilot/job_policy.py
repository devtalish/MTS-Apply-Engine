"""Hard job-policy gates for Muhammad Talish's remote job engine.

This module contains deterministic rules that run before LLM scoring and before
auto-apply. The goal is to protect application quality: Pakistan-compatible,
legitimate technical work, with a primary target of 5-14 hours/week and a
secondary ceiling of 20 hours/week.

Ambiguous postings are not guessed into eligibility; the LLM can score them,
but
hard exclusions are applied when the posting explicitly conflicts with the
frozen search policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PRIMARY_MAX_HOURS = 14
SECONDARY_MAX_HOURS = 20


@dataclass(frozen=True)
class PolicyDecision:
    eligible: bool
    reason: str = ""
    category: str = "eligible"
    hours_per_week: float | None = None


_HARD_TITLE = re.compile(
    r"\b(?:senior|staff|principal|lead|head|director|vice president|vp|chief|cto)\b"
    r"|\bintern(?:ship)?\b|\bco-op\b|\bgraduate\b|\btrainee\b|\bapprentice\b"
    r"|\bfresher\b|\bnew[- ]grad\b"
    r"|\brecruit(?:er|ing)\b|\btalent acquisition\b|\btalent sourc(?:er|ing)\b"
    r"|\baccount executive\b|\baccount manager\b|\bsales engineer\b"
    r"|\bpre[- ]?sales\b|\bcustomer success\b|\bcustomer support\b"
    r"|\bux designer\b|\bui designer\b|\bproduct designer\b|\bgraphic designer\b"
    r"|\bandroid engineer\b|\bios engineer\b|\bmobile engineer\b"
    r"|\bsalesforce developer\b|\bapex developer\b|\bmainframe\b|\bcobol\b"
    r"|\bcommission[- ]only\b",
    re.I,
)

_GEO_REJECT = re.compile(
    r"\b(?:us|usa|united states|canada|uk|united kingdom|europe|eu|india|"
    r"germany|france|spain|italy|ireland|netherlands|australia|new zealand|"
    r"singapore|japan|brazil|mexico)\b.*\bonly\b"
    r"|\bonly\b.*\b(?:us|usa|united states|canada|uk|united kingdom|"
    r"europe|eu|india|germany|france|spain|italy|ireland|netherlands|"
    r"australia|new zealand|singapore|japan|brazil|mexico)\b"
    r"|\bremote\s*[—\-]\s*(?:us|usa|united states|canada|uk|united kingdom|europe|eu|india)\b"
    r"|\bremote\s*\(?(?:us|usa|united states|canada|uk|united kingdom|"
    r"europe|eu|india)\)?\b",
    re.I,
)

_US_AUTH = re.compile(
    r"\b(?:must|need|requires?|requirement)\b[^.\n]{0,100}\b(?:"
    r"authorized|eligible|right)\s+to\s+work\b[^.\n]{0,100}\b(?:us|usa|"
    r"united states)\b"
    r"|\b(?:us|usa|united states)\s+(?:work authorization|residency|residents?)\b"
    r"|\bmust\s+(?:be|reside|live)\s+in\s+(?:the\s+)?(?:us|usa|united states)\b",
    re.I,
)

_EXPLICIT_PART_TIME = re.compile(
    r"\bpart[- ]time\b|\bcontract(?:or)?\b|\bfreelance\b|\basync(?:hronous)?\b",
    re.I,
)

_FULL_TIME_ONLY = re.compile(
    r"\bfull[- ]time only\b|\bfull[- ]time position\b|\bfull[- ]time role\b"
    r"|\b40\s*(?:hours?|hrs?)\s*(?:per|/|a)\s*week\b"
    r"|\b(?:30|35|36|37\.5|38|40)\s*(?:hours?|hrs?)\s*(?:per|/|a)\s*week\b",
    re.I,
)

_HOURS = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:per|/|a)\s*week\b"
    r"|\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:per|/|a)\s*week\b",
    re.I,
)

_BAD_MONEY = re.compile(
    r"\b(?:unpaid|no pay|pay to work|training fee|registration fee|"
    r"upfront fee|membership fee|crypto payment|commission[- ]only)\b",
    re.I,
)

_TECH_ROLE = re.compile(
    r"\b(?:software|software development|developer|engineering|engineer|"
    r"backend|full[- ]stack|frontend|python|typescript|javascript|react|"
    r"ai|machine learning|automation|api|integration|platform|devops|"
    r"qa automation|implementation)\b",
    re.I,
)


def _text(job: dict[str, Any]) -> str:
    return " ".join(
        str(job.get(k) or "")
        for k in ("title", "location", "description", "full_description", "salary", "site", "company")
    ).strip()


def _hours(job_text: str) -> float | None:
    matches = list(_HOURS.finditer(job_text))
    if not matches:
        return None
    values: list[float] = []
    for m in matches:
        if m.group(1) and m.group(2):
            values.append((float(m.group(1)) + float(m.group(2))) / 2)
        else:
            value = m.group(3)
            if value:
                values.append(float(value))
    return min(values) if values else None


def evaluate_job(job: dict[str, Any]) -> PolicyDecision:
    title = str(job.get("title") or "").strip()
    location = str(job.get("location") or "").strip()
    desc = str(job.get("full_description") or job.get("description") or "")
    combined = " ".join((title, location, desc, str(job.get("salary") or ""), str(job.get("site") or "")))

    if not title:
        return PolicyDecision(False, "missing job title", "invalid")

    if _HARD_TITLE.search(title):
        return PolicyDecision(False, f"non-target seniority/role in title: {title}", "role")

    if _GEO_REJECT.search(f"{title} {location}"):
        return PolicyDecision(False, f"explicit geography excludes Pakistan: {location or title}", "geo")

    if _US_AUTH.search(desc):
        return PolicyDecision(False, "posting requires US residency/work authorization", "geo")

    if _BAD_MONEY.search(combined):
        return PolicyDecision(False, "unpaid, pay-to-work, or commission-only compensation", "compensation")

    hours = _hours(combined)
    if hours is not None and hours > SECONDARY_MAX_HOURS:
        return PolicyDecision(False, f"explicit workload exceeds 20 hours/week: {hours:g}", "hours", hours)

    if _FULL_TIME_ONLY.search(combined) and not _EXPLICIT_PART_TIME.search(combined):
        return PolicyDecision(False, "full-time-only workload conflicts with 5-20 hour target", "hours", hours)

    if not _TECH_ROLE.search(title + " " + desc[:5000]):
        return PolicyDecision(False, "role is not substantially technical", "role")

    # No country is inferred from a company headquarters. "Remote", "worldwide",
    # and global postings remain eligible for LLM evaluation.
    return PolicyDecision(True, "", "eligible", hours)


def hard_gate(job: dict[str, Any]) -> PolicyDecision:
    """Alias used by application-time code to make the same decision deterministic."""
    return evaluate_job(job)
