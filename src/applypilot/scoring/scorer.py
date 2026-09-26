"""Job-fit scoring for Talish's remote-first job engine."""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from applypilot.config import RESUME_PATH, load_profile
from applypilot.database import get_connection, get_jobs_by_stage, write_with_retry
from applypilot.job_policy import evaluate_job
from applypilot.llm import get_client

log = logging.getLogger(__name__)

SCORE_PROMPT_TEMPLATE = """You are the job-fit scoring engine for Muhammad Talish.

Candidate: Pakistan-based CS student and Founder & Product/Technical Lead at Velnaro.
He has hands-on project-building experience and is targeting legitimate remote technical work.
Do not convert projects or founder work into fake employment history.

TARGET WORK:
- Primary: recurring part-time / async / flexible contractor work, approximately 5–14 hours/week.
- Secondary: up to 20 hours/week only when genuinely flexible.
- Exclude fixed full-time 30–40 hour/week roles.
- Target compensation is approximately USD $1,000–$2,000/month; undisclosed salary is not an automatic rejection.

TARGET ROLES:
software developer/engineer, backend, full-stack, Python, AI/applied AI, automation, API/integration,
platform/cloud/DevOps, QA automation, technical implementation.

CORE EVIDENCE:
Python, JavaScript, TypeScript, React, Node.js, SQL/Supabase, REST APIs, AI/LLM integrations,
automation/workflows, browser automation, software architecture, Git/GitHub, deployment.

HARD RULES:
1. Pakistan must be a permitted work location. Do not infer a restriction from company headquarters.
2. Explicit US/Canada/UK/EU/India/etc. location, residency, citizenship, or work-authorization restrictions exclude the job.
3. Senior/Staff/Principal/Lead/Head/Director/VP/C-level requirements are not appropriate.
4. Internships, unpaid work, commission-only work, pay-to-work schemes, and non-technical roles are excluded.
5. Explicit workload above 20 hours/week is excluded; 5–14 hours/week is the primary target.
6. Projects count as technical evidence but never as conventional employment.

Score technical fit, seniority, location compatibility, workload fit, compensation/stability, project/domain fit,
and application feasibility. A strong technical match with the wrong workload or location must not receive a high score.

OUTPUT EXACTLY:
ELIGIBILITY: [eligible|non_us_only]
SCORE: [1-10]
KEYWORDS: [comma-separated matching ATS keywords]
REASONING: [2-4 concise sentences; mention workload/location/seniority and important gaps]
Never invent facts.
"""

_INELIGIBLE_TITLE_PATTERNS = re.compile(
    r"\b(?:senior|staff|principal|lead|head|director|vice president|vp|chief|cto)\b"
    r"|\bintern(?:ship)?\b|\bco-op\b|\bgraduate\b|\btrainee\b|\bapprentice\b"
    r"|\bfresher\b|\bnew[- ]grad\b"
    r"|\brecruit(?:er|ing)\b|\btalent acquisition\b|\btalent sourc(?:er|ing)\b"
    r"|\baccount executive\b|\baccount manager\b|\bsales engineer\b"
    r"|\bpre[- ]?sales\b|\bcustomer success\b|\bcustomer support\b"
    r"|\bux designer\b|\bui designer\b|\bproduct designer\b|\bgraphic designer\b"
    r"|\bandroid engineer\b|\bios engineer\b|\bmobile engineer\b"
    r"|\bsalesforce developer\b|\bapex developer\b|\bmainframe\b|\bcobol\b"
    r"|\bcommission[- ]only\b", re.I)

_INELIGIBLE_LOCATION_PATTERNS = re.compile(
    r"\b(?:EMEA|APAC|Europe|EU|Canada|Germany|Netherlands|France|Spain|Italy|Poland|Ukraine|Portugal|"
    r"Ireland|Denmark|Sweden|Norway|Finland|Belgium|Switzerland|Austria|Romania|Hungary|Croatia|"
    r"Greece|Bulgaria|Serbia|Slovakia|Slovenia|Estonia|Latvia|Lithuania|India|Singapore|Japan|"
    r"Vietnam|Thailand|Philippines|Indonesia|Korea|Taiwan|Hong Kong|China|Bangladesh|Malaysia|"
    r"Brazil|Mexico|Argentina|Chile|Colombia|Peru|Uruguay|Egypt|Nigeria|Kenya|South Africa|"
    r"Israel|Turkey|Türkiye|UAE|Saudi Arabia|Australia|New Zealand)\b", re.I)

_INELIGIBLE_DESC_PATTERNS = re.compile(
    r"remote\s*[\(\-—–]\s*(EMEA|Europe|EU|UK|United\s+Kingdom|Germany|India|Canada|Ireland|"
    r"Netherlands|Brazil|Mexico|Australia|New\s+Zealand)"
    r"|based\s+in\s+(the\s+)?(Europe|EU|UK|United\s+Kingdom|Canada|India|Ireland|Germany|"
    r"Netherlands|France|Spain|Italy|Brazil|Mexico|Australia|New\s+Zealand|Singapore|Japan)"
    r"|right\s+to\s+work\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|EU|European\s+Union)"
    r"|requires?\s+(the\s+)?right\s+to\s+work\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|EU)",
    re.I)
_DESC_SCAN_CHARS = 10000


def _check_ineligible(job: dict) -> str | None:
    title = job.get("title") or ""
    location = job.get("location") or ""
    desc = (job.get("full_description") or job.get("description") or "")[:_DESC_SCAN_CHARS]
    if _INELIGIBLE_TITLE_PATTERNS.search(title):
        return f"non-target seniority/role: {title}"
    if _INELIGIBLE_LOCATION_PATTERNS.search(location):
        return f"location excludes Pakistan: {location}"
    m = _INELIGIBLE_DESC_PATTERNS.search(desc)
    if m:
        return f"explicit geographic/work-right restriction: {m.group(0)[:100]}"
    decision = evaluate_job(job)
    if not decision.eligible:
        return decision.reason
    return None


def _parse_score_response(response: str) -> dict:
    score = 0
    keywords = ""
    reasoning = response
    eligibility = "eligible"
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("ELIGIBILITY:"):
            value = line.split(":", 1)[1].strip().lower()
            eligibility = "non_us_only" if "non_us" in value or ("non" in value and "us" in value) else "eligible"
        elif line.startswith("SCORE:"):
            try:
                score = max(1, min(10, int(re.search(r"\d+", line).group())))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.split(":", 1)[1].strip()
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return {"score": score, "keywords": keywords, "reasoning": reasoning, "eligibility": eligibility}


def _build_candidate_summary(profile: dict) -> str:
    exp = profile.get("experience", {})
    boundary = profile.get("skills_boundary", {})
    parts = [
        f"Current role: {exp.get('current_job_title', 'Founder & Product/Technical Lead')}.",
        f"Hands-on experience: {exp.get('years_of_experience_total', 'project-based')}.",
        "Location: Pakistan.",
        "Work target: 5–14 hours/week primary; up to 20 hours/week if flexible.",
    ]
    for category, skills in boundary.items():
        if isinstance(skills, list) and skills:
            parts.append(f"{category}: {', '.join(str(x) for x in skills[:15])}.")
    return " ".join(parts)


def score_job(resume_text: str, job: dict, profile: dict | None = None) -> dict:
    if profile is None:
        profile = load_profile()
    reason = _check_ineligible(job)
    if reason:
        return {"score": 1, "keywords": "", "reasoning": f"Hard policy exclusion: {reason}.",
                "eligibility": "non_us_only"}
    try:
        summary = _build_candidate_summary(profile)
        job_text = (
            f"TITLE: {job.get('title','')}\nCOMPANY: {job.get('company') or job.get('site','')}\n"
            f"LOCATION: {job.get('location','N/A')}\nDESCRIPTION:\n"
            f"{(job.get('full_description') or '')[:10000]}"
        )
        response = get_client().chat([
            {"role": "system", "content": SCORE_PROMPT_TEMPLATE.format(candidate_summary=summary)},
            {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
        ], max_tokens=4096, temperature=0.1)
        parsed = _parse_score_response(response)
        # Never allow the model to override deterministic hard gates.
        if parsed["eligibility"] == "eligible":
            parsed["eligibility"] = "eligible"
        return parsed
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": None, "keywords": "", "reasoning": "", "eligibility": None, "error": f"LLM error: {e}"}


MAX_SCORE_RETRIES = 5


def _score_backoff_minutes(retry_count: int) -> int:
    return min(5 * (4 ** retry_count), 24 * 60)


def _flush_score_batch(conn, batch: list[dict], now: str) -> None:
    from applypilot.database import transition_state
    min_score = 8
    for r in batch:
        if r["score"] is not None:
            eligibility = r.get("eligibility") or "eligible"
            conn.execute(
                "UPDATE jobs SET fit_score=?, score_reasoning=?, scored_at=?, eligibility=?, "
                "score_error=NULL, score_attempts=0, score_next_retry_at=NULL WHERE url=?",
                (r["score"], f"{r['keywords']}\n{r['reasoning']}", now, eligibility, r["url"]),
            )
            if eligibility == "non_us_only":
                transition_state(conn, r["url"], "archived", reason="hard eligibility exclusion",
                                 metadata={"score": r["score"], "eligibility": eligibility}, force=True)
            else:
                transition_state(conn, r["url"], "scored" if r["score"] >= min_score else "low_score",
                                 reason=f"scored {r['score']}/10",
                                 metadata={"score": r["score"], "eligibility": eligibility})
        else:
            row = conn.execute("SELECT COALESCE(score_attempts,0) FROM jobs WHERE url=?", (r["url"],)).fetchone()
            retry_count = row[0] if row else 0
            if retry_count >= MAX_SCORE_RETRIES:
                conn.execute("UPDATE jobs SET score_error=?,score_attempts=?,score_next_retry_at=NULL,scored_at=? WHERE url=?",
                             (r["error"], retry_count + 1, now, r["url"]))
                transition_state(conn, r["url"], "score_failed",
                                 reason=f"LLM failed after {retry_count+1} attempts", metadata={"error": r["error"]})
            else:
                delay = _score_backoff_minutes(retry_count)
                next_retry = (datetime.now(timezone.utc) + timedelta(minutes=delay)).isoformat()
                conn.execute("UPDATE jobs SET score_error=?,score_attempts=?,score_next_retry_at=?,scored_at=? WHERE url=?",
                             (r["error"], retry_count + 1, next_retry, now, r["url"]))


def run_scoring(limit: int = 0, rescore: bool = False, workers: int = 1, max_age_days: int | None = None) -> dict:
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", max_age_days=max_age_days, limit=limit)
    if not jobs:
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}
    jobs = [dict(j) for j in jobs]
    t0 = time.time()
    completed = errors = 0
    batch: list[dict] = []
    def one(job):
        try:
            result = score_job(resume_text, job)
            result["url"] = job["url"]
            return result
        except Exception as e:
            return {"score": None, "keywords": "", "reasoning": "", "eligibility": None,
                    "error": f"Unexpected: {e}", "url": job["url"]}
    def flush():
        nonlocal batch
        if batch:
            write_with_retry(conn, _flush_score_batch, conn, batch, datetime.now(timezone.utc).isoformat())
            batch = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, j) for j in jobs]
            for f in as_completed(futures):
                r = f.result(); batch.append(r); completed += 1; errors += r["score"] is None
                if len(batch) >= 25: flush()
    else:
        for j in jobs:
            r = one(j); batch.append(r); completed += 1; errors += r["score"] is None
            if len(batch) >= 25: flush()
    flush()
    elapsed = time.time() - t0
    dist = conn.execute("SELECT fit_score,COUNT(*) FROM jobs WHERE fit_score IS NOT NULL GROUP BY fit_score ORDER BY fit_score DESC").fetchall()
    return {"scored": completed, "errors": errors, "elapsed": elapsed,
            "distribution": [(r[0], r[1]) for r in dist]}
