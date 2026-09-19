"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from applypilot.config import RESUME_PATH, load_profile
from applypilot.database import get_connection, get_jobs_by_stage, write_with_retry
from applypilot.llm import get_client

log = logging.getLogger(__name__)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT_TEMPLATE = """You are the job-fit scoring engine for Muhammad Talish.

Candidate context:
{candidate_summary}

The candidate is based in Pakistan and is seeking legitimate remote technical employment. Target compensation is approximately USD $1,000–$2,000+ per month, but do not reject a job solely because salary is undisclosed.

PRIMARY TARGET ROLES:
- Software Engineer / Software Developer
- Backend Engineer / Backend Developer
- Full-Stack Engineer / Full-Stack Developer
- Python Engineer / Python Developer
- AI Engineer / Applied AI Engineer
- AI/Automation Engineer / Automation Engineer
- API / Integration Engineer
- Platform / Cloud / DevOps Engineer
- QA Automation Engineer
- Technical Developer / Implementation Engineer

CORE SKILLS TO MATCH WHEN PRESENT IN THE RESUME:
Python, JavaScript, TypeScript, React, Next.js, FastAPI, Flask, APIs, REST, AI/LLM integrations, automation, browser/workflow automation, backend systems, Supabase, PostgreSQL, Git, Docker, cloud deployment, software architecture, full-stack development, AI-assisted development.

==================================================
1. PAKISTAN / REMOTE ELIGIBILITY
==================================================
Determine whether the actual worker location requirement permits someone working from Pakistan.

ELIGIBLE examples:
- Remote / Worldwide / Global remote / Work from anywhere
- Remote with no country restriction
- Remote and explicitly open to Pakistan
- Remote across multiple countries including Pakistan

INELIGIBLE examples:
- US-only when US work authorization or US physical presence is required
- Canada-only
- UK-only
- EU-only / Europe-only
- India-only
- Any other country/region restriction that clearly excludes Pakistan
- Required citizenship, residency, or work authorization that the candidate does not have
- Onsite/hybrid in another country with no genuine remote option

Do NOT reject a job simply because the employer is American, Canadian, British, European, etc.
Do NOT assume that a timezone mention alone excludes Pakistan. Only treat timezone as a hard issue when the posting clearly requires working from a location/timezone incompatible with the candidate's circumstances.
If the description is ambiguous, explain the uncertainty rather than inventing a restriction.

Output ELIGIBILITY: non_us_only for a clear geographic/work-authorization restriction that excludes Pakistan. Otherwise output ELIGIBILITY: eligible.

==================================================
2. ROLE FIT
==================================================
Judge the actual responsibilities, not just the title. Technical roles should be prioritized.

Strong matches include software/backend/full-stack/AI/automation/API/integration/platform/cloud/DevOps engineering.

Usually reject or score very low:
- Sales / marketing / recruiting / HR
- Pure customer support / customer success
- Data entry / virtual assistant / microtasks
- Commission-only work
- Unpaid work
- Pay-to-work schemes
- Pure non-technical design roles

Solutions/implementation/technical customer-facing roles may still be considered when the actual work is substantially technical.

==================================================
3. TECHNICAL FIT
==================================================
Compare REQUIRED qualifications against the actual resume.

- Required skills matter more than nice-to-have skills.
- Give credit for transferable engineering skills rather than demanding an exact framework match.
- Do not invent experience with a technology that is absent from the resume.
- Projects and practical builds count as evidence of technical ability, but must not be converted into fake employment history.
- Highly specialized fields such as embedded hardware, semiconductor engineering, medical research, advanced quantitative finance, or specialized security clearance should score lower unless directly supported by the resume.

==================================================
4. EXPERIENCE / SENIORITY
==================================================
Use the resume's actual experience level. Do not assume the candidate is Senior/Staff/Principal.

Junior and mid-level roles are valid targets when the technical requirements fit.

If a posting explicitly requires 7+ years, 10+ years, Staff/Principal leadership, or another level not supported by the resume, reduce the score substantially.
Do not penalize the candidate merely for being young.
Do not fabricate years of experience.

==================================================
5. COMPENSATION
==================================================
Target is approximately USD $1,000–$2,000+ per month.

- Prefer legitimate salaried or normal professional contract compensation that can realistically meet the target.
- If salary is undisclosed, do not automatically reject; mark it as unknown in reasoning.
- Reject unpaid, pay-to-work, commission-only, or obviously deceptive compensation.
- Never invent a salary figure.

==================================================
6. LEGITIMACY / RISK
==================================================
Look for requests for money, training fees, crypto payments, equipment purchases, unrealistic income promises, pyramid/MLM structures, or an unidentifiable employer.
A normal legitimate employer with an undisclosed salary is not automatically suspicious.

==================================================
7. LEARNING VALUE
==================================================
Learning value is secondary to actual fit. AI, automation, backend, cloud, APIs, architecture, and modern engineering workflows are useful positives, but learning potential must never rescue a fundamentally bad-fit job.

==================================================
SCORING
==================================================
10 = Exceptional fit with strong technical alignment, realistic seniority, Pakistan-compatible location, and minimal gaps.
8-9 = Strong fit and realistic application opportunity with manageable gaps.
7 = Reasonable fit with noticeable but potentially manageable gaps.
5-6 = Partial fit with important gaps.
3-4 = Weak fit or major mismatch.
1-2 = Clearly unsuitable, non-technical, ineligible geographically, unpaid/commission-only, or otherwise unacceptable.

Never inflate a score because the title contains AI, Software, Engineer, or Developer. Judge the actual description.

OUTPUT EXACTLY THESE FOUR LINES:
ELIGIBILITY: [eligible|non_us_only]
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-4 concise sentences covering role fit, technical fit, seniority, Pakistan/remote eligibility, compensation if known, and important gaps]

Never invent facts not present in the resume, profile, or job description.
"""


# ── Rule-based pre-filter (catches obvious ineligible before LLM call) ─────
#
# Patterns validated 2026-04-23 against 5,938 historical scored jobs.
# Any pattern that would reject more than 1-2 jobs scored >=8 is omitted
# (those rare false positives tend to be LLM mis-scores anyway — jobs
# titled "Junior" / "Intern" don't belong in a Senior/Staff queue).

# Patterns that make a job ineligible regardless of tech stack.
# Checked against title + location field only (not full description, to avoid
# false positives from US companies mentioning global offices).
_INELIGIBLE_TITLE_PATTERNS = re.compile(
    # Clear non-Pakistan geographic restrictions in the title.
    r'\bUK[- ]only\b'
    r'|\bUnited Kingdom[- ]only\b'
    r'|\bCanada[- ]only\b'
    r'|\bUS[- ]only\b'
    r'|\bUSA[- ]only\b'
    r'|\bEurope[- ]only\b'
    r'|\bEU[- ]only\b'
    r'|\bIndia[- ]only\b'
    # Obvious non-target job noise.
    r'|\bIntern(ship)?\b'
    r'|\bCashier\b|\bBaker\b|\bCake Decorator\b|\bButcher\b|\bMeat Cutter\b'
    r'|\bGas Station Attendant\b|\bPharmacy Technician\b|\bHearing Aid Dispenser\b'
    r'|\bStocker\b|\bForklift\b|\bWarehouse Associate\b|\bTruck Driver\b'
    r'|\bBakery Clerk\b|\bDeli Clerk\b|\bProduce Clerk\b'
    r'|\bOptician\b'
    r'|\bRecruiter\b|\bTalent Acquisition\b|\bTalent Scout\b|\bTalent Sourcer\b'
    r'|\bAccount Manager\b|\bAccount Executive\b'
    r'|\bUX Designer\b|\bUI Designer\b|\bProduct Designer\b|\bGraphic Designer\b'
    r'|\bAndroid Engineer\b|\biOS Engineer\b|\bMobile Engineer\b'
    r'|\bSalesforce Developer\b|\bApex Developer\b'
    r'|\bMainframe Engineer\b|\bCOBOL Developer\b|\bTIBCO\b',
    re.IGNORECASE,
)

# Patterns checked against the location field specifically.
# Location field explicitly lists a non-US country name → ineligible.
_INELIGIBLE_LOCATION_PATTERNS = re.compile(
    # Regions
    r'\bEMEA\b'
    r'|\bAPAC\b'
    r'|\bEurope\b'
    # Europe
    r'|\bGermany\b|\bNetherlands\b|\bFrance\b|\bSpain\b|\bItaly\b'
    r'|\bPoland\b|\bUkraine\b|\bCzech\b|\bPortugal\b|\bIreland\b'
    r'|\bDenmark\b|\bSweden\b|\bNorway\b|\bFinland\b|\bBelgium\b'
    r'|\bSwitzerland\b|\bAustria\b|\bRomania\b|\bHungary\b|\bCroatia\b'
    r'|\bGreece\b|\bBulgaria\b|\bSerbia\b|\bSlovakia\b|\bSlovenia\b'
    r'|\bEstonia\b|\bLatvia\b|\bLithuania\b'
    # Asia
    r'|\bIndia\b|\bSingapore\b|\bJapan\b|\bVietnam\b|\bThailand\b'
    r'|\bPhilippines\b|\bIndonesia\b|\bKorea\b|\bTaiwan\b|\bHong Kong\b'
    r'|\bChina\b|\bBangladesh\b|\bMalaysia\b'
    # Latin America
    r'|\bBrazil\b|\bBrasil\b|\bMexico\b|\bMéxico\b|\bArgentina\b'
    r'|\bChile\b|\bColombia\b|\bPeru\b|\bUruguay\b'
    # Middle East / Africa
    r'|\bEgypt\b|\bNigeria\b|\bKenya\b|\bSouth Africa\b|\bIsrael\b'
    r'|\bTurkey\b|\bTürkiye\b|\bUAE\b|\bSaudi Arabia\b'
    # Oceania
    r'|\bAustralia\b|\bNew Zealand\b',
    re.IGNORECASE,
)

# Description-level non-US patterns. Scans the full description (capped at
# 6000 chars by the caller). Patterns intentionally narrow — must explicitly
# RESTRICT to a non-US country, not merely mention global offices.
# Tightened 2026-04-30 after Twilio UK/Canada slipped through the 800-char head.
_INELIGIBLE_DESC_PATTERNS = re.compile(
    r'Remote\s*[\(\-—–]\s*(EMEA|Europe|EU|UK|United\s+Kingdom|Germany|India|Canada|Ireland|Netherlands|Brazil|Mexico|Argentina|Colombia|Australia|New\s+Zealand)'
    r'|EMEA\s*(only|region|remote|based)'
    r'|(Europe|European)\s*(only|Time\s*Zone|timezone|based|remote)'
    # Belt-and-suspenders: catches "based in (the) UK", "based in Europe", etc.
    r'|based\s+in\s+(the\s+)?(Europe|EU|UK|United\s+Kingdom|Germany|India|Netherlands|Canada|Ireland|France|Spain|Italy|Brazil|Mexico|Australia|New\s+Zealand|Singapore|Japan|Israel|South\s+Africa|Portugal|Poland|Romania)'
    r'|will\s+be\s+remote\s+and\s+based\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|Germany|Europe|EMEA|India)'
    # Canadian-province patterns (Twilio L3 example)
    r'|Remote\s+(in|from|—|-)\s*(Ontario|British\s+Columbia|Alberta|Quebec|Manitoba|Nova\s+Scotia|Saskatchewan)'
    r'|Ontario,\s*British\s+Columbia'
    # UK/Canada right-to-work questions on the form (often mirrored in JD)
    r"|right\s+to\s+work\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|EU|European\s+Union)"
    r"|requires?\s+(the\s+)?right\s+to\s+work\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|EU)",
    re.IGNORECASE,
)

# Window scanned for description-level patterns. Bumped from 800 to 6000 chars
# 2026-04-30 — Twilio buried the UK restriction in a paragraph below the
# requirements list, past the 800-char head.
_DESC_SCAN_CHARS = 6000


def _check_ineligible(job: dict) -> str | None:
    """Return an ineligibility reason if the job is obviously non-US, else None.

    Checked before the LLM call to save tokens and ensure consistency.
    Scans title, location field, and the first ``_DESC_SCAN_CHARS`` of the
    description.
    """
    title = job.get("title") or ""
    location = job.get("location") or ""
    desc_head = (job.get("full_description") or "")[:_DESC_SCAN_CHARS]

    if _INELIGIBLE_TITLE_PATTERNS.search(title):
        return f"non-US geography in title: {title}"
    if location and _INELIGIBLE_LOCATION_PATTERNS.search(location):
        return f"non-US location field: {location}"
    m = _INELIGIBLE_DESC_PATTERNS.search(desc_head)
    if m:
        return f"non-US geography in description: {m.group(0)[:80]}"
    return None


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str, "eligibility": str}
        eligibility is one of: 'eligible' | 'non_us_only'.
        Older models that omit ELIGIBILITY default to 'eligible' (preserves
        prior behavior; new prompt requires the line so absence is rare).
    """
    score = 0
    keywords = ""
    reasoning = response
    eligibility = "eligible"

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("ELIGIBILITY:"):
            value = line.replace("ELIGIBILITY:", "").strip().lower()
            value = re.sub(r"[\[\]\s]+", "_", value).strip("_")
            if "non" in value and "us" in value:
                eligibility = "non_us_only"
            else:
                eligibility = "eligible"
        elif line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning,
            "eligibility": eligibility}


def _build_candidate_summary(profile: dict) -> str:
    """Build a candidate summary string from profile for the scoring prompt."""
    exp = profile.get("experience", {})
    boundary = profile.get("skills_boundary", {})
    years = exp.get("years_of_experience_total", "several")
    current_title = exp.get("current_job_title", "Software Engineer")
    target = exp.get("target_role", "Software Engineer")
    parts = [f"{current_title} with {years} years experience."]
    for category, skills in boundary.items():
        if isinstance(skills, list) and skills:
            parts.append(f"{category}: {', '.join(str(x) for x in skills[:12])}.")
    parts.append(f"Targets: {target}.")
    parts.append("Location: Pakistan; target arrangement: remote.")
    return " ".join(parts)


def score_job(resume_text: str, job: dict, profile: dict | None = None) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    if profile is None:
        profile = load_profile()

    # Rule-based pre-filter: catch obvious non-US ineligible jobs before LLM call.
    # Tags eligibility=non_us_only so downstream stages skip the job; the score
    # is still recorded so audit views can see the original LLM-style severity.
    ineligible_reason = _check_ineligible(job)
    if ineligible_reason:
        log.info("Pre-filter INELIGIBLE (non_us_only): %s — %s", (job.get("title") or "?")[:60], ineligible_reason)
        return {
            "score": 2,
            "keywords": "",
            "reasoning": f"Ineligible: {ineligible_reason}. Candidate is based in Pakistan.",
            "eligibility": "non_us_only",
        }

    try:
        candidate_summary = _build_candidate_summary(profile)
        score_prompt = SCORE_PROMPT_TEMPLATE.format(candidate_summary=candidate_summary)

        job_text = (
            f"TITLE: {job['title']}\n"
            f"COMPANY: {job['site']}\n"
            f"LOCATION: {job.get('location', 'N/A')}\n\n"
            f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
        )

        messages = [
            {"role": "system", "content": score_prompt},
            {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
        ]

        client = get_client()
        response = client.chat(messages, max_tokens=8192, temperature=0.2)
        return _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", (job or {}).get("title") or "?", e)
        return {"score": None, "keywords": "", "reasoning": "", "eligibility": None,
                "error": f"LLM error: {e}"}


MAX_SCORE_RETRIES = 5


def _score_backoff_minutes(retry_count: int) -> int:
    """Exponential backoff for scoring retries: 5, 20, 80, ~5h, ~21h."""
    return min(5 * (4 ** retry_count), 24 * 60)


def _flush_score_batch(conn, batch: list[dict], now: str) -> None:
    """Write a batch of scoring results to the DB.

    On success (score is not None): writes fit_score, clears score_error.
    On failure (score is None): leaves fit_score NULL, writes score_error + backoff.
    Jobs that have already hit MAX_SCORE_RETRIES stay unscored indefinitely (manual rescue needed).
    """
    from applypilot.database import transition_state
    from applypilot.config import DEFAULTS as _cfg_DEFAULTS
    _min_score = _cfg_DEFAULTS["min_score"]

    for r in batch:
        if r["score"] is not None:
            eligibility = r.get("eligibility") or "eligible"
            conn.execute(
                "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, "
                "eligibility = ?, "
                "score_error = NULL, score_attempts = 0, score_next_retry_at = NULL "
                "WHERE url = ?",
                (r["score"], f"{r['keywords']}\n{r['reasoning']}", now, eligibility,
                 r["url"]),
            )
            # Eligibility-driven state transition. Non-US roles go straight to
            # `archived` (terminal) so tailor/cover/apply never pick them up,
            # bypassing the scored→tailored→ready_to_apply chain entirely.
            if eligibility == "non_us_only":
                transition_state(conn, r["url"], "archived",
                                 reason="non_us_only employer/role",
                                 metadata={"score": r["score"],
                                           "eligibility": eligibility},
                                 force=True)
            else:
                to_state = "scored" if r["score"] >= _min_score else "low_score"
                transition_state(conn, r["url"], to_state,
                                 reason=f"scored {r['score']}/10",
                                 metadata={"score": r["score"],
                                           "eligibility": eligibility})
        else:
            # LLM failure — keep fit_score NULL so it stays in pending_score
            row = conn.execute(
                "SELECT COALESCE(score_attempts, 0) FROM jobs WHERE url = ?", (r["url"],)
            ).fetchone()
            retry_count = row[0] if row else 0
            if retry_count >= MAX_SCORE_RETRIES:
                # Give up — write score_error but don't schedule another retry
                conn.execute(
                    "UPDATE jobs SET score_error = ?, score_attempts = ?, "
                    "score_next_retry_at = NULL, scored_at = ? WHERE url = ?",
                    (r["error"], retry_count + 1, now, r["url"]),
                )
                transition_state(conn, r["url"], "score_failed",
                                 reason=f"LLM failed after {retry_count+1} attempts",
                                 metadata={"error": r["error"]})
            else:
                delay = _score_backoff_minutes(retry_count)
                next_retry = (
                    datetime.now(timezone.utc) + timedelta(minutes=delay)
                ).isoformat()
                conn.execute(
                    "UPDATE jobs SET score_error = ?, score_attempts = ?, "
                    "score_next_retry_at = ?, scored_at = ? WHERE url = ?",
                    (r["error"], retry_count + 1, next_retry, now, r["url"]),
                )
                log.info("  score retry %d/%d scheduled in %d min for %s",
                         retry_count + 1, MAX_SCORE_RETRIES, delay, r["url"][:60])


def run_scoring(limit: int = 0, rescore: bool = False, workers: int = 1,
                max_age_days: int | None = None) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).
        workers: Parallel LLM threads (default 1 = sequential).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        # Note: get_jobs_by_stage now applies a 14-day discovered_at filter by
        # default (config.DEFAULTS["max_job_age_days"]). Pass max_age_days=0
        # to disable.
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score",
                                 max_age_days=max_age_days, limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs (workers=%d)...", len(jobs), workers)
    t0 = time.time()
    completed = 0
    errors = 0
    batch_size = 25  # Commit every N jobs so downstream stages see results sooner
    batch: list[dict] = []

    def _score_one(job: dict) -> dict:
        try:
            result = score_job(resume_text, job)
            result["url"] = job["url"]
        except Exception as e:
            log.error("Unexpected error scoring '%s': %s", (job or {}).get("title") or "?", e)
            result = {
                "score": None, "keywords": "", "reasoning": "",
                "error": f"Unexpected: {e}", "url": (job or {}).get("url", ""),
            }
        return result

    def _flush_and_log(batch: list[dict], completed: int) -> list[dict]:
        now = datetime.now(timezone.utc).isoformat()
        try:
            write_with_retry(conn, _flush_score_batch, conn, batch, now)
        except Exception as flush_err:
            log.exception("Batch flush failed (batch of %d): %s", len(batch), flush_err)
        log.info("Committed batch of %d scores to DB (%d/%d total)", len(batch), completed, len(jobs))
        return []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                completed += 1
                if result["score"] is None:
                    errors += 1
                batch.append(result)
                log.info(
                    "[%d/%d] score=%s  %s",
                    completed, len(jobs),
                    result["score"] if result["score"] is not None else "ERR",
                    (job.get("title") or "")[:60],
                )
                if len(batch) >= batch_size:
                    batch = _flush_and_log(batch, completed)
    else:
        for job in jobs:
            result = _score_one(job)
            completed += 1
            if result["score"] is None:
                errors += 1
            batch.append(result)
            log.info(
                "[%d/%d] score=%s  %s",
                completed, len(jobs),
                result["score"] if result["score"] is not None else "ERR",
                (job.get("title") or "")[:60],
            )
            if len(batch) >= batch_size:
                batch = _flush_and_log(batch, completed)

    # Flush remaining
    if batch:
        now = datetime.now(timezone.utc).isoformat()
        try:
            write_with_retry(conn, _flush_score_batch, conn, batch, now)
        except Exception as flush_err:
            log.exception("Final batch flush failed (batch of %d): %s", len(batch), flush_err)

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", completed, elapsed, completed / elapsed if elapsed > 0 else 0)

    # Score distribution
    try:
        dist = conn.execute("""
            SELECT fit_score, COUNT(*) FROM jobs
            WHERE fit_score IS NOT NULL
            GROUP BY fit_score ORDER BY fit_score DESC
        """).fetchall()
        distribution = [(row[0], row[1]) for row in dist]
    except Exception as dist_err:
        log.exception("Distribution query failed: %s", dist_err)
        distribution = []

    return {
        "scored": completed,
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
