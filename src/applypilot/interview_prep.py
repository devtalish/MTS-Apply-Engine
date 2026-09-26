"""Interview preparation packet generator.

Builds a local, evidence-grounded packet from an interview-state job and the
candidate's actual resume. It never invents experience or claims employment.
"""
from __future__ import annotations

import json
from pathlib import Path

from applypilot.config import APP_DIR, RESUME_PATH, load_profile
from applypilot.database import get_connection
from applypilot.llm import get_client

PROMPT = """Create an interview preparation packet for Muhammad Talish.

Rules:
- Use only the supplied resume/profile and job description.
- Projects and Velnaro founder work are valid evidence, but never present them
  as employment for another company.
- Do not invent metrics, clients, certifications, years, technologies, or outcomes.
- Prepare the candidate to answer himself; this is preparation, not covert
  assistance during an interview.

Return markdown with:
1. Company/role brief
2. Why this role matches (evidence-backed)
3. Technical areas to review
4. 10 likely technical questions
5. 8 project-specific questions
6. 6 behavioral questions with evidence prompts
7. 5 architecture/tradeoff questions
8. 5 questions to ask the interviewer
9. Explicit skill gaps to study
10. A 30-minute pre-interview drill

JOB:
{job}

RESUME:
{resume}

PROFILE:
{profile}
"""


def build_packet(job: dict) -> str:
    profile = load_profile()
    resume = RESUME_PATH.read_text(encoding="utf-8")
    prompt = PROMPT.format(
        job=json.dumps(job, ensure_ascii=False, indent=2),
        resume=resume,
        profile=json.dumps(profile, ensure_ascii=False, indent=2),
    )
    return get_client(quality=True).ask(prompt, max_tokens=6000, temperature=0.1)


def generate_packets(limit: int = 10) -> list[Path]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM jobs
           WHERE state = 'interview'
           ORDER BY last_email_at DESC, fit_score DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    if not rows:
        return []
    out_dir = APP_DIR / "interview_packets"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in rows:
        job = dict(row)
        safe = "".join(ch if ch.isalnum() else "_" for ch in (job.get("company") or "company"))[:40]
        title = "".join(ch if ch.isalnum() else "_" for ch in (job.get("title") or "role"))[:60]
        path = out_dir / f"{safe}_{title}.md"
        path.write_text(build_packet(job), encoding="utf-8")
        paths.append(path)
    return paths
