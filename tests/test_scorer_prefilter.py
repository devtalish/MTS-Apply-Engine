"""Policy and scorer pre-filter regression tests."""
import pytest

from applypilot.job_policy import evaluate_job
from applypilot.scoring.scorer import _check_ineligible, _parse_score_response


def job(title="Software Developer", location="Remote", description="Remote worldwide software development work."):
    return {"title": title, "location": location, "full_description": description, "site": "Example"}


@pytest.mark.parametrize("title", [
    "Senior Software Engineer",
    "Staff Backend Engineer",
    "Principal Engineer",
    "Engineering Lead",
    "Director of Engineering",
    "CTO",
    "Software Engineering Intern",
])
def test_non_target_seniority_or_internship_is_rejected(title):
    assert _check_ineligible(job(title=title)) is not None


@pytest.mark.parametrize("title", [
    "Junior Software Developer",
    "Junior Backend Developer",
    "Entry Level Python Developer",
    "Associate Full Stack Developer",
])
def test_early_career_roles_are_not_rejected_by_seniority(title):
    assert _check_ineligible(job(title=title)) is None


@pytest.mark.parametrize("loc", [
    "Remote — India",
    "Remote — Europe",
    "Remote — Canada",
    "Germany",
    "EMEA",
])
def test_non_pakistan_restricted_location_is_rejected(loc):
    assert _check_ineligible(job(location=loc)) is not None


def test_global_remote_is_allowed():
    assert _check_ineligible(job(location="Remote - Worldwide")) is None


def test_us_company_does_not_mean_us_only():
    assert _check_ineligible(job(
        location="Remote",
        description="US-based company. Work remotely from anywhere in the world."
    )) is None


def test_buried_uk_restriction_is_rejected():
    description = "Engineering role. " * 300 + "This role will be remote and based in the UK."
    assert _check_ineligible(job(description=description)) is not None


@pytest.mark.parametrize("title", [
    "Technical Recruiter",
    "Account Executive",
    "Customer Support Specialist",
    "UX Designer",
    "Mobile Engineer",
    "Salesforce Developer",
])
def test_non_target_roles_are_rejected(title):
    assert _check_ineligible(job(title=title)) is not None


def test_part_time_target_is_allowed():
    result = evaluate_job(job(
        title="Part-Time Python Developer",
        description="Remote worldwide. 10 hours per week. Build Python automation and APIs."
    ))
    assert result.eligible
    assert result.hours_per_week == 10


def test_twenty_hours_is_secondary_ceiling():
    result = evaluate_job(job(
        title="Contract Full Stack Developer",
        description="Remote worldwide. 20 hours per week. Build React and Node.js applications."
    ))
    assert result.eligible


def test_over_twenty_hours_is_rejected():
    result = evaluate_job(job(
        title="Contract Full Stack Developer",
        description="Remote worldwide. 25 hours per week. Build React and Node.js applications."
    ))
    assert not result.eligible
    assert result.category == "hours"


def test_full_time_only_is_rejected():
    result = evaluate_job(job(
        title="Software Developer",
        description="This is a full-time only role. You will work 40 hours per week."
    ))
    assert not result.eligible


def test_unpaid_is_rejected():
    result = evaluate_job(job(
        title="Python Developer",
        description="Remote worldwide unpaid internship. 10 hours per week."
    ))
    assert not result.eligible


def test_score_parser_eligibility():
    parsed = _parse_score_response(
        "ELIGIBILITY: eligible\nSCORE: 9\nKEYWORDS: Python, APIs\nREASONING: Strong match."
    )
    assert parsed["score"] == 9
    assert parsed["eligibility"] == "eligible"


def test_score_parser_non_us():
    parsed = _parse_score_response(
        "ELIGIBILITY: non_us_only\nSCORE: 8\nKEYWORDS: Python\nREASONING: UK-only."
    )
    assert parsed["eligibility"] == "non_us_only"
