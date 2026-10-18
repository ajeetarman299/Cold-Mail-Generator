"""From page HTML to emails: the steps the app runs for one careers page. Used by the UI and the tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chains import Chain, Job, LLMError, Sender
from portfolio import Portfolio, PortfolioMatch
from utils import page_text_from_html

# A page can list dozens of roles. Write emails for the first few only, to keep
# the run quick and the API usage small.
MAX_JOBS = 10


@dataclass
class JobResult:
    job: Job
    matches: list[PortfolioMatch]
    email: str = ""
    error: str = ""


@dataclass
class PageResult:
    jobs_found: int
    results: list[JobResult]
    text_was_cut: bool


def generate(
    html: str | bytes,
    chain: Chain,
    portfolio: Portfolio,
    sender: Sender,
    max_jobs: int = MAX_JOBS,
    on_progress: Callable[[str], None] | None = None,
) -> PageResult:
    """Clean the page, extract its jobs, match each job to the portfolio and write its email.

    Raises PageError, ExtractionError or LLMError when the page or the extraction
    fails. A failed email for one job is recorded on that job, and the others go on.
    """
    report = on_progress or (lambda message: None)
    text, was_cut = page_text_from_html(html)
    report("Extracting the job postings")
    jobs = chain.extract_jobs(text)
    selected = jobs[:max_jobs]
    results = []
    for number, job in enumerate(selected, start=1):
        report(f"Writing email {number} of {len(selected)}: {job.role}")
        matches = portfolio.match(job.skills)
        try:
            results.append(JobResult(job, matches, email=chain.write_mail(job, matches, sender)))
        except LLMError as exc:
            results.append(JobResult(job, matches, error=str(exc)))
    return PageResult(jobs_found=len(jobs), results=results, text_was_cut=was_cut)
