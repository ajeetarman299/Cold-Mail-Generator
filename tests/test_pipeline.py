"""The full page-to-emails run, with fake chat models."""

import json

import groq
import httpx
import pytest

from chains import Chain, Sender
from fakes import RecordingFakeLLM
from pipeline import generate
from utils import PageError

PAGE = (
    "<html><body><h1>Careers</h1>"
    + "".join(f"<p>Opening {n}: we are hiring an engineer who knows Python and SQL.</p>" for n in range(10))
    + "</body></html>"
)


def jobs_answer(count: int) -> str:
    return json.dumps({"jobs": [{"role": f"Engineer {n}", "skills": ["Python"]} for n in range(1, count + 1)]})


def test_generate_writes_one_email_per_job(fake_llm, portfolio):
    llm = fake_llm(jobs_answer(2), "Subject: First", "Subject: Second")

    result = generate(PAGE, Chain(llm), portfolio, Sender())

    assert result.jobs_found == 2
    assert [item.email for item in result.results] == ["Subject: First", "Subject: Second"]
    assert all(item.matches for item in result.results)


def test_generate_writes_emails_for_the_first_jobs_only(fake_llm, portfolio):
    llm = fake_llm(jobs_answer(12), *["Subject: Hi"] * 12)

    result = generate(PAGE, Chain(llm), portfolio, Sender(), max_jobs=10)

    assert result.jobs_found == 12
    assert len(result.results) == 10


def test_generate_reports_a_failed_email_on_its_job_and_goes_on(portfolio):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    rate_limited = groq.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)

    class SecondEmailFails(RecordingFakeLLM):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            answer = super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)
            if answer == "FAIL":
                raise rate_limited
            return answer

    llm = SecondEmailFails(responses=[jobs_answer(3), "Subject: One", "FAIL", "Subject: Three"])

    result = generate(PAGE, Chain(llm), portfolio, Sender())

    assert [item.email for item in result.results] == ["Subject: One", "", "Subject: Three"]
    assert "rate limit" in result.results[1].error


def test_generate_reports_progress(fake_llm, portfolio):
    messages = []

    generate(PAGE, Chain(fake_llm(jobs_answer(1), "Subject: Hi")), portfolio, Sender(), on_progress=messages.append)

    assert messages == ["Extracting the job postings", "Writing email 1 of 1: Engineer 1"]


def test_generate_stops_before_calling_the_model_when_the_page_is_empty(fake_llm, portfolio):
    llm = fake_llm(jobs_answer(1))

    with pytest.raises(PageError):
        generate("<html><body><div id='app'></div></body></html>", Chain(llm), portfolio, Sender())
    assert llm.prompts == []
