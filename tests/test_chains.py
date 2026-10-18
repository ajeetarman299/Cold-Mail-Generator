"""Job extraction (parsing, validation, retry) and email prompts, run on LangChain's fake chat models."""

import json

import groq
import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from chains import Chain, ExtractionError, Job, LLMError, Sender
from portfolio import PortfolioMatch

ML_JOB = {
    "role": "Machine Learning Engineer",
    "experience": "3+ years",
    "skills": ["Python", "TensorFlow"],
    "description": "Build recommendation models.",
}
APP_JOB = {"role": "Android Developer", "experience": "", "skills": ["Kotlin"], "description": "Build the app."}


def test_extract_jobs_reads_every_job_in_the_answer(fake_llm):
    llm = fake_llm(json.dumps({"jobs": [ML_JOB, APP_JOB]}))

    jobs = Chain(llm).extract_jobs("the page text")

    assert [job.role for job in jobs] == ["Machine Learning Engineer", "Android Developer"]
    assert jobs[0].skills == ["Python", "TensorFlow"]
    assert len(llm.prompts) == 1
    assert "the page text" in llm.prompts[0]


@pytest.mark.parametrize(
    "answer",
    [
        json.dumps([ML_JOB]),
        json.dumps(ML_JOB),
        "```json\n" + json.dumps({"jobs": [ML_JOB]}) + "\n```",
        "Here are the jobs:\n" + json.dumps({"jobs": [ML_JOB]}) + "\nLet me know if you need more.",
        "<think>The page lists one job.</think>" + json.dumps({"jobs": [ML_JOB]}),
    ],
    ids=["bare list", "single object", "code fence", "text around the JSON", "reasoning block"],
)
def test_extract_jobs_accepts_common_answer_shapes(fake_llm, answer):
    jobs = Chain(fake_llm(answer)).extract_jobs("the page text")

    assert [job.role for job in jobs] == ["Machine Learning Engineer"]


def test_extract_jobs_returns_an_empty_list_for_a_page_without_jobs(fake_llm):
    assert Chain(fake_llm('{"jobs": []}')).extract_jobs("About us") == []


def test_job_fields_are_normalised():
    job = Job.model_validate(
        {
            "role": " Data Engineer ",
            "experience": None,
            "skills": "Python, SQL, python, ",
            "description": ["Build pipelines", "Own the warehouse"],
        }
    )

    assert job.role == "Data Engineer"
    assert job.experience == ""
    assert job.skills == ["Python", "SQL"]
    assert job.description == "Build pipelines; Own the warehouse"


def test_extract_jobs_retries_once_after_an_answer_without_json(fake_llm):
    llm = fake_llm("Sorry, I cannot format that.", json.dumps({"jobs": [ML_JOB]}))

    jobs = Chain(llm).extract_jobs("the page text")

    assert [job.role for job in jobs] == ["Machine Learning Engineer"]
    assert len(llm.prompts) == 2
    retry = llm.prompts[1]
    assert "Sorry, I cannot format that." in retry  # the model sees its bad answer
    assert "it did not contain any JSON" in retry  # and the reason it was refused


def test_extract_jobs_retries_when_a_job_has_no_role(fake_llm):
    llm = fake_llm(json.dumps({"jobs": [{"role": "", "skills": ["Go"]}]}), json.dumps({"jobs": [APP_JOB]}))

    jobs = Chain(llm).extract_jobs("the page text")

    assert [job.role for job in jobs] == ["Android Developer"]
    assert "jobs.0.role" in llm.prompts[1]


def test_extract_jobs_does_not_accept_truncated_json(fake_llm):
    truncated = json.dumps({"jobs": [ML_JOB]})[:60]
    llm = fake_llm(truncated, json.dumps({"jobs": [ML_JOB]}))

    Chain(llm).extract_jobs("the page text")

    assert len(llm.prompts) == 2
    assert "not valid JSON" in llm.prompts[1]


def test_extract_jobs_gives_up_after_one_retry(fake_llm):
    llm = fake_llm("not JSON", "still not JSON", json.dumps({"jobs": [ML_JOB]}))

    with pytest.raises(ExtractionError, match="after a retry"):
        Chain(llm).extract_jobs("the page text")
    assert len(llm.prompts) == 2


def test_write_mail_uses_the_sender_settings_and_the_matched_links(fake_llm):
    llm = fake_llm("Subject: Hello\n\nBody")
    sender = Sender(
        name="Priya Shah",
        role="Founder",
        company="Example Data Co",
        offer="data engineering teams",
        tone="Formal",
        length="Long",
    )
    matches = [PortfolioMatch("https://example.com/ml", "Machine Learning, Python", 0.2, ["Python"])]

    email = Chain(llm).write_mail(Job.model_validate(ML_JOB), matches, sender)

    assert email == "Subject: Hello\n\nBody"
    prompt = llm.prompts[0]
    for expected in [
        "You are Priya Shah, Founder at Example Data Co",
        "Example Data Co offers data engineering teams",
        "formal and polite",
        "180 to 250 words",
        "- https://example.com/ml (Machine Learning, Python)",
        "Role: Machine Learning Engineer",
        "Skills: Python, TensorFlow",
    ]:
        assert expected in prompt
    assert "Mohan" not in prompt
    assert "AtliQ" not in prompt


def test_write_mail_tells_the_model_when_no_links_matched(fake_llm):
    llm = fake_llm("Subject: Hello")

    Chain(llm).write_mail(Job.model_validate(APP_JOB), [], Sender())

    assert "(no portfolio links matched)" in llm.prompts[0]
    assert "Experience: not stated" in llm.prompts[0]


def test_blank_sender_fields_fall_back_to_the_defaults():
    sender = Sender(name="  ", company="", role="CTO")

    assert sender.name == Sender().name
    assert sender.company == Sender().company
    assert sender.role == "CTO"


def test_groq_errors_become_readable_messages():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)

    class RateLimitedLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise error

    with pytest.raises(LLMError, match="rate limit"):
        Chain(RateLimitedLLM(responses=[""])).extract_jobs("the page text")
