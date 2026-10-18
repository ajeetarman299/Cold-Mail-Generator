"""The two LLM steps: extract the job postings from page text, then write one cold email per job."""

from __future__ import annotations

import json
import os
import re
from typing import Literal

import groq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

from portfolio import PortfolioMatch

# The first version used llama-3.1-70b-versatile, which Groq has retired.
# openai/gpt-oss-120b is a current Groq production model. Set GROQ_MODEL to use
# any other model from https://console.groq.com/docs/models.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# .env.example ships this value; treat it as "no key".
PLACEHOLDER_KEY = "your_groq_api_key_here"

TONES = {
    "Friendly": "warm and conversational, but professional",
    "Formal": "formal and polite",
    "Direct": "brief and direct, with one clear request",
}
LENGTHS = {
    "Short": "a body of at most 120 words",
    "Medium": "a body of 120 to 180 words",
    "Long": "a body of 180 to 250 words",
}

EXTRACT_PROMPT = PromptTemplate.from_template(
    """### SCRAPED TEXT FROM A CAREERS PAGE:
{page_text}

### INSTRUCTION:
The text above was scraped from the careers page of a company website.
Find every job posting in it and return them as JSON: one object with the key "jobs",
holding a list of objects with these keys:
- "role": the job title
- "experience": the experience the job asks for, as a short string ("" if not stated)
- "skills": a list of short skill or technology names, such as "Python" or "Kubernetes"
- "description": one or two sentences on what the person in this role will do
If the text has no job postings, return {{"jobs": []}}.
Only return the valid JSON.
### VALID JSON (NO PREAMBLE):
"""
)

RETRY_MESSAGE = (
    "Your answer could not be used: {error}. Reply again with only one JSON object of the form "
    '{{"jobs": [{{"role": "...", "experience": "...", "skills": ["..."], "description": "..."}}]}}, '
    "and nothing else."
)

EMAIL_PROMPT = PromptTemplate.from_template(
    """### JOB POSTING:
Role: {role}
Experience: {experience}
Skills: {skills}
Description: {description}

### PORTFOLIO LINKS (closest match first):
{links}

### INSTRUCTION:
You are {sender_name}, {sender_role} at {sender_company}. {sender_company} offers {sender_offer}.
Write a cold email to the hiring team for the job above, explaining how {sender_company} can help them with this work.
Include the most relevant links from the portfolio list above, and no other links. If the list is empty, do not mention a portfolio.
Do not invent clients, numbers or results.
Tone: {tone}.
Length: {length}, after a "Subject:" line.
Sign off with your name, role and company.
Do not provide a preamble.
### EMAIL (NO PREAMBLE):
"""
)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class Job(BaseModel):
    """One job posting, as extracted from the page."""

    role: str = Field(min_length=1)
    experience: str = ""
    skills: list[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("role", "experience", "description", mode="before")
    @classmethod
    def _as_text(cls, value):
        """Accept null, numbers or lists from the model and turn them into plain text."""
        if value is None:
            return ""
        if isinstance(value, list):
            return "; ".join(str(item).strip() for item in value if item)
        return str(value).strip()

    @field_validator("skills", mode="before")
    @classmethod
    def _as_skill_list(cls, value):
        """Accept a list or a comma-separated string; drop blanks and repeats."""
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, list):
            raise ValueError("skills must be a list of strings")
        skills: list[str] = []
        for item in value:
            skill = str(item).strip()
            if skill and skill.lower() not in {known.lower() for known in skills}:
                skills.append(skill)
        return skills


class JobList(BaseModel):
    jobs: list[Job]


class Sender(BaseModel):
    """Who the email is from. The app shows these as inputs in its sidebar."""

    name: str = "Alex Morgan"
    role: str = "Business Development Manager"
    company: str = "Acme Software"
    offer: str = (
        "custom software development and dedicated engineers for web, mobile, cloud "
        "and machine learning projects"
    )
    tone: Literal["Friendly", "Formal", "Direct"] = "Friendly"
    length: Literal["Short", "Medium", "Long"] = "Short"

    @field_validator("name", "role", "company", "offer", mode="before")
    @classmethod
    def _default_if_blank(cls, value, info: ValidationInfo):
        """An empty input falls back to the default, so the prompt never has gaps."""
        text = str(value or "").strip()
        return text or cls.model_fields[info.field_name].default


class JobParseError(ValueError):
    """The model's answer was not valid job data. The message is sent back to the model on retry."""


class ExtractionError(Exception):
    """The model did not return valid job data, even after one retry."""


class LLMError(Exception):
    """A model call failed. The message is written for the person using the app."""


def groq_api_key() -> str | None:
    """The Groq key from the environment (or .env), or None when it is missing or the placeholder."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key if key and key != PLACEHOLDER_KEY else None


def model_name() -> str:
    """The Groq model id from GROQ_MODEL, or the default."""
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL


def build_llm(api_key: str, model: str | None = None) -> BaseChatModel:
    """The Groq chat model used in live mode."""
    return ChatGroq(model=model or model_name(), temperature=0, api_key=api_key)


def strip_thinking(text: str) -> str:
    """Remove <think> blocks that some reasoning models put before their answer."""
    return THINK_BLOCK.sub("", text).strip()


def load_json(text: str):
    """Read the JSON value in a model answer, allowing code fences and text around it.

    Truncated JSON is rejected rather than repaired, so it triggers a retry.
    """
    text = strip_thinking(text)
    fenced = CODE_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise JobParseError("it did not contain any JSON")
    try:
        value, _ = json.JSONDecoder().raw_decode(text, min(starts))
    except json.JSONDecodeError as exc:
        raise JobParseError(f"it was not valid JSON ({exc})") from exc
    return value


def parse_jobs(text: str) -> list[Job]:
    """Turn a model answer into validated jobs. Raises JobParseError if it does not fit the schema."""
    data = load_json(text)
    if isinstance(data, dict) and "jobs" in data:
        items = data["jobs"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "role" in data:
        items = [data]
    else:
        raise JobParseError('it was not an object with a "jobs" list')
    try:
        return JobList(jobs=items).jobs
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:3]
        )
        raise JobParseError(f"the job data did not have the expected fields ({problems})") from exc


def format_links(matches: list[PortfolioMatch]) -> str:
    if not matches:
        return "(no portfolio links matched)"
    return "\n".join(f"- {match.link} ({match.techstack})" for match in matches)


def readable_llm_error(exc: groq.APIError) -> str:
    if isinstance(exc, groq.AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
    if isinstance(exc, groq.RateLimitError):
        return "Groq's rate limit was reached. Wait a minute, then try again."
    if isinstance(exc, groq.NotFoundError):
        return "Groq does not offer the model in GROQ_MODEL. Pick a current model id."
    if isinstance(exc, groq.APIConnectionError):
        return "Could not reach Groq. Check your internet connection and try again."
    if isinstance(exc, groq.APIStatusError):
        return f"Groq returned an error (HTTP {exc.status_code}): {exc.message}"
    return f"The call to Groq failed: {exc}"


class Chain:
    """Runs the two prompts on any LangChain chat model: Groq in live mode, saved outputs in demo mode."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def extract_jobs(self, page_text: str) -> list[Job]:
        """Ask the model for the job postings as JSON. If the answer is not valid, retry once."""
        messages: list[BaseMessage] = [HumanMessage(EXTRACT_PROMPT.format(page_text=page_text))]
        problem = None
        for _ in range(2):
            answer = self._ask(messages)
            try:
                return parse_jobs(answer)
            except JobParseError as exc:
                problem = exc
                messages += [AIMessage(answer), HumanMessage(RETRY_MESSAGE.format(error=exc))]
        raise ExtractionError(
            f"The model did not return valid job data after a retry ({problem}). "
            "Try again, or use the URL of a single job posting."
        )

    def write_mail(self, job: Job, matches: list[PortfolioMatch], sender: Sender) -> str:
        prompt = EMAIL_PROMPT.format(
            role=job.role,
            experience=job.experience or "not stated",
            skills=", ".join(job.skills) or "not stated",
            description=job.description or "not stated",
            links=format_links(matches),
            sender_name=sender.name,
            sender_role=sender.role,
            sender_company=sender.company,
            sender_offer=sender.offer,
            tone=TONES[sender.tone],
            length=LENGTHS[sender.length],
        )
        return strip_thinking(self._ask([HumanMessage(prompt)]))

    def _ask(self, messages: list[BaseMessage]) -> str:
        try:
            return self.llm.invoke(messages).text
        except groq.APIError as exc:
            raise LLMError(readable_llm_error(exc)) from exc
