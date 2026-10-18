"""Demo mode: run the app with no API key, on a bundled sample careers page and saved model outputs.

Only the two model calls are replaced. Page cleaning, job validation and portfolio
matching run exactly as in live mode.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_PAGE = SAMPLES_DIR / "sample_careers_page.html"
SAMPLE_OUTPUTS = SAMPLES_DIR / "demo_outputs.json"
DEMO_LABEL = "Demo output, no API key"

EXTRACT_MARKER = "### SCRAPED TEXT FROM A CAREERS PAGE"
ROLE_LINE = re.compile(r"^Role: (.+)$", re.MULTILINE)


class ReplayChatModel(BaseChatModel):
    """A chat model that answers from saved outputs instead of calling an API.

    The extraction prompt gets the saved jobs as JSON. An email prompt gets the
    saved email for the role named in the prompt.
    """

    extraction: str
    emails: dict[str, str]

    @property
    def _llm_type(self) -> str:
        return "replay"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = messages[0].text
        if EXTRACT_MARKER in prompt:
            answer = self.extraction
        else:
            role = ROLE_LINE.search(prompt)
            answer = self.emails.get(role.group(1).strip() if role else "", "")
            answer = answer or "There is no saved demo email for this role."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(answer))])


def load_demo_outputs(path: Path = SAMPLE_OUTPUTS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def demo_llm(path: Path = SAMPLE_OUTPUTS) -> ReplayChatModel:
    """The replay model loaded with the saved outputs for the sample page."""
    outputs = load_demo_outputs(path)
    return ReplayChatModel(
        extraction=json.dumps(outputs["extraction"], indent=2),
        emails=outputs["emails"],
    )


def sample_page_html(path: Path = SAMPLE_PAGE) -> bytes:
    return Path(path).read_bytes()
