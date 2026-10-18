"""Streamlit UI: paste a careers page URL, get one cold email per open role and the portfolio links behind it.

Run from the repository root:  streamlit run app/main.py
"""

from __future__ import annotations

import re

import streamlit as st
from dotenv import load_dotenv

from chains import (
    LENGTHS,
    TONES,
    Chain,
    ExtractionError,
    LLMError,
    Sender,
    build_llm,
    groq_api_key,
    model_name,
)
from demo import DEMO_LABEL, SAMPLE_PAGE, demo_llm, sample_page_html
from pipeline import JobResult, PageResult, generate
from portfolio import MAX_DISTANCE, Portfolio
from utils import MAX_PAGE_CHARS, PageError, fetch_html

load_dotenv()

# Less empty space around the page than Streamlit's default, so a result fits on one screen.
PAGE_STYLE = """
<style>
[data-testid="stMainBlockContainer"] { padding: 3rem 2.5rem 4rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the portfolio and its embedding model...")
def get_portfolio() -> Portfolio:
    return Portfolio()


URL = re.compile(r"(https?://[^\s<>]*[^\s<>.,;:!?)\]'\"])")


def plain(text: str) -> str:
    """Escape Markdown, so text from a scraped page or a model shows exactly as written."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~:$<])", r"\\\1", text)


def email_markdown(email: str) -> str:
    """Show an email as written: Markdown escaped, every line break kept, links clickable."""

    def line(text: str) -> str:
        parts = URL.split(text)
        return "".join(f"<{part}>" if index % 2 else plain(part) for index, part in enumerate(parts))

    paragraphs = re.split(r"\n\s*\n", email.strip())
    return "\n\n".join("  \n".join(line(text) for text in paragraph.splitlines()) for paragraph in paragraphs)


def tab_labels(roles: list[str]) -> list[str]:
    """One tab label per role, numbering repeats so every label is unique."""
    seen: dict[str, int] = {}
    labels = []
    for role in roles:
        seen[role] = seen.get(role, 0) + 1
        labels.append(role if seen[role] == 1 else f"{role} ({seen[role]})")
    return labels


def sender_sidebar(demo: bool, model: str) -> Sender:
    default = Sender()
    with st.sidebar:
        st.header("Sender")
        if demo:
            st.caption(
                "Demo mode replays emails saved with these default settings. "
                "Add a GROQ_API_KEY to write emails with your own."
            )
        name = st.text_input("Your name", default.name, disabled=demo)
        role = st.text_input("Your role", default.role, disabled=demo)
        company = st.text_input("Company", default.company, disabled=demo)
        offer = st.text_area("What you offer", default.offer, height=100, disabled=demo)
        tone = st.segmented_control(
            "Tone", list(TONES), default=default.tone, required=True, disabled=demo
        )
        length = st.segmented_control(
            "Length", list(LENGTHS), default=default.length, required=True, disabled=demo
        )
        st.divider()
        st.caption(f"Model: {model}")
    return Sender(
        name=name,
        role=role,
        company=company,
        offer=offer,
        tone=tone or default.tone,
        length=length or default.length,
    )


def page_form(demo: bool) -> tuple[str, bool]:
    with st.form("page", border=False):
        url_column, button_column = st.columns([4, 1], vertical_alignment="bottom")
        if demo:
            url = url_column.text_input(
                "Careers page", f"Sample page: data/samples/{SAMPLE_PAGE.name}", disabled=True
            )
        else:
            url = url_column.text_input(
                "Careers page URL",
                placeholder="https://careers.example.com/jobs",
                help="A careers page or a single job posting. The app writes one email per role.",
            )
        submitted = button_column.form_submit_button("Write emails", type="primary", width="stretch")
    return url, submitted


def run(url: str, sender: Sender, api_key: str | None) -> PageResult | None:
    """Run the pipeline for one page and show readable errors. Returns None when there is nothing to show."""
    demo = api_key is None
    try:
        portfolio = get_portfolio()
    except Exception as exc:  # noqa: BLE001  (shown to the user)
        reason = str(exc).rstrip(".")
        st.error(
            f"Could not load the portfolio store: {reason}. The first run downloads a small embedding "
            "model (about 80 MB), which needs internet access."
        )
        return None

    chain = Chain(demo_llm() if demo else build_llm(api_key))
    progress = st.empty()
    try:
        with progress.status("Reading the page") as status:
            html = sample_page_html() if demo else fetch_html(url)
            result = generate(
                html, chain, portfolio, sender, on_progress=lambda message: status.update(label=message)
            )
    except (PageError, ExtractionError, LLMError) as exc:
        progress.empty()
        st.error(str(exc))
        return None
    progress.empty()

    if not result.results:
        st.warning(
            "No job postings found on this page. Try a careers page that lists open roles, "
            "or the URL of a single job posting."
        )
        return None
    return result


def show_matches(item: JobResult) -> None:
    st.markdown("**Matched portfolio links**")
    if not item.matches:
        st.info("No portfolio entry is close to this job's skills, so the email cites none.")
        return
    for match in item.matches:
        shown = re.sub(r"^https?://(?:www\.)?", "", match.link)
        st.markdown(
            f"[{plain(shown)}](<{match.link}>)  \n"
            f":gray-badge[distance {match.distance:.2f}] "
            f":small[{plain(match.techstack)}. Matched: {plain(', '.join(match.skills))}]"
        )
    st.caption(
        f"Cosine distance to the closest job skill: 0 is identical, lower is closer. "
        f"Entries above {MAX_DISTANCE} are left out."
    )


def show_job(item: JobResult, demo: bool, key: str) -> None:
    job = item.job
    facts = []
    if job.experience:
        facts.append(f"**Experience:** {plain(job.experience)}")
    if job.skills:
        facts.append("**Skills:** " + " ".join(f":blue-badge[{plain(skill)}]" for skill in job.skills))
    if facts:
        st.markdown("  \n".join(facts))

    email_column, match_column = st.columns([2, 1], gap="medium")
    with email_column:
        st.markdown("**Email**" + (f" :orange-badge[{DEMO_LABEL}]" if demo else ""))
        if item.error:
            st.error(item.error)
        else:
            with st.container(border=True):
                st.markdown(email_markdown(item.email))
            with st.expander("Plain text to copy", key=key):
                st.code(item.email, language=None, wrap_lines=True)
    with match_column:
        show_matches(item)


def show_results(result: PageResult, demo: bool, run_number: int) -> None:
    found, shown = result.jobs_found, len(result.results)
    summary = f"Found {found} open role{'' if found == 1 else 's'} on the page."
    if found > shown:
        summary += f" Emails were written for the first {shown}."
    st.markdown(f"**{summary}**")
    if result.text_was_cut:
        st.caption(f"The page was long, so the app read only its first {MAX_PAGE_CHARS:,} characters.")

    labels = tab_labels([item.job.role for item in result.results])
    for index, (tab, item) in enumerate(zip(st.tabs(labels), result.results, strict=True)):
        with tab:
            show_job(item, demo, key=f"email-{run_number}-{index}")


def main() -> None:
    st.set_page_config(page_title="Cold Mail Generator", page_icon=":material/mail:", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    api_key = groq_api_key()
    demo = api_key is None
    model = "none (demo mode)" if demo else f"{model_name()} on Groq"
    sender = sender_sidebar(demo, model)

    st.title("Cold Mail Generator")
    st.caption(
        "Paste a careers page. Get one cold email per open role, "
        "backed by the closest projects in your portfolio."
    )
    if demo:
        st.info(
            "**Demo mode:** no Groq API key found, so the app uses a sample careers page and saved "
            "model outputs. Portfolio matching runs live.",
            icon=":material/science:",
        )

    url, submitted = page_form(demo)
    if submitted:
        st.session_state["run_number"] = st.session_state.get("run_number", 0) + 1
        st.session_state["result"] = run(url, sender, api_key)
    result = st.session_state.get("result")
    if result:
        show_results(result, demo, st.session_state["run_number"])


if __name__ == "__main__":
    main()
