"""Demo mode: with no API key the app runs on the bundled sample page and saved outputs."""

import json
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import portfolio as portfolio_module
from chains import Chain, Sender, groq_api_key, parse_jobs
from demo import DEMO_LABEL, demo_llm, load_demo_outputs, sample_page_html
from pipeline import generate
from portfolio import DEFAULT_CSV, Portfolio, load_portfolio_csv
from utils import page_text_from_html

APP = Path(__file__).resolve().parent.parent / "app" / "main.py"
SAMPLE_ROLES = [
    "Machine Learning Engineer, Recommendations",
    "Android Developer",
    "Frontend Engineer, E-commerce",
]
MINILM = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"


@pytest.mark.parametrize("value", ["", "   ", "your_groq_api_key_here"])
def test_a_missing_or_placeholder_key_means_demo_mode(monkeypatch, value):
    monkeypatch.setenv("GROQ_API_KEY", value)

    assert groq_api_key() is None


def test_a_set_key_means_live_mode(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-value-not-a-real-key")

    assert groq_api_key() == "test-value-not-a-real-key"


def test_demo_runs_the_whole_pipeline_on_the_sample_page(portfolio):
    result = generate(sample_page_html(), Chain(demo_llm()), portfolio, Sender())

    assert result.jobs_found == 3
    assert not result.text_was_cut
    assert [item.job.role for item in result.results] == SAMPLE_ROLES
    for item in result.results:
        assert item.email.startswith("Subject:")
        assert not item.error
        assert item.matches


def test_saved_jobs_come_from_the_sample_page():
    text, _ = page_text_from_html(sample_page_html())
    jobs = parse_jobs(json.dumps(load_demo_outputs()["extraction"]))

    for job in jobs:
        assert job.role in text
        for skill in job.skills:
            assert skill.lower() in text.lower()


def test_saved_emails_only_cite_links_from_the_portfolio():
    known = set(load_portfolio_csv(DEFAULT_CSV)["Links"])

    for email in load_demo_outputs()["emails"].values():
        links = re.findall(r"https?://\S+", email)
        assert links
        assert set(links) <= known


def test_saved_emails_are_signed_by_the_default_sender():
    outputs = load_demo_outputs()
    sender = Sender()

    assert outputs["sender"] == sender.model_dump()
    for email in outputs["emails"].values():
        assert email.rstrip().endswith(f"{sender.name}\n{sender.role}\n{sender.company}")


@pytest.mark.skipif(not MINILM.exists(), reason="ChromaDB's embedding model is not downloaded yet")
def test_saved_emails_cite_links_that_the_real_matcher_finds(tmp_path):
    portfolio = Portfolio(vectorstore_path=tmp_path / "vectorstore")
    outputs = load_demo_outputs()

    for job in parse_jobs(json.dumps(outputs["extraction"])):
        found = {match.link for match in portfolio.match(job.skills)}
        cited = set(re.findall(r"https?://\S+", outputs["emails"][job.role]))
        assert cited <= found, job.role


def test_the_app_starts_and_runs_the_demo_without_a_key(no_api_key, monkeypatch, tmp_path, keyword_embedding):
    monkeypatch.setattr(portfolio_module, "default_embedding_function", lambda: keyword_embedding)
    monkeypatch.setattr(portfolio_module, "DEFAULT_VECTORSTORE", tmp_path / "vectorstore")
    st.cache_resource.clear()

    app = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not app.exception
    assert "Demo mode" in app.info[0].value
    assert app.sidebar.text_input[0].disabled  # sender settings are fixed in demo mode

    app.button[0].click().run()  # "Write emails"

    assert not app.exception
    assert len(app.tabs) == 3
    assert [tab.label for tab in app.tabs] == SAMPLE_ROLES
    shown = [block.value for block in app.markdown]
    assert sum(DEMO_LABEL in text for text in shown) == 3  # every email is labelled
    assert sum("Matched portfolio links" in text for text in shown) == 3
    assert any("ml-python-portfolio" in text and "distance" in text for text in shown)
    assert app.code[0].value.startswith("Subject:")  # plain text to copy
    st.cache_resource.clear()
