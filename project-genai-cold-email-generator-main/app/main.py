import streamlit as st

from chains import Chain
from portfolio import Portfolio
from utils import clean_text, fetch_page_text


@st.cache_resource
def get_chain() -> Chain:
    return Chain()


@st.cache_resource
def get_portfolio() -> Portfolio:
    portfolio = Portfolio()
    portfolio.load_portfolio()
    return portfolio


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input(
        "Enter a URL:",
        placeholder="https://careers.example.com/job/12345",
        help="A careers page or a single job posting. The app extracts each role and drafts one email per role.",
    )
    submit_button = st.button("Submit", type="primary")

    if submit_button:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
            return
        try:
            with st.spinner("Reading the page and extracting jobs..."):
                data = clean_text(fetch_page_text(url_input.strip()))
                jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get("skills", [])
                links = portfolio.query_links(skills)
                with st.spinner(f"Writing email for {job.get('role', 'this role')}..."):
                    email = llm.write_mail(job, links)
                st.code(email, language="markdown")
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    try:
        chain = get_chain()
        portfolio = get_portfolio()
    except Exception as e:
        st.error(f"Setup failed: {e}")
        st.stop()
    create_streamlit_app(chain, portfolio, clean_text)
