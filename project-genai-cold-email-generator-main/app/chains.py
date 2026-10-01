import os

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# The original tutorial used llama-3.1-70b-versatile, which Groq has retired.
# openai/gpt-oss-120b is a current Groq production model. Set GROQ_MODEL in
# your .env to use any other model from https://console.groq.com/docs/models.
DEFAULT_MODEL = "openai/gpt-oss-120b"

EXTRACT_PROMPT = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """
)

EMAIL_PROMPT = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
    the seamless integration of business processes through automated tools.
    Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
    process optimization, cost reduction, and heightened overall efficiency.
    Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ
    in fulfilling their needs.
    Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
    Remember you are Mohan, BDE at AtliQ.
    Do not provide a preamble.
    ### EMAIL (NO PREAMBLE):

    """
)


class Chain:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
            )
        self.llm = ChatGroq(
            model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
            temperature=0,
            api_key=api_key,
        )

    def extract_jobs(self, cleaned_text: str) -> list[dict]:
        chain_extract = EXTRACT_PROMPT | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            res = JsonOutputParser().parse(res.content)
        except OutputParserException as exc:
            raise OutputParserException("Context too big. Unable to parse jobs.") from exc
        return res if isinstance(res, list) else [res]

    def write_mail(self, job: dict, links: list) -> str:
        chain_email = EMAIL_PROMPT | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
