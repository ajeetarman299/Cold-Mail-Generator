# Cold Mail Generator

Paste a company's careers page URL and get a personalised cold email for each open role, written by an LLM on Groq and backed by relevant portfolio links retrieved from a ChromaDB vector store.

![App screenshot](project-genai-cold-email-generator-main/imgs/img.png)

## What it does

1. Downloads the careers page and extracts each job posting (role, experience, skills, description) as JSON with an LLM.
2. Matches the job's skills against a portfolio of tech stacks stored in ChromaDB.
3. Writes a cold email from a services company to the hiring team, citing the most relevant portfolio links.

Built with Python, Streamlit, LangChain, Groq and ChromaDB.

## Results

The recorded notebook run on a Nike posting ("Software Engineer II, AI/ML Platforms") extracted the role, experience, five skills and a description as JSON, and the generated email cited the Machine Learning (Python) and DevOps portfolio links as the most relevant work.

## Where to find it

All code, notebooks and assets live in [`project-genai-cold-email-generator-main/`](project-genai-cold-email-generator-main/). Its [README](project-genai-cold-email-generator-main/README.md) covers the approach, project structure and setup.

Quick start:

```bash
cd project-genai-cold-email-generator-main
cp .env.example .env            # add your GROQ_API_KEY
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app/main.py
```

## Credits

Based on the [codebasics](https://www.youtube.com/@codebasics) GenAI tutorial and their repository [codebasics/project-genai-cold-email-generator](https://github.com/codebasics/project-genai-cold-email-generator), used under the terms in its [LICENSE](project-genai-cold-email-generator-main/LICENSE) and README. This copy updates it to current library versions and a current Groq model.
