# Cold Mail Generator

A Streamlit app that reads a company's careers page, extracts the open roles with an LLM on Groq, and drafts a personalised cold email for each role with matching portfolio links retrieved from ChromaDB.

![App screenshot](imgs/img.png)

## Scenario

- Nike is hiring a Software Engineer II for its AI/ML Platforms team and is spending time and resources on hiring, onboarding and training.
- AtliQ is a software development company that can provide a dedicated engineer to Nike. Its business development executive, Mohan, reaches out to Nike with a cold email.

The app automates Mohan's job: paste the job URL, get a ready-to-send email that points to the most relevant work in AtliQ's portfolio.

## Results

The notebook `email_generator.ipynb` records a full run on a Nike posting, made with the original tutorial model `llama-3.1-70b-versatile`:

- Extraction: the LLM turned the scraped page into JSON with `role` ("Software Engineer II, AI/ML Platforms"), `experience` ("2+ years professional software development experience"), five `skills` and a `description`.
- Retrieval: each skill was matched against the 20 sample tech stacks in `my_portfolio.csv`, returning the two closest portfolio links per skill.
- Generation: the email cited the Machine Learning (Python) and DevOps portfolio links as the most relevant work.

The screenshot above shows the same flow in the Streamlit app.

## Approach

![Architecture diagram](imgs/architecture.png)

1. Scrape: download the careers page and keep only its visible text (requests and BeautifulSoup), then clean it.
2. Extract: a LangChain prompt asks the Groq LLM for the job postings as JSON with `role`, `experience`, `skills` and `description`.
3. Retrieve: the portfolio CSV (tech stack and link per row) is embedded once into a persistent ChromaDB collection. Each of the job's skills is queried against it to find its two closest portfolio links.
4. Write: a second prompt writes the cold email as Mohan from AtliQ, using the job description and the retrieved links.

## Tech stack

- Python 3.11 or newer
- Streamlit for the UI
- LangChain (`langchain-core`, `langchain-groq`) for prompts, chaining and JSON parsing
- Groq as the LLM provider, default model `openai/gpt-oss-120b` (configurable)
- ChromaDB as the local vector store, with its built-in embedding model
- pandas, requests, BeautifulSoup and python-dotenv

## Project structure

```
project-genai-cold-email-generator-main/
    app/
        main.py              Streamlit UI and request flow
        chains.py            Groq LLM, extraction and email prompts
        portfolio.py         ChromaDB portfolio store and link lookup
        utils.py             Page download and text cleaning
        resource/
            my_portfolio.csv Portfolio used by the app
    email_generator.ipynb    End-to-end walkthrough (original tutorial run)
    tutorial_groq.ipynb      First Groq call
    tutorial_chromadb.ipynb  ChromaDB basics: add, get, query, delete
    my_portfolio.csv         Portfolio used by the notebooks
    LICENSE                  MIT licence from the original project
    imgs/                    Screenshot and architecture diagram
    .env.example             Environment variables to copy into .env
    requirements.txt         Pinned dependencies
```

## How to run

1. Get a free API key from https://console.groq.com/keys.

2. From this folder, create your environment file and add the key:

   ```bash
   cp .env.example .env
   # then edit .env and set GROQ_API_KEY
   ```

3. Install the dependencies, either with uv:

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

   or with pip:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Start the app from this folder and paste a careers page or job posting URL:

   ```bash
   streamlit run app/main.py
   ```

On first run ChromaDB downloads its default embedding model and builds the `vectorstore/` folder. Later runs reuse it. To change the portfolio, edit `app/resource/my_portfolio.csv` and delete `vectorstore/` so it is rebuilt.

### Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | Groq API key |
| `GROQ_MODEL` | No | Any current model id from https://console.groq.com/docs/models. Defaults to `openai/gpt-oss-120b`. |
| `USER_AGENT` | No | User-Agent header used when downloading job pages |

### About the notebooks

The notebooks are the original tutorial walkthrough and keep their recorded outputs. They were run with `llama-3.1-70b-versatile`, which Groq has since retired, and they use `WebBaseLoader` from `langchain-community`, which LangChain is sunsetting. To re-run them from this folder, change the model name to a current one, install `langchain-community` and `jupyter` alongside the requirements, and export `GROQ_API_KEY` in your shell (the notebooks do not read `.env`). The app itself needs none of these changes.

## Credits

This project follows the [codebasics](https://www.youtube.com/@codebasics) GenAI tutorial and is based on their repository [codebasics/project-genai-cold-email-generator](https://github.com/codebasics/project-genai-cold-email-generator). The scenario, prompts, notebooks, portfolio data and images come from that tutorial. This copy updates the code to current library versions and a current Groq model.

Copyright (C) Codebasics Inc. All rights reserved.

**Additional Terms:**
This software is licensed under the MIT License. However, commercial use of this software is strictly prohibited without prior written permission from the author. Attribution must be given in all copies or substantial portions of the software.
