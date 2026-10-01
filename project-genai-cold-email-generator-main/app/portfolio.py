import uuid
from pathlib import Path

import chromadb
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "resource" / "my_portfolio.csv"
DEFAULT_VECTORSTORE = APP_DIR.parent / "vectorstore"


class Portfolio:
    def __init__(self, file_path=DEFAULT_CSV, vectorstore_path=DEFAULT_VECTORSTORE):
        self.file_path = Path(file_path)
        self.data = pd.read_csv(self.file_path)
        self.chroma_client = chromadb.PersistentClient(path=str(vectorstore_path))
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """Embed every portfolio row into ChromaDB once. Later runs reuse the stored vectors."""
        if not self.collection.count():
            self.collection.add(
                documents=[str(stack) for stack in self.data["Techstack"]],
                metadatas=[{"links": str(link)} for link in self.data["Links"]],
                ids=[str(uuid.uuid4()) for _ in range(len(self.data))],
            )

    def query_links(self, skills):
        """Return the two closest portfolio links for each skill."""
        if isinstance(skills, str):
            skills = [skills]
        if not skills:
            return []
        return self.collection.query(query_texts=list(skills), n_results=2).get("metadatas", [])
