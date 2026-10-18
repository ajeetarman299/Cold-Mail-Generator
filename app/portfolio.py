"""Portfolio store: keeps the portfolio CSV in ChromaDB and finds the closest entries for a job's skills."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "my_portfolio.csv"
DEFAULT_VECTORSTORE = ROOT / "vectorstore"
COLLECTION_NAME = "portfolio"
LINKS_PER_SKILL = 2

# Matches further away than this are dropped. Picked by reading the matches for the
# sample jobs with the default embedding model: clearly related entries scored under
# 0.65, and most entries above it were unrelated (for example "A/B testing" matched
# to an iOS project).
MAX_DISTANCE = 0.65


@dataclass
class PortfolioMatch:
    """A portfolio entry that matched one or more of a job's skills."""

    link: str
    techstack: str
    # Lowest cosine distance between the entry and any of the skills that found it.
    # 0 means identical; lower is closer.
    distance: float
    skills: list[str] = field(default_factory=list)


def load_portfolio_csv(path: str | Path) -> pd.DataFrame:
    """Read the portfolio CSV (columns Techstack and Links), dropping blank and repeated rows."""
    data = pd.read_csv(path)
    missing = {"Techstack", "Links"} - set(data.columns)
    if missing:
        raise ValueError(
            f"{Path(path).name} needs the columns Techstack and Links "
            f"(missing: {', '.join(sorted(missing))})."
        )
    data = data[["Techstack", "Links"]].dropna().astype(str)
    data = data.apply(lambda column: column.str.strip())
    data = data[(data["Techstack"] != "") & (data["Links"] != "")]
    return data.drop_duplicates().reset_index(drop=True)


def row_id(techstack: str, link: str) -> str:
    """A stable id for a portfolio row, so the store can tell which rows changed."""
    return hashlib.sha256(f"{techstack}\n{link}".encode()).hexdigest()[:16]


def default_embedding_function():
    """ChromaDB's built-in embedding model (all-MiniLM-L6-v2, run locally with ONNX).

    It needs no API key. ChromaDB downloads it on first use and caches it.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


class Portfolio:
    """The portfolio CSV, embedded in a persistent ChromaDB collection."""

    def __init__(self, csv_path=None, vectorstore_path=None, embedding_function=None):
        self.data = load_portfolio_csv(csv_path or DEFAULT_CSV)
        client = chromadb.PersistentClient(path=str(vectorstore_path or DEFAULT_VECTORSTORE))
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function or default_embedding_function(),
            configuration={"hnsw": {"space": "cosine"}},
        )
        self.sync()

    def sync(self) -> None:
        """Make the store match the CSV: embed new rows and remove deleted ones.

        Editing the CSV is enough; there is no need to delete the vectorstore folder.
        """
        rows = {
            row_id(techstack, link): (techstack, link)
            for techstack, link in self.data.itertuples(index=False)
        }
        stored = set(self.collection.get(include=[])["ids"])
        removed = sorted(stored - rows.keys())
        if removed:
            self.collection.delete(ids=removed)
        added = [row for row in rows if row not in stored]
        if added:
            self.collection.add(
                ids=added,
                documents=[rows[row][0] for row in added],
                metadatas=[{"techstack": rows[row][0], "link": rows[row][1]} for row in added],
            )

    def match(
        self,
        skills: list[str],
        per_skill: int = LINKS_PER_SKILL,
        max_distance: float | None = MAX_DISTANCE,
    ) -> list[PortfolioMatch]:
        """Find the closest portfolio entries for each skill.

        Returns one row per link, closest first, with the skills that found it.
        Entries further than max_distance from every skill are left out.
        """
        skills = list(dict.fromkeys(skill.strip() for skill in skills if skill and skill.strip()))
        count = self.collection.count()
        if not skills or not count:
            return []
        result = self.collection.query(
            query_texts=skills,
            n_results=min(per_skill, count),
            include=["metadatas", "distances"],
        )
        matches: dict[str, PortfolioMatch] = {}
        for skill, metadatas, distances in zip(skills, result["metadatas"], result["distances"], strict=True):
            for metadata, distance in zip(metadatas, distances, strict=True):
                if max_distance is not None and distance > max_distance:
                    continue
                link = metadata["link"]
                match = matches.setdefault(
                    link, PortfolioMatch(link, metadata["techstack"], float(distance))
                )
                match.distance = min(match.distance, float(distance))
                if skill not in match.skills:
                    match.skills.append(skill)
        return sorted(matches.values(), key=lambda match: (match.distance, match.link))
