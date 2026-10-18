"""Portfolio matching on the bundled portfolio CSV, with an offline stand-in for the embedding model."""

import pytest

from portfolio import DEFAULT_CSV, Portfolio, load_portfolio_csv


def test_match_puts_the_closest_entry_first(portfolio):
    matches = portfolio.match(["Machine Learning", "TensorFlow"])

    assert matches[0].link == "https://example.com/ml-python-portfolio"
    assert matches[0].techstack == "Machine Learning, Python, TensorFlow"
    assert matches[0].skills == ["Machine Learning", "TensorFlow"]
    assert [match.distance for match in matches] == sorted(match.distance for match in matches)


def test_match_lists_each_link_once_with_every_skill_that_found_it(portfolio):
    matches = portfolio.match(["Kotlin", "Android", "Firebase"])

    links = [match.link for match in matches]
    assert len(links) == len(set(links))
    kotlin_android = next(match for match in matches if match.link.endswith("/kotlin-android-portfolio"))
    assert set(kotlin_android.skills) == {"Kotlin", "Android", "Firebase"}


def test_match_keeps_the_lowest_distance_for_a_link(portfolio):
    both = portfolio.match(["Machine Learning", "TensorFlow"])[0]
    one = portfolio.match(["Machine Learning"])[0]

    assert both.link == one.link
    assert both.distance == pytest.approx(one.distance)


def test_match_drops_entries_beyond_the_distance_cutoff(portfolio):
    assert portfolio.match(["Quantum chemistry"]) == []
    assert portfolio.match(["Quantum chemistry"], max_distance=None) != []


def test_match_with_no_skills_returns_nothing(portfolio):
    assert portfolio.match([]) == []
    assert portfolio.match(["", "  "]) == []


def test_store_follows_edits_to_the_csv(tmp_path, keyword_embedding):
    csv = tmp_path / "portfolio.csv"
    store = tmp_path / "vectorstore"
    csv.write_text('Techstack,Links\n"Python, Django",https://example.com/a\n"Go, gRPC",https://example.com/b\n')
    first = Portfolio(csv_path=csv, vectorstore_path=store, embedding_function=keyword_embedding)
    assert first.collection.count() == 2

    csv.write_text('Techstack,Links\n"Python, Django",https://example.com/a\n"Rust, Tokio",https://example.com/c\n')
    second = Portfolio(csv_path=csv, vectorstore_path=store, embedding_function=keyword_embedding)

    assert second.collection.count() == 2
    assert [match.link for match in second.match(["Rust"])] == ["https://example.com/c"]
    assert "https://example.com/b" not in [match.link for match in second.match(["gRPC"], max_distance=None)]


def test_csv_must_have_both_columns(tmp_path):
    csv = tmp_path / "portfolio.csv"
    csv.write_text("Stack,URL\nPython,https://example.com/a\n")

    with pytest.raises(ValueError, match="Techstack and Links"):
        load_portfolio_csv(csv)


def test_csv_blank_and_repeated_rows_are_dropped(tmp_path):
    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Techstack,Links\n"
        "Python,https://example.com/a\n"
        " Python ,https://example.com/a\n"
        ",https://example.com/b\n"
        "Go,\n"
    )

    assert load_portfolio_csv(csv).values.tolist() == [["Python", "https://example.com/a"]]


def test_the_bundled_portfolio_has_twenty_unique_links():
    data = load_portfolio_csv(DEFAULT_CSV)

    assert len(data) == 20
    assert data["Links"].is_unique
