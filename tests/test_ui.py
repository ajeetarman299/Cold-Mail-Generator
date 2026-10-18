"""Small display helpers in the Streamlit UI."""

from main import email_markdown, plain, tab_labels


def test_tab_labels_number_repeated_roles():
    roles = ["Data Engineer", "Data Engineer", "Designer", "Data Engineer"]

    assert tab_labels(roles) == ["Data Engineer", "Data Engineer (2)", "Designer", "Data Engineer (3)"]


def test_plain_text_is_not_read_as_markdown():
    assert plain("**C#** _and_ [x](y)") == r"\*\*C\#\*\* \_and\_ \[x\]\(y\)"


def test_email_markdown_keeps_line_breaks_and_links_and_escapes_the_rest():
    email = "Subject: Hi\n\nSee https://example.com/ml-portfolio.\n**Not bold**\n\nBest regards,\nAlex"

    shown = email_markdown(email)

    assert shown.split("\n\n") == [
        r"Subject\: Hi",
        r"See <https://example.com/ml-portfolio>\.  " + "\n" + r"\*\*Not bold\*\*",
        "Best regards,  \nAlex",
    ]
