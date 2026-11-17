import pytest

from talentlens.data.pii import redact_pii


def test_email_removed_and_counted():
    text, counts = redact_pii("Contact: jane.doe+jobs@mail.example.co.uk today")
    assert text == "Contact: [EMAIL] today"
    assert counts["email"] == 1


@pytest.mark.parametrize(
    "phone",
    [
        "(555) 123-4567",
        "555-123-4567",
        "555.123.4567",
        "1-555-123-4567",
        "+1 555 123 4567",
        "+91 98765 43210",
        "+91-9876543210",
        "98765 43210",
        "9876543210",
        "+44 20 7946 0958",
    ],
)
def test_phone_formats_removed(phone):
    text, counts = redact_pii(f"Phone: {phone}.")
    assert text == "Phone: [PHONE].", text
    assert counts["phone"] == 1


@pytest.mark.parametrize(
    "safe",
    [
        "Worked 2010 - 2014 at Acme",
        "Jan 2015-2018",
        "01/2012 to 05/2016",
        "Salary $55,000 - $60,000",
        "Salary $60000-70000",
        "Zip 94105",
        "Managed a team of 25 and a budget of 1,200,000",
        "Python 3.10, ISO 9001:2015",
    ],
)
def test_dates_and_numbers_kept(safe):
    text, counts = redact_pii(safe)
    assert text == safe
    assert counts["phone"] == 0


def test_urls_removed():
    text, counts = redact_pii("See linkedin.com/in/janedoe and https://janedoe.dev/cv")
    assert text == "See [URL] and [URL]"
    assert counts["url"] == 2


def test_email_not_mistaken_for_url():
    text, counts = redact_pii("me@github.com")
    assert text == "[EMAIL]"
    assert counts == {"email": 1, "url": 0, "phone": 0}
