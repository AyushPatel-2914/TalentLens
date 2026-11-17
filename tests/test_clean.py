import math

from talentlens.data.clean import clean_text


def test_strips_tags_and_keeps_block_breaks():
    # </p><p> is a paragraph boundary -> one blank line; <br> -> one line break
    assert clean_text("<div><p>Hello</p><p>World</p></div>") == "Hello\n\nWorld"
    assert clean_text("Line one<br>Line two") == "Line one\nLine two"


def test_inline_tags_removed():
    assert clean_text("Skilled in <b>Python</b> and <span class='x'>SQL</span>") == "Skilled in Python and SQL"


def test_script_and_style_dropped():
    assert clean_text("<style>.a{color:red}</style>Resume<script>alert(1)</script>") == "Resume"


def test_html_entities_decoded():
    assert clean_text("R&amp;D at AT&amp;T") == "R&D at AT&T"


def test_mojibake_repaired():
    assert clean_text("donâ€™t") == "don't"


def test_plain_less_than_is_not_a_tag():
    assert clean_text("Experience <5 years and a < b") == "Experience <5 years and a < b"


def test_whitespace_collapsed():
    assert clean_text("  a   b\t\tc  \r\n\n\n\n\nd ") == "a b c\n\nd"


def test_zero_width_and_nbsp():
    assert clean_text("Py​thon Developer") == "Python Developer"


def test_non_strings_become_empty():
    assert clean_text(None) == ""
    assert clean_text(math.nan) == ""


def test_deterministic():
    text = "<p>Sénior  Engineer</p>&amp; lead"
    assert clean_text(text) == clean_text(text)
