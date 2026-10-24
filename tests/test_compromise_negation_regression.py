from remediation.text_checks import compromise_claims


def test_explicit_no_confirmation_does_not_claim_successful_exploitation():
    text = (
        "The observed evidence consists of SQL-injection-like request patterns. "
        "There is no confirmation of vulnerable server-side query construction, "
        "successful exploitation, or database access."
    )
    claims = compromise_claims(text)
    assert not any("successful exploitation" in c.lower() for c in claims)
    assert not any("database access" in c.lower() for c in claims)


def test_positive_successful_exploitation_is_still_detected():
    claims = compromise_claims("Successful exploitation occurred.")
    assert any("successful exploitation" in c.lower() for c in claims)


def test_positive_claim_after_but_is_not_suppressed_by_denial():
    text = (
        "There is no confirmation of successful exploitation, "
        "but successful exploitation occurred."
    )
    claims = compromise_claims(text)
    assert any("successful exploitation" in c.lower() for c in claims)
