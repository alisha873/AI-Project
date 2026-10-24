from remediation.text_checks import confirmation_claims


def test_negated_proof_is_not_reported_as_confirmation():
    text = (
        "No server-side vulnerability or successful query manipulation "
        "has been proven by the evidence."
    )
    assert confirmation_claims(text) == []


def test_positive_proof_is_still_reported():
    claims = confirmation_claims("The vulnerability has been proven.")
    assert any("has been proven" in claim.lower() for claim in claims)


def test_rather_than_confirmed_exploitation_is_not_reported():
    text = (
        "Suspicious requests represent observed activity rather than "
        "confirmed exploitation."
    )
    assert confirmation_claims(text) == []


def test_positive_confirmation_after_denial_is_still_reported():
    text = "No vulnerability has been proven, but exploitation is confirmed."
    claims = confirmation_claims(text)
    assert any("exploitation" in claim.lower() for claim in claims)
