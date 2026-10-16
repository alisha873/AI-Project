import numpy as np
import pytest
from pydantic import ValidationError

from risk.model_contract import ModelOutputContract


def test_probability_contract_accepts_probabilities():
    contract = ModelOutputContract(
        model_name="example-risk-model",
        output_meaning="risk_probability",
        output_description="Probability of residual risk under the model.",
    )
    result = contract.validate_predictions(np.array([0.0, 0.4, 1.0]))
    assert np.allclose(result, [0.0, 0.4, 1.0])


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probability_contract_rejects_out_of_range_values(value):
    contract = ModelOutputContract(
        model_name="example-risk-model",
        output_meaning="risk_probability",
        output_description="Probability of residual risk under the model.",
    )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        contract.validate_predictions(np.array([value]))


def test_raw_score_contract_allows_finite_scores_outside_probability_range():
    contract = ModelOutputContract(
        model_name="example-score-model",
        output_meaning="raw_score",
        output_description="Uncalibrated scalar score; not a probability.",
    )
    assert contract.validate_predictions(np.array([-2.0, 3.0])).tolist() == [
        -2.0, 3.0
    ]


def test_contract_rejects_non_finite_output():
    contract = ModelOutputContract(
        model_name="example-risk-model",
        output_meaning="risk_probability",
        output_description="Probability of residual risk under the model.",
    )
    with pytest.raises(ValueError, match="non-finite"):
        contract.validate_predictions(np.array([np.nan]))


def test_contract_requires_model_metadata():
    with pytest.raises(ValidationError):
        ModelOutputContract(
            model_name="",
            output_meaning="raw_score",
            output_description="A score.",
        )


def test_bound_model_checks_outputs():
    contract = ModelOutputContract(
        model_name="example-risk-model",
        output_meaning="risk_probability",
        output_description="Probability of residual risk under the model.",
    )
    predict = contract.bind(lambda x: np.full(len(x), 0.3))
    assert predict(np.zeros((2, 7))).tolist() == [0.3, 0.3]
