"""Explicit output contract for AutoShield models used by explainability."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


OutputMeaning = Literal["risk_probability", "raw_score"]


class ModelOutputContract(BaseModel):
    """Document what a model's scalar output means.

    risk_probability must be a probability in [0, 1].
    raw_score is an arbitrary finite scalar and must not be described as
    a probability without a separate calibration/interpretation step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    output_meaning: OutputMeaning
    output_description: str = Field(min_length=1)

    @model_validator(mode="after")
    def description_must_be_explicit(self):
        if not self.output_description.strip():
            raise ValueError("output_description must not be blank")
        return self

    def validate_predictions(self, values: np.ndarray) -> np.ndarray:
        predictions = np.asarray(values, dtype=float)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        if predictions.ndim != 1:
            raise ValueError("Model must return one scalar output per input row.")
        if not np.isfinite(predictions).all():
            raise ValueError("Model returned non-finite predictions.")
        if self.output_meaning == "risk_probability":
            if ((predictions < 0) | (predictions > 1)).any():
                raise ValueError(
                    "risk_probability model outputs must be within [0, 1]."
                )
        return predictions

    def bind(
        self, predict_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap a model function so every output is checked against this contract."""

        def checked_predict_fn(data: np.ndarray) -> np.ndarray:
            return self.validate_predictions(predict_fn(data))

        return checked_predict_fn
