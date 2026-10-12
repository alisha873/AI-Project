from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Literal

class ScenarioExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["manual", "scripted"]
    target_allowlist: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


class ScenarioManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    scenario_family: str = Field(min_length=1)
    description: str = Field(min_length=1)
    environment: Literal["isolated_docker_lab"]
    execution: ScenarioExecution
    ground_truth_reference: str = Field(min_length=1)
