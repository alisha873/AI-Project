from __future__ import annotations

import json
from pathlib import Path

from agents.evidence_extraction import EvidenceExtractionAgent
from evidence.context import EvidenceContextBuilder
from llm.groq import GroqLLMClient
from llm.settings import LLMSettings


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    incident_dir = project_root / "incident_001"

    settings = LLMSettings()

    if settings.provider.lower() != "groq":
        raise ValueError(
            "LLM_PROVIDER must be set to 'groq' in .env"
        )

    api_key = settings.groq_api_key or settings.api_key

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY or LLM_API_KEY is missing. "
            "Check your local .env file."
        )

    # Load only the controlled evidence context.
    # Ground truth is excluded by EvidenceContextBuilder.
    context = EvidenceContextBuilder().build(incident_dir)

    print(f"Incident: {context.incident_id}")
    print(f"Normalized events available: {len(context.events)}")
    print(
        "Configuration artifacts available: "
        f"{len(context.configuration_artifacts)}"
    )
    print(
        "Network artifacts available: "
        f"{len(context.network_artifacts)}"
    )
    print(
        "Supporting artifacts available: "
        f"{len(context.evidence_artifacts)}"
    )

    # Make the real hosted-model call.
    llm_client = GroqLLMClient(
    model=settings.model,
    api_key=api_key,
    )

    agent = EvidenceExtractionAgent(llm_client=llm_client)
    extraction = agent.extract(context)

    output_dir = project_root / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "incident_001_extraction_v2_rerun.json"
    output_path.write_text(
        json.dumps(
            extraction.model_dump(mode="json"),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved extraction to: {output_path}")

    print("\n===== VALIDATED EVIDENCE EXTRACTION =====")
    print(
        json.dumps(
            extraction.model_dump(mode="json"),
            indent=2,
        )
    )
    print("\nEvidence extraction completed successfully.")


if __name__ == "__main__":
    main()
