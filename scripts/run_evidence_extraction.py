from __future__ import annotations

import json
from pathlib import Path

from agents.evidence_extraction import EvidenceExtractionAgent
from evidence.context import EvidenceContextBuilder
from llm.gemini import GeminiLLMClient
from llm.settings import LLMSettings


def make_evidence_llm_client(settings: LLMSettings | None = None) -> GeminiLLMClient:
    """Gemini client for evidence extraction. Does not read or fall back to a Groq key."""
    settings = settings or LLMSettings()
    provider = (settings.provider or "gemini").strip().lower()
    if provider != "gemini":
        raise ValueError(
            f"LLM_PROVIDER is {provider!r}. Evidence extraction uses Gemini. "
            "Set LLM_PROVIDER=gemini or leave it unset. GROQ_API_KEY is not used."
        )
    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Set it in the local .env file or the environment."
        )
    return GeminiLLMClient(api_key=settings.gemini_api_key)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    incident_dir = project_root / "incident_001"

    settings = LLMSettings()
    # Fail on provider/key configuration before any incident file is treated as a model prompt.
    llm_client = make_evidence_llm_client(settings)

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
