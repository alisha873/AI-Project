from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from evidence.schema import EvidenceEvent


class EvidenceEventWriter:
    """
    Serializes normalized EvidenceEvents to JSON Lines format.
    """

    def write(
        self,
        events: Iterable[EvidenceEvent],
        output_path: Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for event in events:
                payload = event.model_dump(mode="json")

                handle.write(
                    json.dumps(
                        payload,
                        sort_keys=True,
                    )
                )
                handle.write("\n")