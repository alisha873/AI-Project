from abc import ABC, abstractmethod
from pathlib import Path

from evidence.schema import EvidenceEvent


class EvidenceParser(ABC):
    """
    Base contract for all source-specific evidence parsers.

    Each parser converts raw evidence from one source into
    normalized EvidenceEvent objects.
    """

    source: str

    @abstractmethod
    def parse(self, path: Path) -> list[EvidenceEvent]:
        """
        Parse one evidence artifact and return normalized events.
        """
        raise NotImplementedError