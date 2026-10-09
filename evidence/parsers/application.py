import re
from pathlib import Path
from urllib.parse import urlsplit

from evidence.parsers.base import EvidenceParser
from evidence.schema import EvidenceEvent


APPLICATION_LOG_PATTERN = re.compile(
    r"""
    ^
    (?P<docker_timestamp>\S+)
    \s+INFO:\s+
    (?P<client_ip>[^:]+):(?P<client_port>\d+)
    \s+-\s+
    "(?P<method>[A-Z]+)
    \s+
    (?P<request_target>\S+)
    \s+
    HTTP/(?P<http_version>[^"]+)"
    \s+
    (?P<status>\d{3})
    \s+
    (?P<status_text>.*)
    $
    """,
    re.VERBOSE,
)


class ApplicationParser(EvidenceParser):
    source = "application"

    def parse(self, path: Path) -> list[EvidenceEvent]:
        events: list[EvidenceEvent] = []

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\n")
                event = self._parse_line(line, line_number)

                if event is not None:
                    events.append(event)

        return events

    def _parse_line(
        self,
        line: str,
        line_number: int,
    ) -> EvidenceEvent | None:
        match = APPLICATION_LOG_PATTERN.match(line)

        if match is None:
            return None

        data = match.groupdict()
        request_target = data["request_target"]
        parsed_target = urlsplit(request_target)

        return EvidenceEvent(
            event_id=f"application-{line_number}",
            timestamp=data["docker_timestamp"],
            source=self.source,
            event_type="http_request",
            source_entity=data["client_ip"],
            destination_entity=None,
            action=f"{data['method']} {parsed_target.path}",
            attributes={
                "client_port": int(data["client_port"]),
                "method": data["method"],
                "path": parsed_target.path,
                "query": parsed_target.query,
                "http_version": data["http_version"],
                "status": int(data["status"]),
                "status_text": data["status_text"],
            },
        )