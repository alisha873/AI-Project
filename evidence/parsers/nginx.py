import re
from pathlib import Path
from urllib.parse import urlsplit

from evidence.parsers.base import EvidenceParser
from evidence.schema import EvidenceEvent


NGINX_ACCESS_PATTERN = re.compile(
    r"""
    ^
    (?P<client_ip>\S+)
    \s+\S+\s+\S+
    \s+\[(?P<nginx_timestamp>[^\]]+)\]
    \s+"(?P<method>[A-Z]+)\s+(?P<request_target>\S+)\s+HTTP/(?P<http_version>[^"]+)"
    \s+(?P<status>\d{3})
    \s+(?P<body_bytes>\d+)
    \s+"(?P<referer>[^"]*)"
    \s+"(?P<user_agent>[^"]*)"
    \s+upstream=(?P<upstream>\S+)
    \s+upstream_status=(?P<upstream_status>\S+)
    \s+request_time=(?P<request_time>\S+)
    $
    """,
    re.VERBOSE,
)


class NginxParser(EvidenceParser):
    """
    Parser for Docker-captured Nginx logs.

    The Docker timestamp is used as the canonical event timestamp.
    Nginx's internal timestamp is preserved as an attribute.

    Lines that are not Nginx access-log records are ignored.
    """

    source = "nginx"

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
        docker_timestamp, separator, nginx_line = line.partition(" ")

        if not separator:
            return None

        match = NGINX_ACCESS_PATTERN.match(nginx_line)

        if not match:
            return None

        data = match.groupdict()

        request_target = data["request_target"]
        parsed_target = urlsplit(request_target)

        upstream = data["upstream"]

        if upstream == "-":
            destination_entity = None
        elif ":" in upstream:
            destination_entity = upstream.rsplit(":", 1)[0]
        else:
            destination_entity = upstream

        upstream_status = data["upstream_status"]

        try:
            status = int(data["status"])
        except ValueError:
            status = None

        try:
            body_bytes = int(data["body_bytes"])
        except ValueError:
            body_bytes = None

        try:
            request_time = float(data["request_time"])
        except ValueError:
            request_time = None

        return EvidenceEvent(
            event_id=f"nginx-{line_number}",
            timestamp=docker_timestamp,
            source=self.source,
            event_type="http_request",
            source_entity=data["client_ip"],
            destination_entity=destination_entity,
            action=f"{data['method']} {parsed_target.path}",
            attributes={
                "method": data["method"],
                "path": parsed_target.path,
                "query": parsed_target.query,
                "http_version": data["http_version"],
                "status": status,
                "upstream": upstream,
                "upstream_status": upstream_status,
                "body_bytes": body_bytes,
                "request_time": request_time,
                "referer": data["referer"],
                "user_agent": data["user_agent"],
                "nginx_timestamp": data["nginx_timestamp"],
            },
        )