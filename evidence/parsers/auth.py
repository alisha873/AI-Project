import re
from pathlib import Path

from evidence.parsers.base import EvidenceParser
from evidence.schema import EvidenceEvent


AUTH_LOG_PATTERN = re.compile(
    r"""
    ^
    (?P<timestamp>\S+)
    \s+
    (?P<container_id>\S+)
    \s+
    sshd:\s+
    pam_unix\(sshd:auth\):\s+
    (?P<message>.*)
    $
    """,
    re.VERBOSE,
)

AUTH_ATTRIBUTE_PATTERN = re.compile(
    r"(?P<key>[A-Za-z_]+)=(?P<value>\S*)"
)


class AuthParser(EvidenceParser):
    source = "linux_auth"

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
        match = AUTH_LOG_PATTERN.match(line)

        if match is None:
            return None

        data = match.groupdict()
        message = data["message"]

        attributes = {
            key: value
            for key, value in AUTH_ATTRIBUTE_PATTERN.findall(message)
        }

        event_type = self._event_type(message)

        return EvidenceEvent(
            event_id=f"linux-auth-{line_number}",
            timestamp=data["timestamp"],
            source=self.source,
            event_type=event_type,
            source_entity=attributes.get("rhost"),
            destination_entity=None,
            action=self._action(message),
            attributes={
                "container_id": data["container_id"],
                "user": attributes.get("user"),
                "rhost": attributes.get("rhost"),
                "tty": attributes.get("tty"),
                "logname": attributes.get("logname"),
                "raw_message": message,
            },
        )

    @staticmethod
    def _event_type(message: str) -> str:
        if "authentication failure" in message:
            return "authentication_failure"

        if "session opened" in message:
            return "session_opened"

        if "session closed" in message:
            return "session_closed"

        return "authentication_event"

    @staticmethod
    def _action(message: str) -> str:
        if "authentication failure" in message:
            return "authentication_failed"

        if "session opened" in message:
            return "session_opened"

        if "session closed" in message:
            return "session_closed"

        return "authentication_event"