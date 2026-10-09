from pathlib import Path

from evidence.parsers.nginx import NginxParser


ACCESS_LINE = (
    '2026-10-09T11:07:46.241152552Z '
    '192.168.65.1 - - [09/Oct/2026:11:07:46 +0000] '
    '"GET /api/search?message=%27%20OR%20%271%27%3D%271 HTTP/1.1" '
    '200 113 "-" "curl/8.1.2" '
    'upstream=172.19.0.6:8000 '
    'upstream_status=200 '
    'request_time=0.057'
)


def test_parse_nginx_access_line(tmp_path: Path):
    log_file = tmp_path / "nginx.log"
    log_file.write_text(ACCESS_LINE + "\n", encoding="utf-8")

    events = NginxParser().parse(log_file)

    assert len(events) == 1

    event = events[0]

    assert event.event_id == "nginx-1"
    assert event.source == "nginx"
    assert event.event_type == "http_request"

    assert event.source_entity == "192.168.65.1"
    assert event.destination_entity == "172.19.0.6"

    assert event.action == "GET /api/search"

    assert event.attributes["method"] == "GET"
    assert event.attributes["path"] == "/api/search"
    assert event.attributes["status"] == 200
    assert event.attributes["upstream"] == "172.19.0.6:8000"
    assert event.attributes["upstream_status"] == "200"
    assert event.attributes["request_time"] == 0.057


def test_parse_nginx_startup_line_is_ignored(tmp_path: Path):
    log_file = tmp_path / "nginx.log"

    log_file.write_text(
        "2026-10-09T10:29:17.911742304Z "
        "/docker-entrypoint.sh: Configuration complete; ready for start up\n",
        encoding="utf-8",
    )

    events = NginxParser().parse(log_file)

    assert events == []


def test_parse_multiple_nginx_events(tmp_path: Path):
    log_file = tmp_path / "nginx.log"

    log_file.write_text(
        "\n".join(
            [
                ACCESS_LINE,
                ACCESS_LINE.replace(
                    "GET /api/search",
                    "GET /api/health",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = NginxParser().parse(log_file)

    assert len(events) == 2
    assert events[0].action == "GET /api/search"
    assert events[1].action == "GET /api/health"