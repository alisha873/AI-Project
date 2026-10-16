"""
A live HTTP transport for one approved origin. It is not a general-purpose URL tester.

Safety (fail closed):
  * constructed only for a normalized `http://host:port` origin whose host is `localhost` or a literal IP;
  * destination IPs are resolved (or taken from the literal) at send time, every request;
  * cloud-metadata, link-local, unspecified, multicast and globally-routable addresses are refused;
  * if the origin is localhost / loopback, every resolved address must be loopback (DNS-rebinding guard);
  * redirects are never followed;
  * TLS verification is not disabled (HTTPS is not used: approved origins are HTTP lab origins);
  * no retries;
  * no cookies, authorization headers or proxy settings from the environment;
  * response bodies are capped; oversized bodies become an error, not a truncated success.

The runner records body hash and length; this module never stores credentials.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Callable
from urllib.parse import urlsplit

import requests

from validation.probe_requests import ProbeRequest
from validation.scope import parse_target_origin
from validation.transport import MAX_BODY_CHARS, TransportResponse

LIVE_TRANSPORT_ID = "live-http-requests"
CONNECT_TIMEOUT_SECONDS = 3.0
# Cloud / instance-metadata destinations that must never be contacted, even if a caller names them.
_FORBIDDEN_NETWORKS = (
    ipaddress.ip_network("169.254.169.254/32"),
    ipaddress.ip_network("169.254.170.2/32"),
    ipaddress.ip_network("fd00:ec2::254/128"),
)


class LiveTransportConfigError(ValueError):
    """The transport was asked to target something it is not allowed to construct for."""


Resolver = Callable[[str], list[str]]
HttpGet = Callable[[str, dict[str, str], tuple[float, float]], requests.Response]


def _format_ip(address: str) -> str:
    parsed = ipaddress.ip_address(address)
    return f"[{parsed}]" if parsed.version == 6 else str(parsed)


def destination_forbidden(address: str, *, origin_host: str) -> str | None:
    """A non-secret reason the resolved address must not be contacted, or None if it is allowed."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return "resolved_address_unparseable"
    if ip.is_unspecified or ip.is_multicast or ip.is_link_local:
        return "destination_not_a_lab_address"
    if any(ip in network for network in _FORBIDDEN_NETWORKS):
        return "destination_forbidden"
    if ip.is_global:
        return "destination_is_public"
    if origin_host in ("localhost", "127.0.0.1", "::1"):
        return None if ip.is_loopback else "localhost_resolved_off_loopback"
    try:
        origin_ip = ipaddress.ip_address(origin_host)
    except ValueError:
        return "origin_host_not_literal"
    if ip != origin_ip:
        return "resolved_address_mismatch"
    return None


def _default_resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    addresses: list[str] = []
    for info in infos:
        address = info[4][0]
        if address not in addresses:
            addresses.append(address)
    return addresses


def _default_get(url: str, headers: dict[str, str], timeout: tuple[float, float]) -> requests.Response:
    session = requests.Session()
    session.trust_env = False  # ignore HTTP(S)_PROXY so the request cannot leave the approved destination
    try:
        return session.get(
            url, headers=headers, timeout=timeout, allow_redirects=False, verify=True,
        )
    finally:
        session.close()


class LiveHttpTransport:
    """Sends one GET `ProbeRequest` to the single approved origin it was constructed for. kind is always live."""

    kind = "live"

    def __init__(
        self,
        approved_origin: str,
        *,
        transport_id: str = LIVE_TRANSPORT_ID,
        connect_timeout_seconds: float = CONNECT_TIMEOUT_SECONDS,
        resolver: Resolver | None = None,
        http_get: HttpGet | None = None,
    ) -> None:
        parsed = parse_target_origin(approved_origin)
        if parsed is None:
            raise LiveTransportConfigError("approved_origin must be a normalized http://host:port origin")
        host, port = parsed
        if host != "localhost":
            try:
                ipaddress.ip_address(host)
            except ValueError as exc:
                raise LiveTransportConfigError(
                    "approved_origin host must be localhost or a literal IP address"
                ) from exc
        if destination_forbidden(host if host != "localhost" else "127.0.0.1", origin_host=host):
            raise LiveTransportConfigError("approved_origin is not an allowed lab destination")
        self.approved_origin = approved_origin
        self.transport_id = transport_id
        self.connect_timeout_seconds = connect_timeout_seconds
        self._host = host
        self._port = port
        self._resolver = resolver or _default_resolve
        self._http_get = http_get or _default_get

    def send(self, request: ProbeRequest) -> TransportResponse:
        if not isinstance(request, ProbeRequest):
            return TransportResponse.error()
        if request.origin != self.approved_origin or request.method != "GET":
            return TransportResponse.error()
        if request.path.startswith("/api/data"):
            return TransportResponse.error()
        try:
            parts = urlsplit(request.target)
        except ValueError:
            return TransportResponse.error()
        if parts.path != request.path or (parts.query and f"{request.path}?{request.query}" != request.target):
            return TransportResponse.error()

        destinations = self._resolve_destinations()
        if destinations is None:
            return TransportResponse.unreachable()
        destination = destinations[0]
        url = f"http://{_format_ip(destination)}:{self._port}{request.target}"
        headers = {"Host": self._host, "Accept": "application/json"}
        timeout = (self.connect_timeout_seconds, float(request.timeout_seconds))
        try:
            response = self._http_get(url, headers, timeout)
        except requests.Timeout:
            return TransportResponse.timeout()
        except requests.ConnectionError:
            return TransportResponse.unreachable()
        except requests.RequestException:
            return TransportResponse.error()

        location = response.headers.get("Location") or response.headers.get("location")
        if 300 <= response.status_code < 400 and location:
            # Never follow. A redirect off the approved origin is an error; an on-origin 3xx is still
            # returned as a response so the oracle can classify it inconclusive (non-200).
            if not self._location_stays_in_scope(location):
                return TransportResponse.error()

        try:
            body = response.text
        except Exception:  # noqa: BLE001 - an unreadable body is not a successful observation
            return TransportResponse.error()
        if len(body) > MAX_BODY_CHARS:
            return TransportResponse.error()
        if not 100 <= response.status_code <= 599:
            return TransportResponse.error()
        return TransportResponse(outcome="response", status_code=response.status_code, body=body)

    def _resolve_destinations(self) -> list[str] | None:
        if self._host != "localhost":
            reason = destination_forbidden(self._host, origin_host=self._host)
            return None if reason else [self._host]
        resolved = self._resolver(self._host)
        if not resolved:
            return None
        for address in resolved:
            if destination_forbidden(address, origin_host=self._host):
                return None
        return resolved

    def _location_stays_in_scope(self, location: str) -> bool:
        if location.startswith("/") and not location.startswith("//"):
            return True
        parsed = urlsplit(location)
        if parsed.scheme and parsed.scheme != "http":
            return False
        host = parsed.hostname
        port = parsed.port if parsed.port is not None else self._port
        if host is None:
            return False
        return host == self._host and port == self._port
