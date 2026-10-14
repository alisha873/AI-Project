"""Deterministic features extracted from observed HTTP graph edges."""

import re
from collections.abc import Mapping
from urllib.parse import unquote


METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH", "OTHER")
PATH_CATEGORIES = (
    "api_search",
    "login",
    "ftp",
    "product_search",
    "root",
    "other",
)
STATUS_CLASSES = ("2xx", "3xx", "4xx", "5xx", "other")

# 6 method + 6 path + 5 status + 6 request indicators = 23 features.
HTTP_FEATURE_DIM = (
    len(METHODS) + len(PATH_CATEGORIES) + len(STATUS_CLASSES) + 6
)


def extract_http_features(attributes: Mapping[str, object]) -> list[float]:
    """Encode observed HTTP attributes without consulting ground truth."""

    method = str(attributes.get("method") or "").upper()
    path = str(attributes.get("path") or "/")
    query = str(attributes.get("query") or "")
    status_raw = str(attributes.get("status") or "")

    decoded = query
    for _ in range(2):
        decoded = unquote(decoded)
    lowered = decoded.lower()
    normalized_path = path.lower().split("?", 1)[0]

    method_category = method if method in METHODS[:-1] else "OTHER"

    if normalized_path == "/api/search":
        path_category = "api_search"
    elif "login" in normalized_path:
        path_category = "login"
    elif normalized_path.startswith("/ftp"):
        path_category = "ftp"
    elif "product" in normalized_path:
        path_category = "product_search"
    elif normalized_path in ("", "/"):
        path_category = "root"
    else:
        path_category = "other"

    try:
        status_code = int(status_raw)
    except (TypeError, ValueError):
        status_code = 0

    if 200 <= status_code < 300:
        status_category = "2xx"
    elif 300 <= status_code < 400:
        status_category = "3xx"
    elif 400 <= status_code < 500:
        status_category = "4xx"
    elif 500 <= status_code < 600:
        status_category = "5xx"
    else:
        status_category = "other"

    sql_like = bool(re.search(
        r"\b(union|select|insert|update|delete)\b|"
        r"'\s*(or|and)\b|--|/\*|\b1\s*=\s*1\b",
        lowered,
    ))
    xss_like = bool(re.search(
        r"<\s*script|<\s*svg|\bonload\s*=|alert\s*\(",
        lowered,
    ))
    traversal_like = bool(re.search(r"\.\.[/\\]", lowered))
    special_chars = any(char in decoded for char in ("'", '"', "<", ">", ";"))
    query_present = bool(query)
    query_length = min(len(query), 512) / 512.0

    one_hot = (
        [float(method_category == item) for item in METHODS]
        + [float(path_category == item) for item in PATH_CATEGORIES]
        + [float(status_category == item) for item in STATUS_CLASSES]
    )

    indicators = [
        float(query_present),
        float(sql_like),
        float(xss_like),
        float(traversal_like),
        float(special_chars),
        query_length,
    ]

    features = one_hot + indicators
    assert len(features) == HTTP_FEATURE_DIM
    return features
