"""Tests for app.tools.desert.shape.shape_paginated."""

import json

from app.tools.desert.shape import shape_paginated


def _parse(s: str) -> dict:
    return json.loads(s)


def test_paginator_response_passes_through_total():
    """Laravel paginator with total=28 must yield total=28, NOT len(items)."""
    laravel_response = {
        "data": [{"id": i, "name": f"item-{i}"} for i in range(5)],
        "total": 28,
        "current_page": 1,
        "per_page": 100,
        "last_page": 1,
    }
    out = _parse(shape_paginated(laravel_response))
    assert out["total"] == 28
    assert out["total_known"] is True
    assert out["showing"] == 5
    assert out["page"] == 1
    assert out["per_page"] == 100
    assert out["last_page"] == 1
    assert len(out["items"]) == 5


def test_paginator_response_uses_custom_items_key():
    laravel_response = {
        "data": [{"id": 1}],
        "total": 1,
        "current_page": 1,
        "per_page": 50,
        "last_page": 1,
    }
    out = _parse(shape_paginated(laravel_response, items_key="equipment"))
    assert "equipment" in out
    assert "items" not in out
    assert len(out["equipment"]) == 1


def test_raw_list_response():
    """A bare list response should yield total = len(list)."""
    out = _parse(shape_paginated([{"id": 1}, {"id": 2}, {"id": 3}]))
    assert out["total"] == 3
    assert out["showing"] == 3
    assert out["total_known"] is True
    assert len(out["items"]) == 3


def test_dict_without_data_key_is_passed_through_raw():
    out = _parse(shape_paginated({"unexpected": "shape"}))
    assert "raw" in out


def test_total_missing_from_paginator_marks_total_unknown():
    response = {
        "data": [{"id": i} for i in range(3)],
        # no "total" field
    }
    out = _parse(shape_paginated(response))
    assert out["total_known"] is False
    assert out["total"] == 3  # falls back to len(items) when unknown


def test_truncation_preserves_total():
    """Big response should truncate items but keep `total` accurate."""
    big_items = [{"id": i, "blob": "x" * 500} for i in range(200)]
    response = {
        "data": big_items,
        "total": 200,
        "current_page": 1,
        "per_page": 200,
        "last_page": 1,
    }
    out = _parse(shape_paginated(response))
    assert out["total"] == 200
    assert out["total_known"] is True
    assert out["showing"] == 200  # what the server returned
    # Items in the response must be truncated (response > 24KB threshold)
    assert out.get("items_truncated") is True
    assert out["items_shown_in_response"] < 200
    assert len(out["items"]) == out["items_shown_in_response"]


def test_response_under_threshold_does_not_truncate():
    response = {
        "data": [{"id": i} for i in range(5)],
        "total": 5,
        "current_page": 1,
        "per_page": 100,
        "last_page": 1,
    }
    out = _parse(shape_paginated(response))
    assert "items_truncated" not in out
    assert len(out["items"]) == 5
