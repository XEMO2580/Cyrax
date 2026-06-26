"""
core/trace.py — Trace ID generation for CYRAX 3.0

Single source of truth for correlation ID format.
Every inbound event receives a Trace ID at the app/ boundary.
All downstream log lines must carry this ID for async correlation.

No dependencies on any CYRAX domain module.
"""

import uuid


_TRACE_ID_LENGTH = 12  # 12 hex chars = 48 bits of entropy, readable in logs


def new_trace_id() -> str:
    """
    Generates a short, URL-safe, unique correlation ID.

    Format: 12 lowercase hex characters (e.g. 'a3f9c2e10b44')
    Collision probability at 1M events/day: negligible for a single-device OS.

    If a distributed deployment ever requires longer IDs or UUIDs,
    change this function — all callers update automatically.
    """
    return uuid.uuid4().hex[:_TRACE_ID_LENGTH]