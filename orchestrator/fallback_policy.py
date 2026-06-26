"""
orchestrator/fallback_policy.py — CYRAX 3.0 Declarative Fallback Rules

Single source of truth for tool failure recovery decisions.
The Dispatcher queries this object instead of encoding recovery logic inline.

To add a new fallback rule:
  - Add an entry to _FALLBACK_TABLE.
  - No changes to dispatcher.py are required.

Table schema:
  (failed_tool, error_status) -> {
      "tool":          str,           # fallback tool to execute
      "args_from":     str | None,    # key in original_args to forward
      "args_override": dict | None,   # static args to merge into the call
  }
"""

from __future__ import annotations

from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK TABLE
# ══════════════════════════════════════════════════════════════════════════════

# Key:   (failed_tool_name, error_status)
# Value: fallback descriptor dict
#
# "args_from":     name of the key in original_args whose value becomes
#                  the fallback tool's primary argument.
# "args_key":      name of the argument key the fallback tool expects.
# "args_override": additional static arguments merged into the fallback call.

_FALLBACK_TABLE: dict[tuple[str, str], dict[str, Any]] = {
    ("OPEN_APP", "error"): {
        "tool":         "WEB_SEARCH",
        "args_from":    "app",
        "args_key":     "query",
        "args_override": {},
    },
    ("DELETE_FILE", "error"): {
        "tool":         "WEB_SEARCH",
        "args_from":    "filepath",
        "args_key":     "query",
        "args_override": {"query": "how to delete file"},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK POLICY
# ══════════════════════════════════════════════════════════════════════════════

class FallbackPolicy:
    """
    Satisfies FallbackPolicyProtocol defined in core/context.py.

    Stateless — safe to instantiate once in bootstrap() and share across
    all requests via CyraxContext.
    """

    def get_fallback(
        self,
        failed_tool: str,
        error_status: str,
        original_args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Returns a fallback execution descriptor when a recovery rule exists,
        or None when the failure should be escalated without retry.

        Return shape (when a fallback exists):
            {
                "tool": str,       # name of the fallback tool to execute
                "args": dict,      # arguments to pass to that tool
            }

        The dispatcher passes this directly to _execute_step — it does not
        inspect which tools are named here.
        """
        descriptor = _FALLBACK_TABLE.get((failed_tool, error_status))
        if descriptor is None:
            return None

        args_from_key: str | None = descriptor.get("args_from")
        args_key: str | None = descriptor.get("args_key")
        args_override: dict = descriptor.get("args_override", {})

        fallback_args: dict[str, Any] = {}

        if args_from_key and args_key:
            source_value = original_args.get(args_from_key, "")
            if source_value:
                fallback_args[args_key] = source_value

        fallback_args.update(args_override)

        return {
            "tool": descriptor["tool"],
            "args": fallback_args,
        }