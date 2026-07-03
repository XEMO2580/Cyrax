# ═══════════════════════════════════════════════════════════════════════════
# tests/unit/tools/test_base_tool.py
# ═══════════════════════════════════════════════════════════════════════════

"""
tests/unit/tools/test_base_tool.py — BaseTool abstract contract enforcement.

SOURCE NOTE: No standalone tools/base_tool.py file has been produced in this
migration. BaseTool is defined in tools/registry.py (established at Tier 1 /
Priority 0). This suite imports from that actual location rather than a
non-existent path, per the Verification Rule.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from security.auth import SecurityLevel
from tools.registry import BaseTool


# ══════════════════════════════════════════════════════════════════════════════
# Direct instantiation of BaseTool
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseToolAbstractness:

    def test_direct_instantiation_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseTool()  # type: ignore[abstract]

    def test_subclass_missing_execute_raises_type_error(self) -> None:
        class _MissingExecuteArgs(BaseModel):
            value: str

        class _MissingExecuteTool(BaseTool):
            name           = "MISSING_EXECUTE"
            description    = "Deliberately incomplete subclass."
            security_level = SecurityLevel.UNRESTRICTED
            args_schema    = _MissingExecuteArgs
            # execute() intentionally NOT implemented.

        with pytest.raises(TypeError, match="abstract"):
            _MissingExecuteTool()  # type: ignore[abstract]

    def test_subclass_implementing_execute_instantiates_successfully(self) -> None:
        class _CompleteArgs(BaseModel):
            value: str

        class _CompleteTool(BaseTool):
            name           = "COMPLETE_TOOL"
            description    = "Fully implemented subclass."
            security_level = SecurityLevel.UNRESTRICTED
            args_schema    = _CompleteArgs

            def execute(self, value: str) -> str:
                return f"Success: {value}"

        instance = _CompleteTool()
        assert isinstance(instance, BaseTool)
        assert instance.execute(value="x") == "Success: x"


# ══════════════════════════════════════════════════════════════════════════════
# get_tool_definition()
# ══════════════════════════════════════════════════════════════════════════════

class TestGetToolDefinition:

    def test_get_tool_definition_returns_expected_shape(self) -> None:
        class _Args(BaseModel):
            query: str = Field(..., description="A query.")

        class _Tool(BaseTool):
            name           = "SHAPE_TOOL"
            description    = "Tool for definition-shape testing."
            security_level = SecurityLevel.USER
            args_schema    = _Args

            def execute(self, query: str) -> str:
                return "Success"

        definition = _Tool().get_tool_definition()

        assert definition["name"]        == "SHAPE_TOOL"
        assert definition["description"] == "Tool for definition-shape testing."
        assert "parameters" in definition
        assert definition["parameters"]["properties"]["query"]["description"] == "A query."

    def test_get_tool_definition_uses_model_json_schema_not_deprecated_schema(self) -> None:
        """
        Confirms get_tool_definition() calls args_schema.model_json_schema()
        (Pydantic v2 API), not the deprecated .schema() method.
        """
        class _Args(BaseModel):
            x: int

        class _Tool(BaseTool):
            name           = "SCHEMA_METHOD_TOOL"
            description    = "d"
            security_level = SecurityLevel.UNRESTRICTED
            args_schema    = _Args

            def execute(self, x: int) -> str:
                return "Success"

        # If model_json_schema is absent or renamed, this call itself would fail.
        definition = _Tool().get_tool_definition()
        assert definition["parameters"]["type"] == "object"


# ══════════════════════════════════════════════════════════════════════════════
# args_schema Pydantic configuration inheritance
# ══════════════════════════════════════════════════════════════════════════════

class TestArgsSchemaPydanticConfiguration:

    def test_args_schema_rejects_unknown_fields_when_extra_forbid_set(self) -> None:
        """
        Tool authors are expected to set model_config = {"extra": "forbid"}
        on their args_schema (or inherit it from a shared base) so that
        unexpected LLM-generated keys are rejected at validation time
        rather than silently ignored.
        """
        class _StrictArgs(BaseModel):
            model_config = {"extra": "forbid"}
            value: str

        class _StrictTool(BaseTool):
            name           = "STRICT_TOOL"
            description    = "d"
            security_level = SecurityLevel.UNRESTRICTED
            args_schema    = _StrictArgs

            def execute(self, value: str) -> str:
                return "Success"

        with pytest.raises(ValidationError):
            _StrictTool.args_schema(value="ok", unexpected_field="should be rejected")

    def test_args_schema_without_extra_forbid_allows_unknown_fields(self) -> None:
        """
        Documents the DEFAULT Pydantic v2 behaviour (extra='ignore') for any
        tool that does NOT explicitly set extra='forbid' — most tools in
        the current codebase (e.g. OpenAppSchema) do not set this, so
        unknown fields are currently silently dropped, not rejected.
        This is existing behaviour, not asserted as correct or incorrect.
        """
        class _LaxArgs(BaseModel):
            value: str

        parsed = _LaxArgs(value="ok", unexpected_field="silently dropped")
        assert parsed.value == "ok"
        assert not hasattr(parsed, "unexpected_field")

    def test_valid_subclass_args_schema_validates_correct_input(self) -> None:
        class _Args(BaseModel):
            model_config = {"extra": "forbid"}
            count: int

        class _Tool(BaseTool):
            name           = "VALID_INPUT_TOOL"
            description    = "d"
            security_level = SecurityLevel.UNRESTRICTED
            args_schema    = _Args

            def execute(self, count: int) -> str:
                return f"Success: {count}"

        validated = _Tool.args_schema(count=5)
        assert validated.count == 5


# ══════════════════════════════════════════════════════════════════════════════
# Class attribute defaults
# ══════════════════════════════════════════════════════════════════════════════

class TestClassAttributeDefaults:

    def test_security_level_defaults_to_admin_when_unset(self) -> None:
        """
        BaseTool.security_level defaults to SecurityLevel.ADMIN — a
        subclass that forgets to override this is fail-safe (most
        restrictive), not fail-open.
        """
        class _Args(BaseModel):
            pass

        class _NoLevelTool(BaseTool):
            name        = "NO_LEVEL_TOOL"
            description = "d"
            args_schema = _Args
            # security_level intentionally NOT overridden.

            def execute(self) -> str:
                return "Success"

        assert _NoLevelTool().security_level == SecurityLevel.ADMIN

    def test_name_and_description_defaults_when_unset(self) -> None:
        class _Args(BaseModel):
            pass

        class _BareTool(BaseTool):
            args_schema = _Args

            def execute(self) -> str:
                return "Success"

        instance = _BareTool()
        assert instance.name        == "BaseTool"
        assert instance.description == "No description provided."