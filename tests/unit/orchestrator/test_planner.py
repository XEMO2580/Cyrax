"""
tests/unit/orchestrator/test_planner.py — Comprehensive Unit & Hostile Tests for ReAct Planner (V12-M2)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, Field

from brain.providers.base import (
    GenerationCancelledError,
    GenerationMode,
    ProviderCapabilityError,
    ProviderError,
)
from config.settings import settings
from core.interrupt_controller import CancellationToken
from orchestrator.planner import (
    Planner,
    PlannerState,
    ReActDecision,
    Scratchpad,
    _parse_decision_with_repairs,
)

# ══════════════════════════════════════════════════════════════════════════════
# MOCK SCHEMAS & FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Query to search")
    max_results: int = Field(default=5, ge=1, le=20)


class FileWriteArgs(BaseModel):
    path: str = Field(..., description="File path")
    content: str = Field(..., description="Content to write")


def _make_mock_context(tool_schemas: dict[str, type[BaseModel]] | None = None) -> Mock:
    ctx = Mock()
    ctx.settings = settings
    
    if tool_schemas is None:
        tool_schemas = {
            "WEB_SEARCH": WebSearchArgs,
            "WRITE_FILE": FileWriteArgs,
        }

    tool_defs = [
        {
            "name": name,
            "description": f"Mock tool for {name}",
            "parameters": schema.model_json_schema(),
        }
        for name, schema in tool_schemas.items()
    ]

    tool_registry = Mock()
    tool_registry.get_all_definitions.return_value = tool_defs
    tool_registry.get_tool_names.return_value = list(tool_schemas.keys())
    tool_registry.get_tool_schema.side_effect = lambda name: tool_schemas.get(name)
    tool_registry.execute_tool = AsyncMock(
        return_value={"status": "success", "response": "Tool execution succeeded."}
    )

    ctx.tool_registry = tool_registry
    ctx.fallback_policy = Mock()
    ctx.fallback_policy.get_fallback.return_value = None
    ctx.memory = Mock()
    ctx.memory.state = Mock()
    
    ctx.brain_router = Mock()
    ctx.brain_router.generate = AsyncMock()

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSER & NORMALIZATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerParser:

    def test_clean_native_structured_json_no_repairs(self) -> None:
        raw = json.dumps({
            "thought": "I know the answer.",
            "action": None,
            "action_args": {},
            "final_answer": "The answer is 42."
        })
        decision, repairs = _parse_decision_with_repairs(raw)
        assert decision is not None
        assert decision.thought == "I know the answer."
        assert decision.action is None
        assert decision.final_answer == "The answer is 42."
        assert len(repairs) == 0

    def test_markdown_fences_stripped(self) -> None:
        raw = "```json\n" + json.dumps({
            "thought": "Searching...",
            "action": "WEB_SEARCH",
            "action_args": {"query": "weather"},
            "final_answer": None
        }) + "\n```"
        decision, repairs = _parse_decision_with_repairs(raw)
        assert decision is not None
        assert decision.action == "WEB_SEARCH"
        assert decision.action_args == {"query": "weather"}
        assert "MARKDOWN_FENCE_STRIPPED" in repairs

    def test_surrounding_prose_extracted(self) -> None:
        raw = "Here is my reasoning:\n" + json.dumps({
            "thought": "Done",
            "action": None,
            "action_args": {},
            "final_answer": "Complete"
        }) + "\nHope that helps!"
        decision, repairs = _parse_decision_with_repairs(raw)
        assert decision is not None
        assert decision.final_answer == "Complete"
        assert "PROSE_EXTRACTED" in repairs

    def test_key_normalization(self) -> None:
        raw = json.dumps({
            "Thought": "Looking up query",
            "Action": "WEB_SEARCH",
            "actionArgs": {"query": "news"},
            "FinalAnswer": None
        })
        decision, repairs = _parse_decision_with_repairs(raw)
        assert decision is not None
        assert decision.thought == "Looking up query"
        assert decision.action == "WEB_SEARCH"
        assert decision.action_args == {"query": "news"}
        assert "KEY_NORMALIZED" in repairs

    def test_stringified_json_args_unpacked(self) -> None:
        raw = json.dumps({
            "thought": "Unpacking",
            "action": "WEB_SEARCH",
            "action_args": "{\"query\": \"news\", \"max_results\": 3}",
            "final_answer": None
        })
        decision, repairs = _parse_decision_with_repairs(raw)
        assert decision is not None
        assert decision.action_args == {"query": "news", "max_results": 3}
        assert "ARGS_UNPACKED" in repairs

    def test_list_unwrapped(self) -> None:
        raw = json.dumps([{
            "thought": "Inside list",
            "action": None,
            "action_args": {},
            "final_answer": "List response"
        }])
        decision, repairs = _parse_decision_with_repairs(raw)
        assert decision is not None
        assert decision.final_answer == "List response"
        assert "LIST_UNWRAPPED" in repairs

    def test_malformed_json_returns_none(self) -> None:
        raw = "{thought: bad json"
        decision, repairs = _parse_decision_with_repairs(raw)
        assert decision is None

    def test_empty_string_returns_none(self) -> None:
        decision, repairs = _parse_decision_with_repairs("")
        assert decision is None
        decision, repairs = _parse_decision_with_repairs(None)
        assert decision is None

    def test_wrong_root_type_returns_none(self) -> None:
        decision, _ = _parse_decision_with_repairs('"just a string"')
        assert decision is None
        decision, _ = _parse_decision_with_repairs("12345")
        assert decision is None
        decision, _ = _parse_decision_with_repairs('["item1", "item2"]')
        assert decision is None

    def test_mutual_exclusivity_both_set_returns_none(self) -> None:
        raw = json.dumps({
            "thought": "Invalid",
            "action": "WEB_SEARCH",
            "action_args": {"query": "test"},
            "final_answer": "I also answered"
        })
        decision, _ = _parse_decision_with_repairs(raw)
        assert decision is None

    def test_mutual_exclusivity_neither_set_returns_none(self) -> None:
        raw = json.dumps({
            "thought": "Invalid",
            "action": None,
            "action_args": {},
            "final_answer": None
        })
        decision, _ = _parse_decision_with_repairs(raw)
        assert decision is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. PLANNER ORCHESTRATION & RELIABILITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestPlannerOrchestration:

    async def test_valid_final_answer_immediate(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.return_value = json.dumps({
            "thought": "I know everything.",
            "action": None,
            "action_args": {},
            "final_answer": "Paris is the capital of France."
        })

        planner = Planner(ctx=ctx, req_id="req-01")
        result = await planner.run(user_input="Capital of France?", history=[])

        assert result["status"] == "success"
        assert result["response"] == "Paris is the capital of France."
        ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_valid_tool_action_and_progression(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = [
            json.dumps({
                "thought": "I need to search for current weather.",
                "action": "WEB_SEARCH",
                "action_args": {"query": "London weather"},
                "final_answer": None
            }),
            json.dumps({
                "thought": "I have the weather information.",
                "action": None,
                "action_args": {},
                "final_answer": "It is currently 15°C and sunny in London."
            })
        ]

        ctx.tool_registry.execute_tool.return_value = {
            "status": "success",
            "response": "London: 15°C Sunny"
        }

        planner = Planner(ctx=ctx, req_id="req-02")
        result = await planner.run(user_input="Weather in London?", history=[])

        assert result["status"] == "success"
        assert "15°C" in result["response"]
        assert ctx.brain_router.generate.await_count == 2
        ctx.tool_registry.execute_tool.assert_awaited_once_with(
            "WEB_SEARCH", {"query": "London weather"}, ctx=ctx
        )

    async def test_unknown_tool_rejected_before_execution(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = [
            json.dumps({
                "thought": "Using non-existent tool",
                "action": "NON_EXISTENT_TOOL",
                "action_args": {"foo": "bar"},
                "final_answer": None
            }),
            json.dumps({
                "thought": "Tool wasn't available, giving answer",
                "action": None,
                "action_args": {},
                "final_answer": "I cannot use that tool."
            })
        ]

        planner = Planner(ctx=ctx, req_id="req-03")
        result = await planner.run(user_input="Do something weird", history=[])

        assert result["status"] == "success"
        assert result["response"] == "I cannot use that tool."
        ctx.tool_registry.execute_tool.assert_not_awaited()

        # Verify scratchpad observation for unknown tool
        second_call_messages = ctx.brain_router.generate.call_args_list[1].kwargs["messages"]
        user_prompt_step_2 = second_call_messages[-1]["content"]
        assert "Tool 'NON_EXISTENT_TOOL' is not registered." in user_prompt_step_2
        assert "Available tools: WEB_SEARCH, WRITE_FILE" in user_prompt_step_2
        assert "No execution occurred." in user_prompt_step_2
        assert "Choose a valid tool name from the AVAILABLE TOOLS list and retry." in user_prompt_step_2

    async def test_planner_max_tokens_default_and_override(self) -> None:
        # Default test
        ctx = _make_mock_context()
        ctx.brain_router.generate.return_value = json.dumps({
            "thought": "Direct answer",
            "action": None,
            "action_args": {},
            "final_answer": "Done"
        })

        planner = Planner(ctx=ctx, req_id="req-default-tokens")
        await planner.run(user_input="hello", history=[])
        assert ctx.brain_router.generate.call_args.kwargs["max_tokens"] == 2048

        # Custom override test
        with patch.object(settings, "PLANNER_MAX_TOKENS", 1024):
            ctx_override = _make_mock_context()
            ctx_override.brain_router.generate.return_value = json.dumps({
                "thought": "Direct answer",
                "action": None,
                "action_args": {},
                "final_answer": "Done"
            })
            planner_override = Planner(ctx=ctx_override, req_id="req-override-tokens")
            await planner_override.run(user_input="hello", history=[])
            assert ctx_override.brain_router.generate.call_args.kwargs["max_tokens"] == 1024

    async def test_invalid_tool_arguments_rejected_before_execution(self) -> None:
        ctx = _make_mock_context()
        # WEB_SEARCH requires query: str. Passing empty / missing query.
        ctx.brain_router.generate.side_effect = [
            json.dumps({
                "thought": "Searching with invalid args",
                "action": "WEB_SEARCH",
                "action_args": {"wrong_field": 123},
                "final_answer": None
            }),
            json.dumps({
                "thought": "Correcting search args",
                "action": "WEB_SEARCH",
                "action_args": {"query": "valid query"},
                "final_answer": None
            }),
            json.dumps({
                "thought": "Finished",
                "action": None,
                "action_args": {},
                "final_answer": "Found results."
            })
        ]

        ctx.tool_registry.execute_tool.return_value = {
            "status": "success",
            "response": "Results found"
        }

        planner = Planner(ctx=ctx, req_id="req-04")
        result = await planner.run(user_input="Search something", history=[])

        assert result["status"] == "success"
        assert result["response"] == "Found results."
        
        # 1. Verify PLANNER_MAX_TOKENS = 2048 passed to all think calls
        for call_item in ctx.brain_router.generate.call_args_list:
            assert call_item.kwargs["max_tokens"] == 2048

        # 2. Verify invalid arguments never reached tool execution
        assert ctx.tool_registry.execute_tool.await_count == 1
        ctx.tool_registry.execute_tool.assert_awaited_once_with(
            "WEB_SEARCH", {"query": "valid query"}, ctx=ctx
        )

        # 3. Verify actionable validation feedback in scratchpad fed into next step
        second_call_messages = ctx.brain_router.generate.call_args_list[1].kwargs["messages"]
        user_prompt_step_2 = second_call_messages[-1]["content"]
        assert "Step 1:" in user_prompt_step_2
        assert "Action: WEB_SEARCH({'wrong_field': 123})" in user_prompt_step_2
        assert "Tool WEB_SEARCH validation failed:" in user_prompt_step_2
        assert "query: Field required." in user_prompt_step_2
        assert "No execution occurred." in user_prompt_step_2
        assert "Correct the arguments and retry." in user_prompt_step_2

    async def test_malformed_json_triggers_bounded_correction_retry(self, caplog) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = [
            "{thought: not valid json",
            json.dumps({
                "thought": "Corrected JSON",
                "action": None,
                "action_args": {},
                "final_answer": "Recovered from bad JSON"
            })
        ]

        with caplog.at_level("WARNING"):
            planner = Planner(ctx=ctx, req_id="req-05")
            result = await planner.run(user_input="test", history=[])

        assert result["status"] == "success"
        assert result["response"] == "Recovered from bad JSON"
        assert ctx.brain_router.generate.await_count == 2
        # Verify second call was a schema correction
        second_call = ctx.brain_router.generate.call_args_list[1].kwargs
        assert second_call["intent"] == "react_think_correction"

    async def test_schema_retries_exhausted_returns_error_payload(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.return_value = "still invalid json"

        planner = Planner(ctx=ctx, req_id="req-06")
        result = await planner.run(user_input="test", history=[])

        assert result["status"] == "error"
        assert result["error_code"] == "schema_parse_failed"
        assert "Could not parse a valid reasoning step" in result["response"]

    async def test_max_steps_exhaustion_terminates_safely(self) -> None:
        ctx = _make_mock_context()
        # Keep returning action calls indefinitely
        ctx.brain_router.generate.return_value = json.dumps({
            "thought": "Looping action",
            "action": "WEB_SEARCH",
            "action_args": {"query": "loop"},
            "final_answer": None
        })
        ctx.tool_registry.execute_tool.return_value = {
            "status": "success",
            "response": "Search output"
        }

        planner = Planner(ctx=ctx, req_id="req-07")
        result = await planner.run(user_input="test", history=[])

        assert result["status"] == "error"
        assert result["error_code"] == "max_steps_exceeded"
        assert "exceeded maximum reasoning steps" in result["response"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. HOSTILE TESTS (A through K)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestPlannerHostileScenarios:

    async def test_hostile_a_schema_validation_without_retrieving_basetool(self) -> None:
        """
        Planner must validate arguments against get_tool_schema without retrieving BaseTool.
        """
        from tools.registry import ToolRegistry
        assert not hasattr(ToolRegistry, "get_tool")
        
        ctx = _make_mock_context()
        
        ctx.brain_router.generate.side_effect = [
            json.dumps({
                "thought": "Validating schema",
                "action": "WEB_SEARCH",
                "action_args": {"query": "test", "max_results": 3},
                "final_answer": None
            }),
            json.dumps({
                "thought": "Done",
                "action": None,
                "action_args": {},
                "final_answer": "Done"
            })
        ]

        planner = Planner(ctx=ctx, req_id="hostile-a")
        result = await planner.run(user_input="test", history=[])
        assert result["status"] == "success"
        ctx.tool_registry.get_tool_schema.assert_called_with("WEB_SEARCH")

    async def test_hostile_b_unknown_tool_rejected(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = [
            json.dumps({
                "thought": "Bad tool",
                "action": "EVIL_ROOT_SHELL",
                "action_args": {},
                "final_answer": None
            }),
            json.dumps({
                "thought": "Done",
                "action": None,
                "action_args": {},
                "final_answer": "Safe answer"
            })
        ]

        planner = Planner(ctx=ctx, req_id="hostile-b")
        result = await planner.run(user_input="hack", history=[])
        assert result["status"] == "success"
        ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_hostile_c_invalid_args_rejected(self) -> None:
        ctx = _make_mock_context()
        # WRITE_FILE expects path and content (both strings)
        ctx.brain_router.generate.side_effect = [
            json.dumps({
                "thought": "Writing with bad args",
                "action": "WRITE_FILE",
                "action_args": {"invalid_arg": True},
                "final_answer": None
            }),
            json.dumps({
                "thought": "Giving up",
                "action": None,
                "action_args": {},
                "final_answer": "Could not write"
            })
        ]

        planner = Planner(ctx=ctx, req_id="hostile-c")
        result = await planner.run(user_input="write", history=[])
        assert result["status"] == "success"
        ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_hostile_d_valid_args_reach_execution(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = [
            json.dumps({
                "thought": "Writing valid file",
                "action": "WRITE_FILE",
                "action_args": {"path": "test.txt", "content": "hello world"},
                "final_answer": None
            }),
            json.dumps({
                "thought": "Done",
                "action": None,
                "action_args": {},
                "final_answer": "File written."
            })
        ]

        planner = Planner(ctx=ctx, req_id="hostile-d")
        result = await planner.run(user_input="write file", history=[])
        assert result["status"] == "success"
        ctx.tool_registry.execute_tool.assert_awaited_once_with(
            "WRITE_FILE", {"path": "test.txt", "content": "hello world"}, ctx=ctx
        )

    async def test_hostile_e_clean_native_output_produces_no_repair_warning(self, caplog) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.return_value = json.dumps({
            "thought": "Clean answer",
            "action": None,
            "action_args": {},
            "final_answer": "Clean output"
        })

        with caplog.at_level("WARNING"):
            planner = Planner(ctx=ctx, req_id="hostile-e")
            result = await planner.run(user_input="test", history=[])

        assert result["status"] == "success"
        repair_warnings = [r for r in caplog.records if "repair applied" in r.message]
        assert len(repair_warnings) == 0

    async def test_hostile_f_fenced_json_produces_warning_and_parses(self, caplog) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.return_value = "```json\n" + json.dumps({
            "thought": "Fenced",
            "action": None,
            "action_args": {},
            "final_answer": "Fenced output"
        }) + "\n```"

        with caplog.at_level("WARNING"):
            planner = Planner(ctx=ctx, req_id="hostile-f")
            result = await planner.run(user_input="test", history=[])

        assert result["status"] == "success"
        assert result["response"] == "Fenced output"
        repair_warnings = [r for r in caplog.records if "STRUCTURED_JSON repair applied" in r.message]
        assert len(repair_warnings) >= 1
        assert "MARKDOWN_FENCE_STRIPPED" in repair_warnings[0].message

    async def test_hostile_g_prose_surrounded_json_produces_warning_and_parses(self, caplog) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.return_value = "Here is your answer:\n" + json.dumps({
            "thought": "Prose",
            "action": None,
            "action_args": {},
            "final_answer": "Prose answer"
        }) + "\nHope this helps!"

        with caplog.at_level("WARNING"):
            planner = Planner(ctx=ctx, req_id="hostile-g")
            result = await planner.run(user_input="test", history=[])

        assert result["status"] == "success"
        assert result["response"] == "Prose answer"
        repair_warnings = [r for r in caplog.records if "STRUCTURED_JSON repair applied" in r.message]
        assert len(repair_warnings) >= 1
        assert "PROSE_EXTRACTED" in repair_warnings[0].message

    async def test_hostile_h_malformed_json_after_repairs_terminates_safely(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.return_value = "```json\n{broken json\n```"

        planner = Planner(ctx=ctx, req_id="hostile-h")
        result = await planner.run(user_input="test", history=[])

        assert result["status"] == "error"
        assert result["error_code"] == "schema_parse_failed"

    async def test_hostile_i_cancellation_stops_immediately_without_failover_loop(self) -> None:
        ctx = _make_mock_context()
        token = CancellationToken()
        token.cancel("User cancelled operation")

        planner = Planner(ctx=ctx, req_id="hostile-i")
        with pytest.raises(asyncio.CancelledError):
            await planner.run(user_input="test", history=[], cancel_token=token)

        ctx.brain_router.generate.assert_not_awaited()
        ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_hostile_i2_cancellation_during_generation_propagates(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = GenerationCancelledError(partial_chunks_discarded=2)
        token = CancellationToken()

        planner = Planner(ctx=ctx, req_id="hostile-i2")
        with pytest.raises(GenerationCancelledError):
            await planner.run(user_input="test", history=[], cancel_token=token)

    async def test_hostile_j_provider_capability_exhaustion_handled(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = ProviderCapabilityError("No candidates available for STRUCTURED_JSON")

        planner = Planner(ctx=ctx, req_id="hostile-j")
        result = await planner.run(user_input="test", history=[])

        assert result["status"] == "error"
        assert result["error_code"] == "provider_capability_exhaustion"
        assert "No eligible AI provider available" in result["response"]

    async def test_hostile_k_non_retryable_provider_error_halts(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = ProviderError("401 Unauthorized", provider="groq", status_code=401, retryable=False)

        planner = Planner(ctx=ctx, req_id="hostile-k")
        result = await planner.run(user_input="test", history=[])

        assert result["status"] == "error"
        assert result["error_code"] == "provider_error"
        assert "configuration error" in result["response"]

    async def test_hostile_l_timeout_handled_cleanly(self) -> None:
        ctx = _make_mock_context()
        ctx.brain_router.generate.side_effect = asyncio.TimeoutError()

        planner = Planner(ctx=ctx, req_id="hostile-l")
        result = await planner.run(user_input="test", history=[])

        assert result["status"] == "error"
        assert result["error_code"] == "planner_timeout"
