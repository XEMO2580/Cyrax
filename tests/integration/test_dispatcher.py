# ═══════════════════════════════════════════════════════════════════════════
# tests/integration/test_dispatcher.py
# ═══════════════════════════════════════════════════════════════════════════

"""
tests/integration/test_dispatcher.py — Dispatcher orchestration contracts.

CyraxContext, BrainRouter, and ToolRegistry are fully mocked. This suite
certifies DISPATCHER routing decisions (fast-path vs LLM-plan vs chat
fallback, auth interception/resume, shutdown), not any concrete tool's
or provider's business logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from orchestrator.dispatcher import Dispatcher
from orchestrator.decision_engine import CognitiveRoutingDecision, ExecutionMode

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_ctx() -> Mock:
    ctx = Mock(name="CyraxContext")
    ctx.session_id = "test_session"

    ctx.memory = Mock(name="MemoryStack")
    ctx.memory.state = Mock(name="SessionStateStore")
    ctx.memory.state.get.return_value = None
    ctx.memory.state.set = Mock()
    ctx.memory.state.clear = Mock()
    ctx.memory.conversation = Mock(name="ConversationStore")
    ctx.memory.conversation.add_interaction = AsyncMock()
    ctx.memory.conversation.get_history.return_value = []

    ctx.security = Mock(name="SecurityGuard")
    ctx.security.authenticate.return_value = True
    ctx.security.is_locked_out.return_value = (False, 0.0)

    ctx.tool_registry = Mock(name="ToolRegistry")
    ctx.tool_registry.get_all_definitions.return_value = [
        {"name": "OPEN_APP", "description": "Opens an app.", "parameters": {}}
    ]
    ctx.tool_registry.execute_tool = AsyncMock(
        return_value={"status": "success", "response": "Done."}
    )

    ctx.brain_router = Mock(name="BrainRouter")
    ctx.brain_router.plan = AsyncMock(return_value=[])
    ctx.brain_router.chat = AsyncMock(return_value="Conversational response.")
    ctx.brain_router.generate = AsyncMock(
        return_value='{"thought": "done", "action": null, "action_args": {}, "final_answer": "Done."}'
    )

    ctx.fallback_policy = Mock(name="FallbackPolicy")
    ctx.fallback_policy.get_fallback.return_value = None

    ctx.decision_engine = Mock(name="DecisionEngine")
    ctx.decision_engine.classify = AsyncMock(
        return_value=CognitiveRoutingDecision(
            intent="GENERAL_CHAT",
            tools_required=False,
            selected_provider="groq",
            execution_mode=ExecutionMode.IMMEDIATE,
            confidence=1.0,
            reason="General chat.",
        )
    )

    ctx.interrupt_controller = Mock(name="InterruptController")
    ctx.interrupt_controller.cancel_all.return_value = []

    return ctx


@pytest.fixture
def dispatcher() -> Dispatcher:
    return Dispatcher()


# ══════════════════════════════════════════════════════════════════════════════
# Fast-path route
# ══════════════════════════════════════════════════════════════════════════════

class TestFastPathRoute:

    async def test_fast_path_match_executes_tool_without_llm_planner(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        from orchestrator.intent_parser.fast_path import ParsedIntent

        mock_parsed = ParsedIntent(
            tool_name="OPEN_APP", confidence=0.9, parameters={"app": "chrome"}
        )

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=mock_parsed,
        ):
            result = await dispatcher.handle(
                user_input="open chrome",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        assert result["status"] == "success"
        mock_ctx.tool_registry.execute_tool.assert_awaited_once()
        called_tool_name = mock_ctx.tool_registry.execute_tool.call_args[0][0]
        assert called_tool_name == "OPEN_APP"
        mock_ctx.brain_router.plan.assert_not_awaited()

    async def test_fast_path_updates_last_opened_app_on_success(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        from orchestrator.intent_parser.fast_path import ParsedIntent

        mock_parsed = ParsedIntent(
            tool_name="OPEN_APP", confidence=0.9, parameters={"app": "notepad"}
        )

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=mock_parsed,
        ):
            await dispatcher.handle(
                user_input="open notepad",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        mock_ctx.memory.state.set.assert_any_call("last_opened_app", "notepad")

    async def test_fast_path_writes_conversation_memory(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        from orchestrator.intent_parser.fast_path import ParsedIntent

        mock_parsed = ParsedIntent(
            tool_name="OPEN_APP", confidence=0.9, parameters={"app": "chrome"}
        )

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=mock_parsed,
        ):
            await dispatcher.handle(
                user_input="open chrome",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        assert mock_ctx.memory.conversation.add_interaction.await_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# LLM plan route — now routes through Decision Engine + Planner (ReAct)
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMPlanRoute:

    async def test_valid_plan_invokes_agent_supervisor(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        """
        The Dispatcher now routes through the Decision Engine and Planner.
        This test verifies that when the decision engine returns
        tools_required=True / execution_mode=INTERACTIVE, the Planner is
        instantiated and invoked correctly.
        """
        mock_ctx.decision_engine.classify = AsyncMock(
            return_value=CognitiveRoutingDecision(
                intent="OS_CONTROL",
                tools_required=True,
                selected_provider="groq",
                execution_mode=ExecutionMode.INTERACTIVE,
                confidence=1.0,
                reason="OS control action detected.",
            )
        )

        mock_planner_instance = Mock()
        mock_planner_instance.run = AsyncMock(
            return_value={"status": "success", "response": "I successfully executed: OPEN_APP."}
        )

        mock_planner_class = Mock(return_value=mock_planner_instance)

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=None,
        ), patch(
            "orchestrator.dispatcher.Planner", mock_planner_class
        ):
            result = await dispatcher.handle(
                user_input="open chrome and search RTX 5090",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        assert result["status"] == "success"
        mock_planner_class.assert_called_once()
        _, kwargs = mock_planner_class.call_args
        assert kwargs["ctx"] is mock_ctx
        mock_planner_instance.run.assert_awaited_once()

    async def test_supervisor_receives_initial_plan_from_brain_router(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        """
        The Planner receives the user_input and history. The initial plan
        comes from brain_router.plan() which is called inside the Dispatcher's
        _llm_pipeline before the Planner is invoked.
        """
        mock_ctx.decision_engine.classify = AsyncMock(
            return_value=CognitiveRoutingDecision(
                intent="OS_CONTROL",
                tools_required=True,
                selected_provider="groq",
                execution_mode=ExecutionMode.INTERACTIVE,
                confidence=1.0,
                reason="OS control action detected.",
            )
        )

        mock_planner_instance = Mock()
        mock_planner_instance.run = AsyncMock(
            return_value={"status": "success", "response": "Done."}
        )

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=None,
        ), patch(
            "orchestrator.dispatcher.Planner",
            return_value=mock_planner_instance,
        ):
            await dispatcher.handle(
                user_input="open brave",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        mock_planner_instance.run.assert_awaited_once()
        _, call_kwargs = mock_planner_instance.run.call_args
        assert call_kwargs["user_input"] == "open brave"


# ══════════════════════════════════════════════════════════════════════════════
# Auth-required and resume flow
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthRequiredAndResumeFlow:

    async def test_fast_path_auth_required_sets_pending_state(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        from orchestrator.intent_parser.fast_path import ParsedIntent

        mock_ctx.tool_registry.execute_tool = AsyncMock(
            return_value={"status": "auth_required", "response": "PIN required."}
        )
        mock_parsed = ParsedIntent(
            tool_name="SYSTEM_POWER", confidence=1.0, parameters={"action": "shutdown"}
        )

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=mock_parsed,
        ):
            result = await dispatcher.handle(
                user_input="restart",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        assert result["status"] == "auth_required"
        mock_ctx.memory.state.set.assert_any_call("pending_auth", True)
        mock_ctx.memory.state.set.assert_any_call("pending_tool", "SYSTEM_POWER")
        mock_ctx.memory.state.set.assert_any_call("pending_args", {"action": "shutdown"})

    async def test_pending_auth_true_routes_to_resume_after_auth(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        mock_ctx.memory.state.get.side_effect = lambda key: {
            "pending_auth": True,
            "pending_tool": "SYSTEM_POWER",
            "pending_args": {"action": "shutdown"},
        }.get(key)

        result = await dispatcher.handle(
            user_input="1234",
            session_id="test_session",
            trace_id="trace123",
            ctx=mock_ctx,
        )

        mock_ctx.security.authenticate.assert_called_once_with("1234")
        assert result["status"] == "success"
        mock_ctx.tool_registry.execute_tool.assert_awaited_once()
        called_tool_name = mock_ctx.tool_registry.execute_tool.call_args[0][0]
        assert called_tool_name == "SYSTEM_POWER"

    async def test_resume_after_auth_clears_pending_state_on_success(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        mock_ctx.memory.state.get.side_effect = lambda key: {
            "pending_auth": True,
            "pending_tool": "SYSTEM_POWER",
            "pending_args": {"action": "shutdown"},
        }.get(key)

        await dispatcher.handle(
            user_input="1234",
            session_id="test_session",
            trace_id="trace123",
            ctx=mock_ctx,
        )

        mock_ctx.memory.state.clear.assert_any_call("pending_auth")
        mock_ctx.memory.state.clear.assert_any_call("pending_tool")
        mock_ctx.memory.state.clear.assert_any_call("pending_args")

    async def test_resume_after_auth_wrong_pin_returns_auth_required_again(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        mock_ctx.memory.state.get.side_effect = lambda key: {
            "pending_auth": True,
            "pending_tool": "SYSTEM_POWER",
            "pending_args": {"action": "shutdown"},
        }.get(key)
        mock_ctx.security.authenticate.return_value = False
        mock_ctx.security.is_locked_out.return_value = (False, 0.0)

        result = await dispatcher.handle(
            user_input="0000",
            session_id="test_session",
            trace_id="trace123",
            ctx=mock_ctx,
        )

        assert result["status"] == "auth_required"
        mock_ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_resume_after_auth_locked_out_clears_pending_and_errors(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        mock_ctx.memory.state.get.side_effect = lambda key: {
            "pending_auth": True,
            "pending_tool": "SYSTEM_POWER",
            "pending_args": {"action": "shutdown"},
        }.get(key)
        mock_ctx.security.authenticate.return_value = False
        mock_ctx.security.is_locked_out.return_value = (True, 120.0)

        result = await dispatcher.handle(
            user_input="0000",
            session_id="test_session",
            trace_id="trace123",
            ctx=mock_ctx,
        )

        assert result["status"] == "error"
        assert "locked" in result["response"].lower()
        mock_ctx.memory.state.clear.assert_any_call("pending_auth")


# ══════════════════════════════════════════════════════════════════════════════
# Shutdown path
# ══════════════════════════════════════════════════════════════════════════════

class TestShutdownPath:

    @pytest.mark.parametrize(
        "phrase", ["shutdown", "exit", "quit", "shutdown cyrax", "SHUTDOWN"]
    )
    async def test_shutdown_phrases_return_shutdown_status(
        self, dispatcher: Dispatcher, mock_ctx: Mock, phrase: str
    ) -> None:
        result = await dispatcher.handle(
            user_input=phrase,
            session_id="test_session",
            trace_id="trace123",
            ctx=mock_ctx,
        )

        assert result["status"] == "shutdown"

    async def test_shutdown_bypasses_fast_path_and_llm(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        await dispatcher.handle(
            user_input="shutdown",
            session_id="test_session",
            trace_id="trace123",
            ctx=mock_ctx,
        )

        mock_ctx.brain_router.plan.assert_not_awaited()
        mock_ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_empty_input_returns_error_not_shutdown(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        result = await dispatcher.handle(
            user_input="   ",
            session_id="test_session",
            trace_id="trace123",
            ctx=mock_ctx,
        )

        assert result["status"] == "error"


# ══════════════════════════════════════════════════════════════════════════════
# Malformed / empty plan → chat fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestEmptyPlanChatFallback:

    async def test_empty_plan_falls_back_to_chat(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        """
        When the decision engine returns tools_required=False and
        execution_mode=IMMEDIATE, the dispatcher should route to
        brain_router.chat() directly (chat fallback).
        """
        mock_ctx.decision_engine.classify = AsyncMock(
            return_value=CognitiveRoutingDecision(
                intent="GENERAL_CHAT",
                tools_required=False,
                selected_provider="groq",
                execution_mode=ExecutionMode.IMMEDIATE,
                confidence=1.0,
                reason="Simple greeting.",
            )
        )
        mock_ctx.brain_router.chat = AsyncMock(return_value="Hello, how can I help?")

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=None,
        ):
            result = await dispatcher.handle(
                user_input="hello cyrax",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        assert result["status"] == "success"
        assert result["response"] == "Hello, how can I help?"
        mock_ctx.brain_router.chat.assert_awaited_once()
        mock_ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_chat_fallback_writes_conversation_memory(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        """
        When the decision engine says tools_required=False, the conversation
        memory should still be written.
        """
        mock_ctx.decision_engine.classify = AsyncMock(
            return_value=CognitiveRoutingDecision(
                intent="GENERAL_CHAT",
                tools_required=False,
                selected_provider="groq",
                execution_mode=ExecutionMode.IMMEDIATE,
                confidence=1.0,
                reason="Simple chat.",
            )
        )
        mock_ctx.brain_router.chat = AsyncMock(return_value="I am CYRAX.")

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=None,
        ):
            await dispatcher.handle(
                user_input="who are you?",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        assert mock_ctx.memory.conversation.add_interaction.await_count == 2

    async def test_chat_fallback_does_not_instantiate_supervisor(
        self, dispatcher: Dispatcher, mock_ctx: Mock
    ) -> None:
        """
        When the decision engine returns tools_required=False, the Planner
        should NOT be instantiated — the request goes directly to chat.
        """
        mock_ctx.decision_engine.classify = AsyncMock(
            return_value=CognitiveRoutingDecision(
                intent="GENERAL_CHAT",
                tools_required=False,
                selected_provider="groq",
                execution_mode=ExecutionMode.IMMEDIATE,
                confidence=1.0,
                reason="Simple greeting.",
            )
        )
        mock_planner_class = Mock()

        with patch(
            "orchestrator.intent_parser.fast_path.IntentRouter.parse",
            return_value=None,
        ), patch(
            "orchestrator.dispatcher.Planner", mock_planner_class
        ):
            await dispatcher.handle(
                user_input="hello",
                session_id="test_session",
                trace_id="trace123",
                ctx=mock_ctx,
            )

        mock_planner_class.assert_not_called()
