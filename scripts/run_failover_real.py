# scripts/run_failover_real.py
import asyncio
import logging
import time
from unittest.mock import AsyncMock

from brain.moe_router import MoERouter
from brain.provider_metrics import ProviderMetricsManager
from core.resource_manager import ResourceManager
from brain.provider_selector import ProviderSelector
from brain.providers.base import ProviderCapabilities, ProviderError
from core.task_queue import TaskQueue
from core.job_store import SQLiteJobStore
from core.task import TaskType
from core.context import CyraxContext
from orchestrator.executor import TaskExecutor
from core.notification_center import NotificationCenter
from core.interrupt_controller import CancellationToken, InterruptController
from core.resource_manager import ResourceManager
from datetime import datetime
from typing import Any
import tempfile
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("failover-test")

# Minimal fake job store to avoid touching user's home dir DB
class InMemoryJobStore:
    def __init__(self):
        self._store = {}

    async def init(self):
        return

    async def close(self):
        return

    async def insert_job(self, task):
        self._store[task.task_id] = task

    async def update_status(self, job_id, status, result=None, error=None):
        t = self._store.get(job_id)
        if t is None:
            return
        t = t.model_copy(update={"status": status, "result": result or t.result, "error": error or t.error, "updated_at": datetime.now()} )
        self._store[job_id] = t

    async def get_job(self, task_id):
        return self._store.get(task_id)

    async def get_jobs(self, limit=20, offset=0, status=None):
        items = list(self._store.values())
        return items[offset: offset + limit], len(items)

    async def get_due_jobs(self):
        # No scheduled jobs in this test
        return []

    async def recover_pending_jobs(self):
        # No recovery scenario in this test
        return []


# Minimal tool registry (Planner won't call tools in this scenario)
class MinimalToolRegistry:
    def count(self):
        return 0

    def get_all_definitions(self):
        return []

    async def execute_tool(self, tool_name, args, ctx=None):
        return {"status": "success", "response": "ok"}


class MinimalMemory:
    class Conv:
        def __init__(self):
            self._history = []
        def get_history(self, conversation_id=None):
            return []
        async def add_interaction(self, role, content, conversation_id=None):
            logger.info(f"[memory] add_interaction {role}: {content}")

    class Profile:
        def __init__(self):
            self._data = {}
        def get(self, k, default=None):
            return self._data.get(k, default)
        def set(self, k, v):
            self._data[k] = v

    def __init__(self):
        self.conversation = MinimalMemory.Conv()
        self.profile = MinimalMemory.Profile()
        self.state = type("S", (), {"set": lambda self, k, v: None})()


class MinimalNotificationCenter:
    async def subscribe(self, listener):
        return
    async def unsubscribe(self, listener):
        return
    async def publish(self, event):
        logger.info(f"[notification] {event}")


class MinimalFallbackPolicy:
    def get_fallback(self, failed_tool, error_status, original_args):
        return None


class MinimalDecisionEngine:
    async def classify(self, user_input: str):
        # Always say tools_required True for this test's intent
        return {"tools_required": True}


class MinimalDispatcher:
    async def handle(self, user_input: str, session_id: str, trace_id: str, ctx=None):
        return {"status": "ok"}


class MinimalSecurity:
    def authenticate(self, raw_pin: str) -> bool:
        return True
    def is_locked_out(self) -> tuple[bool, float]:
        return (False, 0.0)
    def is_session_valid(self) -> bool:
        return True
    def clear_session(self) -> None:
        return


class MinimalLearningRouter:
    async def select_provider(self, intent_family: str, recommended_provider: str, trace_id: str, active_providers: list[str] | None = None):
        # Just return recommended provider
        return recommended_provider, "reason"


# Fake Groq provider that always fails
class FailingGroqProvider:
    provider_name = "groq"
    capabilities = ProviderCapabilities(
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_tool_schemas=True,
        max_output_tokens=8192,
        context_window_tokens=131072,
    )
    # Ensure structured_json flag exists for the router's capability checks
    capabilities.structured_json = True
    capabilities.text = True
    capabilities.streaming_text = False
    capabilities.streaming_structured_json = False
    capabilities.cancellation = False


    async def generate(self, *args, **kwargs):
        logger.info("[GROQ] generate called - will raise ProviderError (simulate json_validate_failed)")
        raise ProviderError("Simulated Groq json_validate_failed", provider="groq", status_code=400, retryable=True)

    async def health_check(self):
        return False


# Fake Gemini provider that returns a machine-readable planner JSON string
class GeminiProviderMock:
    provider_name = "gemini"
    capabilities = ProviderCapabilities(
        supports_json_mode=False,
        supports_system_prompt=True,
        supports_tool_schemas=False,
        max_output_tokens=8192,
        context_window_tokens=1048576,
    )
    capabilities.structured_json = True  # For this integration test, allow gemini as structured fallback
    capabilities.text = True
    capabilities.streaming_text = False
    capabilities.streaming_structured_json = False
    capabilities.cancellation = False


    async def generate(self, *args, **kwargs):
        logger.info("[GEMINI] generate called - returning machine-readable planner JSON (no server-side JSON)")
        # Return a planner-style JSON object that Planner._parse_plan/_parse_decision can consume.
        # For Planner (ReAct), planner returns Thought+action+... but we will return final_answer to avoid tool execution.
        return '{"thought": "Decided", "action": null, "action_args": {}, "final_answer": "Gemini final answer: task completed."}'

    async def health_check(self):
        return True


async def main():
    # Build providers dict with Groq failing and Gemini succeeding
    providers = {
        "groq": FailingGroqProvider(),
        "gemini": GeminiProviderMock(),
    }

    metrics_manager = ProviderMetricsManager()
    resource_manager = ResourceManager()
    provider_selector = ProviderSelector()

    brain_router = MoERouter(providers=providers, metrics_manager=metrics_manager, resource_manager=resource_manager)

    # Build minimal CyraxContext
    job_store = InMemoryJobStore()
    await job_store.init()
    task_queue = TaskQueue(job_store=job_store)

    ctx = CyraxContext(
        session_id="test-session",
        dispatcher=MinimalDispatcher(),
        tool_registry=MinimalToolRegistry(),
        brain_router=brain_router,
        memory=MinimalMemory(),
        security=MinimalSecurity(),
        fallback_policy=MinimalFallbackPolicy(),
        decision_engine=MinimalDecisionEngine(),
        task_queue=task_queue,
        notification_center=MinimalNotificationCenter(),
        interrupt_controller=InterruptController(),
        job_store=job_store,
        resource_manager=resource_manager,
        learning_router=MinimalLearningRouter(),
        job_scheduler=None,
    )

    executor = TaskExecutor(ctx)
    await executor.start()

    # Submit a background task that requires tools (so Planner runs) and requests provider "groq"
    task = await task_queue.submit(
        user_input="Perform a short task that can be answered by the planner",
        task_type=TaskType.BACKGROUND,
        tools_required=True,
        provider_name="groq",
    )

    logger.info(f"Submitted task {task.task_id} requesting provider {task.provider_name}")

    # Wait for task to complete with timeout
    deadline = time.time() + 15
    result_task = None
    while time.time() < deadline:
        t = await task_queue.get_task(task.task_id)
        if t is not None and t.status.value == "completed":
            result_task = t
            break
        if t is not None and t.status.value in ("failed", "cancelled"):
            result_task = t
            break
        await asyncio.sleep(0.5)

    if result_task is None:
        logger.error("Task did not complete within timeout")
    else:
        logger.info(f"Task ended: id={result_task.task_id} status={result_task.status} result={result_task.result} error={result_task.error}")

    await executor.stop()

if __name__ == "__main__":
    asyncio.run(main())
