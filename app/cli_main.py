"""
app/cli_main.py — CYRAX 3.0 CLI Entry Point (DI Corrected)
"""

import sys
import uuid
import asyncio
import argparse

from dotenv import load_dotenv
load_dotenv(override=True)

from app.bootstrap import bootstrap
from core.job_scheduler import JobScheduler
from core.notification_center import TaskNotificationEvent
from logging_.event_logger import get_logger
from config.settings import settings
from orchestrator.executor import TaskExecutor

# 🛑 CRITICAL FIX: We no longer import handle_user from orchestrator.
# The dispatcher is now securely injected via CyraxContext.

# Voice text-triggered (:voice) uses Phase 5.0 voice subsystem.
from voice.manager import VoiceManager
from voice.audio_in import AudioCapture
from voice.stt.openai_adapter import OpenAISTTAdapter
from voice.state import VoiceStateError


async def on_task_notification(event: TaskNotificationEvent):
    # \r moves cursor to start of line, \033[K clears the line
    sys.stdout.write('\r\033[K')
    sys.stdout.flush()

    if "completed" in event.status.lower():
        print(f"\n[BACKGROUND NOTIFICATION]\n✓ {event.title}\n{event.payload}\n")
    elif "failed" in event.status.lower():
        print(f"\n[BACKGROUND NOTIFICATION]\n❌ {event.title}\n{event.payload}\n")

    # Reprint the input prompt so the user isn't lost
    sys.stdout.write('You: ')
    sys.stdout.flush()


def _make_event(event_type: str, payload: str) -> dict:
    return {
        "type": event_type,
        "payload": payload,
        "trace_id": uuid.uuid4().hex[:12],
    }

async def text_worker(event_bus: asyncio.Queue, voice_manager: VoiceManager | None) -> None:
    loop = asyncio.get_running_loop()
    logger = get_logger()
    logger.info("Text worker started.")

    while True:
        # Yield briefly so the brain router can print its startup logs 
        # before we show the input prompt.
        await asyncio.sleep(0.2)

        try:
            user_text: str = await loop.run_in_executor(None, input, "You: ")
        except EOFError:
            logger.info("Text worker: stdin closed.")
            await event_bus.put(_make_event("system", "shutdown"))
            break
        except Exception as e:
            logger.error(f"Input error: {e}")
            break

        user_text = user_text.strip()
        if not user_text:
            continue

        # ── INTERCEPT CLI COMMANDS ───────────────────────────────────────────
        if user_text.lower() in {"shutdown", "exit", "quit", "shutdown cyrax"}:
            await event_bus.put(_make_event("system", "shutdown"))
            break

        # ── Phase 5.3: :voice intercept ──────────────────────────────────────
        if user_text == ":voice":
            if voice_manager is None:
                print("[CYRAX]: Voice interface is unavailable.\n")
                continue

            print("[🎤 RECORDING...]")

            try:
                result = await voice_manager.handle_voice_command()
            except VoiceStateError as exc:
                print(f"[CYRAX]: {exc}\n")
                continue
            except Exception as exc:
                print(f"[CYRAX]: Voice command failed: {exc}\n")
                continue

            # Print the transcript so you know what CYRAX heard
            transcribed = result.get("transcript")
            if transcribed:
                print(f"You (voice): {transcribed}")

            response_text = result.get("response", "Action completed.")
            print(f"\n[CYRAX]:\n{response_text}\n")

            if result.get("status") == "shutdown":
                await event_bus.put(_make_event("system", "shutdown"))
                break

            continue
        # ─────────────────────────────────────────────────────────────────────

        # Standard text input routing
        await event_bus.put(_make_event("intent", user_text))
        await event_bus.join()

    logger.info("Text worker exited.")

async def voice_worker(event_bus: asyncio.Queue) -> None:
    """Legacy full-voice worker. Phase 5 uses text_worker with :voice instead."""
    loop = asyncio.get_running_loop()
    logger = get_logger()
    logger.info("Voice worker started.")
    is_awake: bool = False

    while True:
        # NOTE: wait_for_wake_word, stop_speaking, listen are placeholders
        # from legacy code. Do not use --mode=voice until Phase 5.4.
        if not is_awake:
            # woken_up: bool = await loop.run_in_executor(None, wait_for_wake_word)
            woken_up = False 
            if woken_up:
                is_awake = True
                # stop_speaking()
                print("🎤 CYRAX is awake and listening...")
                await asyncio.sleep(0.5)
            await asyncio.sleep(1)
            continue

async def router_brain(
    event_bus: asyncio.Queue,
    mode: str,
    session_id: str,
    ctx
) -> None:
    logger = get_logger()
    logger.info(f"Router brain online. Session: {session_id} | Mode: {mode.upper()}")

    while True:
        event = await event_bus.get()
        trace_id: str = event.get("trace_id", "no-trace")

        try:
            if event["type"] == "system" and event.get("payload") == "shutdown":
                print("\n[CYRAX]: Shutting down. Goodbye. 🛑\n")
                break

            if event["type"] == "intent":
                user_text: str = event["payload"]

                try:
                    # 🛑 CRITICAL FIX: Use ctx.dispatcher.handle instead of the global handle_user
                    result = await asyncio.wait_for(
                        ctx.dispatcher.handle(
                            user_input=user_text,
                            session_id=session_id,
                            trace_id=trace_id,
                            ctx=ctx
                        ),
                        timeout=settings.BRAIN_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[{trace_id}] BRAIN_TIMEOUT")
                    result = {"response": "Request timed out. Please try again."}
                except Exception as e:
                    logger.error(f"[{trace_id}] ERROR: {str(e)}")
                    result = {"response": f"Internal Error: {str(e)}"}

                response_text: str = result.get("response", "Action completed.")
                print(f"\n[CYRAX]:\n{response_text}\n")

        finally:
            event_bus.task_done()

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CYRAX 3.0 CLI")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--session", default="device_master_001")
    return parser.parse_args()

async def main() -> None:
    args = _parse_args()
    mode: str = args.mode
    session_id: str = args.session

    print("\n=== CYRAX OS INITIALIZING ===")

    # 🛑 CRITICAL FIX: Capture the context from bootstrap
    ctx = bootstrap()
    
    logger = get_logger()
    logger.info(f"System boot sequence initiated in {mode.upper()} mode.")

# ── Phase 8.5: Initialise SQLite job store ─────────────────────────────
    await ctx.job_store.init()
    logger.info("SQLite job store initialised.")

    # ── Phase 8.5: Recover crashed jobs ────────────────────────────────────
    recovered_jobs = await ctx.job_store.recover_pending_jobs()
    for job in recovered_jobs:
        await ctx.task_queue.push_recovered_job(job)
    if recovered_jobs:
        logger.info(f"Recovered {len(recovered_jobs)} pending job(s) from crash.")
    else:
        logger.info("No pending jobs to recover.")

    # ── Phase 8.5: Inject and start JobScheduler ───────────────────────────
    # ctx is frozen, so we use object.__setattr__ to inject the scheduler
    # (since JobScheduler needs ctx, creating a circular dependency at
    # construction time).
    object.__setattr__(ctx, 'job_scheduler', JobScheduler(ctx))
    await ctx.job_scheduler.start()
    logger.info("JobScheduler started.")

    # ── Phase 8.2: Background Task Executor ─────────────────────────────────
    executor = TaskExecutor(ctx)
    await executor.start()
    logger.info("TaskExecutor started.")

    # ── Phase 8.3: Subscribe to background task notifications ───────────────
    await ctx.notification_center.subscribe(on_task_notification)
    logger.info("Notification listener subscribed.")

    event_bus: asyncio.Queue = asyncio.Queue()

    voice_manager = None
    if mode == "text":
        # Phase 5.0 voice subsystem (text-triggered only via :voice)
        try:
            audio_capture = AudioCapture()
            stt = OpenAISTTAdapter()
            voice_manager = VoiceManager(
                ctx=ctx,
                session_id=session_id,
                audio_capture=audio_capture,
                stt=stt,
            )
            logger.info("VoiceManager initialised.")
        except Exception as exc:
            logger.warning(f"VoiceManager unavailable: {exc}. ':voice' will be disabled.")
            voice_manager = None

    sensor_task = asyncio.create_task(
        text_worker(event_bus, voice_manager) if mode == "text" else voice_worker(event_bus),
        name=f"sensor-{mode}",
    )
    brain_task = asyncio.create_task(
        router_brain(event_bus, mode, session_id, ctx), # 🛑 Pass context here
        name="router-brain",
    )

    print(f"\n=== CYRAX IS ONLINE | Mode: {mode.upper()} ===")
    if mode == "text":
        print("Type your commands. Type ':voice' to speak. Type 'exit' to quit.")
    else:
        print("Say 'Hey Jarvis' to wake me. Say 'shutdown cyrax' to exit.")

    results = await asyncio.gather(sensor_task, brain_task, return_exceptions=True)

    for task, result in zip([sensor_task, brain_task], results):
        if isinstance(result, Exception):
            logger.error(f"TASK_CRASH in {task.get_name()}: {result}")

# ── Phase 8.5: Stop JobScheduler and close job store ───────────────────
    if ctx.job_scheduler is not None:
        await ctx.job_scheduler.stop()
        logger.info("JobScheduler stopped.")
    await ctx.job_store.close()
    logger.info("SQLite job store closed.")

    # ── Phase 8.3: Unsubscribe notification listener ────────────────────────
    await ctx.notification_center.unsubscribe(on_task_notification)
    logger.info("Notification listener unsubscribed.")

    # ── Phase 8.2: Cleanly stop the background executor ─────────────────────
    await executor.stop()
    logger.info("TaskExecutor stopped.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CYRAX] Emergency manual override. Exiting.")
