# """
# app/cli_main.py — CYRAX 3.0 CLI Entry Point

# Responsibilities:
# - Trace ID generation at the ingestion boundary
# - Event bus creation and worker lifecycle management
# - Mode selection via CLI argument
# - Delegates ALL tool registration and config loading to bootstrap.py
# """

# import sys
# import uuid
# import asyncio
# import argparse
# from venv import logger
# # ── Environment must load before any CYRAX module import ──────────────────────
# from dotenv import load_dotenv
# load_dotenv()

# # ── CYRAX 3.0 internal imports (post-env-load) ────────────────────────────────
# from app.bootstrap import bootstrap
# from logging_.event_logger import get_logger
# from config.settings import settings
# from orchestrator.dispatcher import handle_user
# from voice.stt import listen
# from voice.edge_tts import speak, stop_speaking
# from voice.wake_word import wait_for_wake_word


# # ══════════════════════════════════════════════════════════════════════════════
# # INTERNAL HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def _make_event(event_type: str, payload: str) -> dict:
#     """
#     Stamps every inbound event with a unique Trace ID at the ingestion boundary.
#     All downstream log lines must propagate this ID.
#     """
#     return {
#         "type": event_type,
#         "payload": payload,
#         "trace_id": uuid.uuid4().hex[:12],
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# # INPUT PRODUCERS
# # ══════════════════════════════════════════════════════════════════════════════

# async def text_worker(event_bus: asyncio.Queue) -> None:
#     """Reads keyboard input in a thread, publishes typed events to the event bus."""
#     loop = asyncio.get_running_loop()
#     logger.info("Text worker started.")

#     while True:
#         # Drain any pending events before accepting new input —
#         # prevents queuing a second command while the brain is mid-execution.
#         await event_bus.join()
#         await asyncio.sleep(0.2)

#         user_text: str = await loop.run_in_executor(None, input, "You: ")
#         user_text = user_text.strip()

#         if not user_text:
#             continue

#         logger.log_user_input(user_text)

#         if user_text.lower() in {"shutdown", "exit", "quit", "shutdown cyrax"}:
#             await event_bus.put(_make_event("system", "shutdown"))
#             break

#         await event_bus.put(_make_event("intent", user_text))


# async def voice_worker(event_bus: asyncio.Queue) -> None:
#     """
#     Two-state voice loop: sleeping (waiting for wake word) → awake (listening).
#     Blocking hardware calls are offloaded to a thread pool via run_in_executor.
#     """
#     loop = asyncio.get_running_loop()
#     logger.info("Voice worker started.")
#     is_awake: bool = False

#     while True:
#         if not is_awake:
#             woken_up: bool = await loop.run_in_executor(None, wait_for_wake_word)
#             if woken_up:
#                 is_awake = True
#                 stop_speaking()
#                 print("🎤 CYRAX is awake and listening...")
#                 await asyncio.sleep(0.5)
#             continue

#         result: dict = await loop.run_in_executor(None, listen)

#         if result["status"] == "success":
#             text = result["text"].strip(" .!?")

#             if not text:
#                 # Recogniser returned empty string — stay awake, try again
#                 continue

#             print(f"\n[MIC] Heard: '{text}'")
#             logger.log_user_input(text)

#             if any(x in text.lower() for x in {"go to sleep", "sleep", "stop listening"}):
#                 logger.info("Voice worker: sleep command received.")
#                 is_awake = False
#                 continue

#             if any(x in text.lower() for x in {"shutdown cyrax", "exit cyrax"}):
#                 await event_bus.put(_make_event("system", "shutdown"))
#                 break

#             await event_bus.put(_make_event("intent", text))
#             # Wait for brain to finish processing before accepting next command.
#             await event_bus.join()
#             await asyncio.sleep(0.5)

#         elif result["status"] == "silence":
#             # Intentional silence — go back to sleep, no log noise
#             logger.debug("Voice worker: silence detected, returning to sleep.")
#             is_awake = False

#         elif result["status"] == "unrecognized":
#             # Speech detected but not intelligible — stay awake for retry
#             logger.warning("Voice worker: audio unrecognized, staying awake.")
#             print("🔇 Didn't catch that. Still listening...")

#         else:
#             # Hardware or network error
#             logger.log_error(
#                 "STT_ERROR",
#                 result.get("message", "Unknown STT error"),
#                 context={"status": result["status"]},
#             )
#             print("🔇 Microphone error. Going back to sleep.")
#             is_awake = False


# # ══════════════════════════════════════════════════════════════════════════════
# # CENTRAL BRAIN (Event Consumer)
# # ══════════════════════════════════════════════════════════════════════════════

# async def router_brain(
#     event_bus: asyncio.Queue,
#     mode: str,
#     session_id: str,
# ) -> None:
#     """
#     Consumes events from the bus and dispatches them to the orchestrator.
#     Enforces a per-turn timeout so a stalled LLM call cannot freeze the input
#     pipeline indefinitely.
#     """
#     logger.info(f"Router brain online. Session: {session_id} | Mode: {mode.upper()}")

#     while True:
#         event = await event_bus.get()
#         trace_id: str = event.get("trace_id", "no-trace")

#         try:
#             if event["type"] == "system" and event.get("payload") == "shutdown":
#                 print("\n[CYRAX]: Shutting down. Goodbye. 🛑\n")
#                 if mode == "voice":
#                     await speak("Shutting down. Goodbye.")
#                 break

#             if event["type"] == "intent":
#                 user_text: str = event["payload"]

#                 try:
#                     result = await asyncio.wait_for(
#                         handle_user(
#                             user_input=user_text,
#                             session_id=session_id,
#                             trace_id=trace_id,
#                         ),
#                         timeout=settings.BRAIN_TIMEOUT_SECONDS,
#                     )
#                 except asyncio.TimeoutError:
#                     logger.log_error(
#                         "BRAIN_TIMEOUT",
#                         f"handle_user timed out after {settings.BRAIN_TIMEOUT_SECONDS}s",
#                         context={"trace_id": trace_id, "input": user_text[:80]},
#                     )
#                     result = {"response": "Request timed out. Please try again."}

#                 response_text: str = result.get("response", "Action completed.")
#                 print(f"\n[CYRAX]:\n{response_text}\n")

#                 if mode == "voice":
#                     await speak(response_text)

#         finally:
#             # task_done() must fire even on exception to unblock event_bus.join()
#             event_bus.task_done()


# # ══════════════════════════════════════════════════════════════════════════════
# # SYSTEM BOOT
# # ══════════════════════════════════════════════════════════════════════════════

# def _parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="CYRAX 3.0 CLI")
#     parser.add_argument(
#         "--mode",
#         choices=["text", "voice"],
#         default="text",
#         help="Input mode (default: text)",
#     )
#     parser.add_argument(
#         "--session",
#         default="device_master_001",
#         help="Session / device ID (default: device_master_001)",
#     )
#     return parser.parse_args()


# async def main() -> None:
#     args = _parse_args()
#     mode: str = args.mode
#     session_id: str = args.session

#     print("\n=== CYRAX OS INITIALIZING ===")

#     # Phase 1: Synchronous boot — config validation + tool registration.
#     # Must complete before the event loop starts accepting work.
#     bootstrap()

#     logger.info(f"System boot sequence initiated in {mode.upper()} mode.")

#     # Phase 2: Async boot — event bus + worker tasks.
#     event_bus: asyncio.Queue = asyncio.Queue()

#     sensor_task = asyncio.create_task(
#         text_worker(event_bus) if mode == "text" else voice_worker(event_bus),
#         name=f"sensor-{mode}",
#     )
#     brain_task = asyncio.create_task(
#         router_brain(event_bus, mode, session_id),
#         name="router-brain",
#     )

#     print(f"\n=== CYRAX IS ONLINE | Mode: {mode.upper()} ===")
#     if mode == "text":
#         print("Type your commands. Type 'exit' to quit.")
#     else:
#         print("Say 'Hey Jarvis' to wake me. Say 'shutdown cyrax' to exit.")
#         await speak("Cyrax systems online and ready.")

#     # return_exceptions=True ensures both tasks are awaited even if one raises.
#     # Without it, an exception in brain_task leaves sensor_task as an orphan.
#     results = await asyncio.gather(sensor_task, brain_task, return_exceptions=True)

#     for task, result in zip([sensor_task, brain_task], results):
#         if isinstance(result, Exception):
#             logger.log_error(
#                 "TASK_CRASH",
#                 str(result),
#                 context={"task": task.get_name()},
#             )


# if __name__ == "__main__":
#     if sys.platform == "win32":
#         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n[CYRAX] Emergency manual override. Exiting.")


"""
app/cli_main.py — CYRAX 3.0 CLI Entry Point (DI Corrected)
"""

import sys
import uuid
import asyncio
import argparse

from dotenv import load_dotenv
load_dotenv()

from app.bootstrap import bootstrap
from logging_.event_logger import get_logger
from config.settings import settings

# 🛑 CRITICAL FIX: We no longer import handle_user from orchestrator.
# The dispatcher is now securely injected via CyraxContext.

from voice.stt import listen
from voice.edge_tts import speak, stop_speaking
from voice.wake_word import wait_for_wake_word

def _make_event(event_type: str, payload: str) -> dict:
    return {
        "type": event_type,
        "payload": payload,
        "trace_id": uuid.uuid4().hex[:12],
    }

async def text_worker(event_bus: asyncio.Queue) -> None:
    loop = asyncio.get_running_loop()
    logger = get_logger()
    logger.info("Text worker started.")

    while True:
        await event_bus.join()
        await asyncio.sleep(0.2)
        try:
            user_text: str = await loop.run_in_executor(None, input, "You: ")
        except EOFError:
            await event_bus.put(_make_event("system", "shutdown"))
            break

        user_text = user_text.strip()
        if not user_text:
            continue

        if user_text.lower() in {"shutdown", "exit", "quit", "shutdown cyrax"}:
            await event_bus.put(_make_event("system", "shutdown"))
            break

        await event_bus.put(_make_event("intent", user_text))

async def voice_worker(event_bus: asyncio.Queue) -> None:
    loop = asyncio.get_running_loop()
    logger = get_logger()
    logger.info("Voice worker started.")
    is_awake: bool = False

    while True:
        if not is_awake:
            woken_up: bool = await loop.run_in_executor(None, wait_for_wake_word)
            if woken_up:
                is_awake = True
                stop_speaking()
                print("🎤 CYRAX is awake and listening...")
                await asyncio.sleep(0.5)
            continue

        result: dict = await loop.run_in_executor(None, listen)

        if result["status"] == "success":
            text = result["text"].strip(" .!?")
            if not text:
                continue

            print(f"\n[MIC] Heard: '{text}'")
            if any(x in text.lower() for x in {"go to sleep", "sleep", "stop listening"}):
                is_awake = False
                continue

            if any(x in text.lower() for x in {"shutdown cyrax", "exit cyrax"}):
                await event_bus.put(_make_event("system", "shutdown"))
                break

            await event_bus.put(_make_event("intent", text))
            await event_bus.join()
            await asyncio.sleep(0.5)

        elif result["status"] == "silence":
            is_awake = False
        elif result["status"] == "unrecognized":
            print("🔇 Didn't catch that. Still listening...")
        else:
            print("🔇 Microphone error. Going back to sleep.")
            is_awake = False

async def router_brain(
    event_bus: asyncio.Queue,
    mode: str,
    session_id: str,
    ctx  # 🛑 CRITICAL FIX: The context is passed in here
) -> None:
    logger = get_logger()
    logger.info(f"Router brain online. Session: {session_id} | Mode: {mode.upper()}")

    while True:
        event = await event_bus.get()
        trace_id: str = event.get("trace_id", "no-trace")

        try:
            if event["type"] == "system" and event.get("payload") == "shutdown":
                print("\n[CYRAX]: Shutting down. Goodbye. 🛑\n")
                if mode == "voice":
                    await speak("Shutting down. Goodbye.")
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

                if mode == "voice":
                    await speak(response_text)

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

    event_bus: asyncio.Queue = asyncio.Queue()

    sensor_task = asyncio.create_task(
        text_worker(event_bus) if mode == "text" else voice_worker(event_bus),
        name=f"sensor-{mode}",
    )
    brain_task = asyncio.create_task(
        router_brain(event_bus, mode, session_id, ctx), # 🛑 Pass context here
        name="router-brain",
    )

    print(f"\n=== CYRAX IS ONLINE | Mode: {mode.upper()} ===")
    if mode == "text":
        print("Type your commands. Type 'exit' to quit.")
    else:
        print("Say 'Hey Jarvis' to wake me. Say 'shutdown cyrax' to exit.")
        await speak("Cyrax systems online and ready.")

    results = await asyncio.gather(sensor_task, brain_task, return_exceptions=True)

    for task, result in zip([sensor_task, brain_task], results):
        if isinstance(result, Exception):
            logger.error(f"TASK_CRASH in {task.get_name()}: {result}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CYRAX] Emergency manual override. Exiting.")