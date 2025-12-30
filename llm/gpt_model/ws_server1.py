#!/usr/bin/env python3
"""
Phi-3 WebSocket server using llama-cpp-python for Mac.
"""

import asyncio
import json
import logging
import signal
import time
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import websockets
from websockets.server import WebSocketServerProtocol

from llama_cpp import Llama

# ---------------------------
WS_HOST = "localhost"
WS_PORT = 8763
MODEL_PATH = "./models/Phi-3-mini-4k-instruct-q4.gguf"

SYSTEM_INSTRUCTION = "You are a Mental Health Support VOICE BOT, a friendly, helpful robot."

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 200
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phi3-ws-server")

llm = None
executor = ThreadPoolExecutor(max_workers=4)
_running_generations: Dict[str, Dict[str, Any]] = {}


def load_model():
    global llm
    logger.info(f"Loading model: {MODEL_PATH}")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,          # Context window
        n_threads=8,         # CPU threads (adjust based on your Mac)
        n_gpu_layers=-1,     # Use all layers on GPU (Metal)
        verbose=False,
    )
    
    logger.info("Model loaded successfully")


def format_prompt(user_message: str) -> str:
    """Format prompt for Phi-3 instruct model."""
    return f"""<|system|>
{SYSTEM_INSTRUCTION}<|end|>
<|user|>
{user_message}<|end|>
<|assistant|>
"""


def generate_sync(prompt: str, temperature: float, top_p: float, max_tokens: int, cancel_event: asyncio.Event):
    """Synchronous generation with streaming - runs in thread pool."""
    try:
        stream = llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=["<|end|>", "<|user|>", "<|endoftext|>"],
        )
        
        for output in stream:
            if cancel_event.is_set():
                break
            token = output["choices"][0]["text"]
            yield token
            
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        raise


async def handle_generate_message(ws: WebSocketServerProtocol, msg: Dict[str, Any]) -> None:
    request_id = msg.get("request_id") or f"request-{time.time()}"
    caller_id = msg.get("caller_id", "unknown")
    text = msg.get("text", "")
    temperature = float(msg.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(msg.get("top_p", DEFAULT_TOP_P))
    max_tokens = int(msg.get("max_tokens", DEFAULT_MAX_TOKENS))

    client_key = f"{id(ws)}"
    if client_key not in _running_generations:
        _running_generations[client_key] = {}

    logger.info(f"[{caller_id}] Received generate request {request_id}: {text[:50]}...")

    prompt = format_prompt(text)
    cancel_event = asyncio.Event()

    async def _run_generation():
        final_text = ""
        try:
            await ws.send(json.dumps({"type": "started", "request_id": request_id}))
            logger.debug(f"[{caller_id}] started {request_id}")

            loop = asyncio.get_event_loop()
            
            # Run the synchronous generator in a thread
            gen = generate_sync(prompt, temperature, top_p, max_tokens, cancel_event)
            
            def get_next():
                try:
                    return next(gen)
                except StopIteration:
                    return None
            
            while True:
                if cancel_event.is_set():
                    break
                    
                token = await loop.run_in_executor(executor, get_next)
                
                if token is None:
                    break
                
                final_text += token
                
                # Send partial update
                payload = {
                    "type": "partial",
                    "request_id": request_id,
                    "caller_id": caller_id,
                    "text": final_text
                }
                await ws.send(json.dumps(payload))

            # Send completed
            if not cancel_event.is_set():
                completed_payload = {
                    "type": "completed",
                    "request_id": request_id,
                    "caller_id": caller_id,
                    "text": final_text
                }
                await ws.send(json.dumps(completed_payload))
                logger.info(f"[{caller_id}] completed {request_id}, length: {len(final_text)}")

        except asyncio.CancelledError:
            cancel_event.set()
            logger.info(f"[{caller_id}] Generation cancelled: {request_id}")
            try:
                await ws.send(json.dumps({
                    "type": "error",
                    "request_id": request_id,
                    "caller_id": caller_id,
                    "error": "generation_cancelled"
                }))
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"Error during generation {request_id}: {e}")
            try:
                await ws.send(json.dumps({
                    "type": "error",
                    "request_id": request_id,
                    "caller_id": caller_id,
                    "error": str(e)
                }))
            except Exception:
                pass
        finally:
            _running_generations.get(client_key, {}).pop(request_id, None)

    task = asyncio.create_task(_run_generation())
    _running_generations[client_key][request_id] = {"task": task, "cancel": cancel_event}


async def handler(ws: WebSocketServerProtocol):
    """Per-connection WebSocket handler."""
    client_key = f"{id(ws)}"
    _running_generations[client_key] = {}
    caller_id = "unknown"

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "error": "invalid_json"}))
                continue

            mtype = msg.get("type")
            
            if mtype == "connect":
                caller_id = msg.get("caller_id", caller_id)
                await ws.send(json.dumps({"type": "connected"}))
                logger.info(f"[{caller_id}] connected (ws_id={client_key})")

            elif mtype == "generate":
                await handle_generate_message(ws, msg)

            elif mtype == "cancel":
                req_id = msg.get("request_id")
                if not req_id:
                    await ws.send(json.dumps({"type": "error", "error": "missing_request_id"}))
                    continue
                    
                gen_info = _running_generations.get(client_key, {}).get(req_id)
                if gen_info:
                    gen_info["cancel"].set()
                    task = gen_info["task"]
                    if not task.done():
                        task.cancel()
                    await ws.send(json.dumps({"type": "cancelled", "request_id": req_id}))
                else:
                    await ws.send(json.dumps({"type": "error", "request_id": req_id, "error": "no_active_request"}))

            else:
                await ws.send(json.dumps({"type": "error", "error": f"unknown_message_type:{mtype}"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{caller_id}] connection closed")
    finally:
        # Cancel any running tasks
        for gen_info in _running_generations.get(client_key, {}).values():
            gen_info["cancel"].set()
            if not gen_info["task"].done():
                gen_info["task"].cancel()
        _running_generations.pop(client_key, None)


async def main():
    stop = asyncio.Event()

    async def _stop_signal():
        logger.info("SIGTERM/SIGINT received. Stopping server...")
        stop.set()

    load_model()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(_stop_signal()))
    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(_stop_signal()))

    logger.info(f"Starting Phi-3 WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(handler, WS_HOST, WS_PORT, max_size=None, ping_interval=20, ping_timeout=20):
        await stop.wait()

    logger.info("Server shutting down...")
    executor.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())