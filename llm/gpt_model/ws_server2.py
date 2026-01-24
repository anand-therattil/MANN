# #!/usr/bin/env python3
# """
# Fine-tuned TinyLlama Mental Health WebSocket server using llama-cpp-python.
# Optimized for Mac M1/M2/M3 with Metal acceleration.
# """

# import asyncio
# import json
# import logging
# import signal
# import time
# from typing import Dict, Any
# from concurrent.futures import ThreadPoolExecutor

# import websockets
# from websockets.server import WebSocketServerProtocol

# from llama_cpp import Llama

# # ---------------------------
# WS_HOST = "localhost"
# WS_PORT = 8763

# # Path to your GGUF model file
# MODEL_PATH = "./models/tinyllama-mental-health-q4_k_m.gguf"

# # TinyLlama chat format
# SYSTEM_INSTRUCTION = "You are a Mental Health Support assistant. You are empathetic, supportive, and non-judgmental. You help people explore their feelings and provide coping strategies."

# DEFAULT_TEMPERATURE = 0.7
# DEFAULT_TOP_P = 0.9
# DEFAULT_TOP_K = 50
# DEFAULT_MAX_TOKENS = 200
# DEFAULT_REPETITION_PENALTY = 1.2
# # ---------------------------

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger("tinyllama-ws-server")

# llm = None
# executor = ThreadPoolExecutor(max_workers=4)
# _running_generations: Dict[str, Dict[str, Any]] = {}


# def load_model():
#     """Load the GGUF model with Metal acceleration for Mac."""
#     global llm
#     logger.info(f"Loading model: {MODEL_PATH}")
    
#     llm = Llama(
#         model_path=MODEL_PATH,
#         n_ctx=2048,          # Context window (TinyLlama supports up to 2048)
#         n_threads=8,         # CPU threads (adjust based on your Mac)
#         n_gpu_layers=-1,     # Use all layers on GPU (Metal) - set to 0 for CPU only
#         verbose=False,
#         chat_format="chatml",  # TinyLlama uses ChatML format
#     )
    
#     logger.info("Model loaded successfully with Metal acceleration")


# def format_prompt(user_message: str) -> str:
#     """
#     Format prompt for TinyLlama chat model.
#     TinyLlama uses a similar format to ChatML.
#     """
#     # Option 1: Simple format (matches training format)
#     return f"<|user|>\n{user_message}</s>\n<|assistant|>\n"
    
#     # Option 2: With system instruction
#     # return f"""<|system|>
# # {SYSTEM_INSTRUCTION}</s>
# # <|user|>
# # {user_message}</s>
# # <|assistant|>
# # """


# def generate_sync(
#     prompt: str, 
#     temperature: float, 
#     top_p: float, 
#     top_k: int,
#     max_tokens: int, 
#     repetition_penalty: float,
#     cancel_event: asyncio.Event
# ):
#     """Synchronous generation with streaming - runs in thread pool."""
#     try:
#         stream = llm.create_completion(
#             prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             repeat_penalty=repetition_penalty,
#             stream=True,
#             stop=["</s>", "<|user|>", "<|end|>", "<|endoftext|>"],
#         )
        
#         for output in stream:
#             if cancel_event.is_set():
#                 break
#             token = output["choices"][0]["text"]
#             yield token
            
#     except Exception as e:
#         logger.exception(f"Generation error: {e}")
#         raise


# def generate_non_streaming(
#     prompt: str, 
#     temperature: float, 
#     top_p: float, 
#     top_k: int,
#     max_tokens: int, 
#     repetition_penalty: float
# ) -> str:
#     """Non-streaming generation for simpler use cases."""
#     try:
#         output = llm.create_completion(
#             prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             repeat_penalty=repetition_penalty,
#             stream=False,
#             stop=["</s>", "<|user|>", "<|end|>", "<|endoftext|>"],
#         )
        
#         return output["choices"][0]["text"]
        
#     except Exception as e:
#         logger.exception(f"Generation error: {e}")
#         raise


# async def handle_generate_message(ws: WebSocketServerProtocol, msg: Dict[str, Any]) -> None:
#     """Handle a generate request from the client."""
#     request_id = msg.get("request_id") or f"request-{time.time()}"
#     caller_id = msg.get("caller_id", "unknown")
#     text = msg.get("text", "")
#     temperature = float(msg.get("temperature", DEFAULT_TEMPERATURE))
#     top_p = float(msg.get("top_p", DEFAULT_TOP_P))
#     top_k = int(msg.get("top_k", DEFAULT_TOP_K))
#     max_tokens = int(msg.get("max_tokens", DEFAULT_MAX_TOKENS))
#     repetition_penalty = float(msg.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
#     streaming = msg.get("streaming", True)

#     client_key = f"{id(ws)}"
#     if client_key not in _running_generations:
#         _running_generations[client_key] = {}

#     logger.info(f"[{caller_id}] Received generate request {request_id}: {text[:50]}...")

#     prompt = format_prompt(text)
#     cancel_event = asyncio.Event()

#     async def _run_generation():
#         final_text = ""
#         try:
#             await ws.send(json.dumps({"type": "started", "request_id": request_id}))
#             logger.debug(f"[{caller_id}] started {request_id}")

#             loop = asyncio.get_event_loop()
            
#             if streaming:
#                 # Streaming generation
#                 gen = generate_sync(
#                     prompt, temperature, top_p, top_k, 
#                     max_tokens, repetition_penalty, cancel_event
#                 )
                
#                 def get_next():
#                     try:
#                         return next(gen)
#                     except StopIteration:
#                         return None
                
#                 while True:
#                     if cancel_event.is_set():
#                         break
                        
#                     token = await loop.run_in_executor(executor, get_next)
                    
#                     if token is None:
#                         break
                    
#                     final_text += token
                    
#                     # Send partial update
#                     payload = {
#                         "type": "partial",
#                         "request_id": request_id,
#                         "caller_id": caller_id,
#                         "text": final_text
#                     }
#                     await ws.send(json.dumps(payload))
#             else:
#                 # Non-streaming generation
#                 final_text = await loop.run_in_executor(
#                     executor,
#                     generate_non_streaming,
#                     prompt, temperature, top_p, top_k,
#                     max_tokens, repetition_penalty
#                 )

#             # Send completed
#             if not cancel_event.is_set():
#                 completed_payload = {
#                     "type": "completed",
#                     "request_id": request_id,
#                     "caller_id": caller_id,
#                     "text": final_text.strip()
#                 }
#                 await ws.send(json.dumps(completed_payload))
#                 logger.info(f"[{caller_id}] completed {request_id}, length: {len(final_text)}")

#         except asyncio.CancelledError:
#             cancel_event.set()
#             logger.info(f"[{caller_id}] Generation cancelled: {request_id}")
#             try:
#                 await ws.send(json.dumps({
#                     "type": "error",
#                     "request_id": request_id,
#                     "caller_id": caller_id,
#                     "error": "generation_cancelled"
#                 }))
#             except Exception:
#                 pass
#         except Exception as e:
#             logger.exception(f"Error during generation {request_id}: {e}")
#             try:
#                 await ws.send(json.dumps({
#                     "type": "error",
#                     "request_id": request_id,
#                     "caller_id": caller_id,
#                     "error": str(e)
#                 }))
#             except Exception:
#                 pass
#         finally:
#             _running_generations.get(client_key, {}).pop(request_id, None)

#     task = asyncio.create_task(_run_generation())
#     _running_generations[client_key][request_id] = {"task": task, "cancel": cancel_event}


# async def handler(ws: WebSocketServerProtocol):
#     """Per-connection WebSocket handler."""
#     client_key = f"{id(ws)}"
#     _running_generations[client_key] = {}
#     caller_id = "unknown"

#     try:
#         async for raw in ws:
#             try:
#                 msg = json.loads(raw)
#             except json.JSONDecodeError:
#                 await ws.send(json.dumps({"type": "error", "error": "invalid_json"}))
#                 continue

#             mtype = msg.get("type")
            
#             if mtype == "connect":
#                 caller_id = msg.get("caller_id", caller_id)
#                 await ws.send(json.dumps({"type": "connected"}))
#                 logger.info(f"[{caller_id}] connected (ws_id={client_key})")

#             elif mtype == "generate":
#                 await handle_generate_message(ws, msg)

#             elif mtype == "cancel":
#                 req_id = msg.get("request_id")
#                 if not req_id:
#                     await ws.send(json.dumps({"type": "error", "error": "missing_request_id"}))
#                     continue
                    
#                 gen_info = _running_generations.get(client_key, {}).get(req_id)
#                 if gen_info:
#                     gen_info["cancel"].set()
#                     task = gen_info["task"]
#                     if not task.done():
#                         task.cancel()
#                     await ws.send(json.dumps({"type": "cancelled", "request_id": req_id}))
#                     logger.info(f"[{caller_id}] cancelled {req_id}")
#                 else:
#                     await ws.send(json.dumps({"type": "error", "request_id": req_id, "error": "no_active_request"}))

#             elif mtype == "ping":
#                 await ws.send(json.dumps({"type": "pong"}))

#             else:
#                 await ws.send(json.dumps({"type": "error", "error": f"unknown_message_type:{mtype}"}))

#     except websockets.exceptions.ConnectionClosed:
#         logger.info(f"[{caller_id}] connection closed")
#     finally:
#         # Cancel any running tasks
#         for gen_info in _running_generations.get(client_key, {}).values():
#             gen_info["cancel"].set()
#             if not gen_info["task"].done():
#                 gen_info["task"].cancel()
#         _running_generations.pop(client_key, None)


# async def main():
#     stop = asyncio.Event()

#     async def _stop_signal():
#         logger.info("SIGTERM/SIGINT received. Stopping server...")
#         stop.set()

#     # Load the model
#     load_model()

#     loop = asyncio.get_running_loop()
#     loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(_stop_signal()))
#     loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(_stop_signal()))

#     logger.info(f"Starting TinyLlama Mental Health WebSocket server on ws://{WS_HOST}:{WS_PORT}")
#     async with websockets.serve(handler, WS_HOST, WS_PORT, max_size=None, ping_interval=20, ping_timeout=20):
#         await stop.wait()

#     logger.info("Server shutting down...")
#     executor.shutdown(wait=False)


# if __name__ == "__main__":
#     asyncio.run(main())


#!/usr/bin/env python3
"""
Qwen2.5-1.5B WebSocket server using llama-cpp-python.
Optimized for Mac M1/M2/M3 with Metal acceleration.
"""

import asyncio
import json
import logging
import signal
import time
import re
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import websockets
from websockets.server import WebSocketServerProtocol

from llama_cpp import Llama

# ---------------------------
WS_HOST = "localhost"
WS_PORT = 8763

# Path to your GGUF model file
MODEL_PATH = "./qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Default system instruction
DEFAULT_SYSTEM = """You are a friendly mental health support assistant.
RESPONSE in ENGLISH.
Be empathetic, supportive, and conversational.
Give short, helpful responses.
Never include role labels like "User:" or "Assistant:" in your response.
Never simulate conversations or include fake dialogue."""

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 10
DEFAULT_MAX_TOKENS = 150
DEFAULT_REPETITION_PENALTY = 1.2
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen-ws-server")

llm = None
executor = ThreadPoolExecutor(max_workers=4)
_running_generations: Dict[str, Dict[str, Any]] = {}


def load_model():
    """Load the GGUF model with Metal acceleration for Mac."""
    global llm
    logger.info(f"Loading model: {MODEL_PATH}")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )
    
    logger.info("Model loaded successfully with Metal acceleration")


def clean_response(text: str) -> str:
    """Clean the model output to remove garbage and special tokens."""
    if not text:
        return ""
    
    result = text
    
    # List of patterns to remove
    bad_patterns = [
        # Special tokens
        r'<\|im_start\|>', r'<\|im_end\|>', r'<\|endoftext\|>',
        r'<\|user\|>', r'<\|assistant\|>', r'<\|system\|>',
        # Role markers (various formats)
        r'\buser\b:', r'\bassistant\b:', r'\bsystem\b:',
        r'\bUser\b:', r'\bAssistant\b:', r'\bSystem\b:',
        r'\bUSER\b:', r'\bASSISTANT\b:', r'\bSYSTEM\b:',
        r'\bHuman\b:', r'\bhuman\b:',
        # Corrupted markers
        r'\bouser\b', r'\bousser\b', r'\boussey\b', r'\bousy\b',
        r'\biya\b,?', r'\biyaiya\b,?',
        # Chinese garbage characters
        r'[赶戤狨]+',
    ]
    
    for pattern in bad_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Remove any remaining special token patterns
    result = re.sub(r'<\|[^>]*\|>', '', result)
    
    # Fix missing spaces (common issue) - add space before capital letters
    # But be careful not to break things like "I'm" or abbreviations
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)
    
    # Fix common concatenations
    result = re.sub(r'(\.)([A-Z])', r'\1 \2', result)  # Period followed by capital
    result = re.sub(r'(\?)([A-Z])', r'\1 \2', result)  # Question mark followed by capital
    result = re.sub(r'(\!)([A-Z])', r'\1 \2', result)  # Exclamation followed by capital
    result = re.sub(r'(\,)([A-Z])', r'\1 \2', result)  # Comma followed by capital
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)
    
    # Clean up multiple newlines
    result = re.sub(r'\n\s*\n', '\n', result)
    
    return result.strip()


def format_prompt(user_message: str, system_instruction: str) -> str:
    """
    Format prompt for Qwen2.5 model using ChatML format.
    """
    prompt = f"""<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    return prompt


def generate_response(
    user_message: str,
    system_instruction: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
) -> str:
    """Generate a complete response (non-streaming)."""
    
    prompt = format_prompt(user_message, system_instruction)
    
    logger.debug(f"Prompt:\n{prompt}")
    
    stop_sequences = [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "\nUser:",
        "\nuser:",
        "\nAssistant:",
        "\nassistant:",
        "\nHuman:",
        "\nhuman:",
        "\nSystem:",
        "\nsystem:",
    ]
    
    try:
        output = llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stop=stop_sequences,
            stream=False,
        )
        
        raw_text = output["choices"][0]["text"]
        logger.debug(f"Raw output: {raw_text}")
        
        # Clean the response
        cleaned = clean_response(raw_text)
        logger.debug(f"Cleaned output: {cleaned}")
        
        return cleaned
        
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        return "I'm here to help. Could you tell me more?"


def generate_streaming(
    user_message: str,
    system_instruction: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    cancel_event: asyncio.Event,
):
    """Generate response with streaming."""
    
    prompt = format_prompt(user_message, system_instruction)
    
    stop_sequences = [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "\nUser:",
        "\nuser:",
        "\nAssistant:",
        "\nassistant:",
        "\nHuman:",
        "\nhuman:",
        "\nSystem:",
        "\nsystem:",
    ]
    
    try:
        stream = llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stop=stop_sequences,
            stream=True,
        )
        
        accumulated = ""
        for output in stream:
            if cancel_event.is_set():
                break
                
            token = output["choices"][0]["text"]
            accumulated += token
            
            # Yield the accumulated cleaned text
            yield clean_response(accumulated)
            
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        yield "I'm here to help. Could you tell me more?"


async def handle_generate_message(ws: WebSocketServerProtocol, msg: Dict[str, Any]) -> None:
    """Handle a generate request from the client."""
    request_id = msg.get("request_id") or f"request-{time.time()}"
    caller_id = msg.get("caller_id", "unknown")
    user_message = msg.get("text", "").strip()
    system_instruction = msg.get("system", DEFAULT_SYSTEM).strip()
    temperature = float(msg.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(msg.get("top_p", DEFAULT_TOP_P))
    top_k = int(msg.get("top_k", DEFAULT_TOP_K))
    max_tokens = int(msg.get("max_tokens", DEFAULT_MAX_TOKENS))
    repetition_penalty = float(msg.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
    streaming = msg.get("streaming", True)

    # Validate input
    if not user_message:
        logger.warning(f"[{caller_id}] Empty user message, skipping")
        await ws.send(json.dumps({
            "type": "error",
            "request_id": request_id,
            "error": "empty_message"
        }))
        return

    client_key = f"{id(ws)}"
    if client_key not in _running_generations:
        _running_generations[client_key] = {}

    logger.info(f"[{caller_id}] Generate request {request_id}: '{user_message[:50]}...'")

    cancel_event = asyncio.Event()

    async def _run_generation():
        try:
            await ws.send(json.dumps({"type": "started", "request_id": request_id}))

            loop = asyncio.get_event_loop()
            
            if streaming:
                # Streaming generation
                gen = generate_streaming(
                    user_message, system_instruction,
                    temperature, top_p, top_k, 
                    max_tokens, repetition_penalty, cancel_event
                )
                
                previous_text = ""
                
                def get_next():
                    try:
                        return next(gen)
                    except StopIteration:
                        return None
                
                while True:
                    if cancel_event.is_set():
                        break
                        
                    current_text = await loop.run_in_executor(executor, get_next)
                    
                    if current_text is None:
                        break
                    
                    # Only send if text changed
                    if current_text != previous_text:
                        previous_text = current_text
                        payload = {
                            "type": "partial",
                            "request_id": request_id,
                            "caller_id": caller_id,
                            "text": current_text
                        }
                        await ws.send(json.dumps(payload))
                
                final_text = previous_text
            else:
                # Non-streaming generation
                final_text = await loop.run_in_executor(
                    executor,
                    generate_response,
                    user_message, system_instruction,
                    temperature, top_p, top_k,
                    max_tokens, repetition_penalty
                )

            # Send completed
            if not cancel_event.is_set():
                # Final validation
                if not final_text or len(final_text) < 2:
                    final_text = "I'm here to listen. How can I help you today?"
                
                completed_payload = {
                    "type": "completed",
                    "request_id": request_id,
                    "caller_id": caller_id,
                    "text": final_text
                }
                await ws.send(json.dumps(completed_payload))
                logger.info(f"[{caller_id}] Completed {request_id}: '{final_text[:50]}...'")

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
                logger.info(f"[{caller_id}] Connected")

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
                    logger.info(f"[{caller_id}] Cancelled {req_id}")
                else:
                    await ws.send(json.dumps({"type": "error", "request_id": req_id, "error": "no_active_request"}))

            elif mtype == "ping":
                await ws.send(json.dumps({"type": "pong"}))

            else:
                await ws.send(json.dumps({"type": "error", "error": f"unknown_message_type:{mtype}"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{caller_id}] Connection closed")
    finally:
        for gen_info in _running_generations.get(client_key, {}).values():
            gen_info["cancel"].set()
            if not gen_info["task"].done():
                gen_info["task"].cancel()
        _running_generations.pop(client_key, None)


async def main():
    stop = asyncio.Event()

    async def _stop_signal():
        logger.info("Stopping server...")
        stop.set()

    load_model()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(_stop_signal()))
    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(_stop_signal()))

    logger.info(f"Starting Qwen2.5 WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(handler, WS_HOST, WS_PORT, max_size=None, ping_interval=20, ping_timeout=20):
        await stop.wait()

    logger.info("Server shut down")
    executor.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())