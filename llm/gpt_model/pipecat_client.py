import asyncio
import json
import uuid
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass

import websockets
from websockets.client import WebSocketClientProtocol
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    CancelFrame,
    ErrorFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMMessagesFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService


@dataclass
class GPT2InputParams:
    """Input parameters for model configuration."""
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 0.9
    top_k: int = 40


class GPT2WebSocketLLMService(LLMService):
    """Pipecat LLM service that connects to a Qwen WebSocket server."""

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8763",
        caller_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 0.9,
        top_k: int = 40,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self._ws_url = ws_url
        self._caller_id = caller_id or f"llm_pipecat_{uuid.uuid4().hex[:8]}"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._connecting = False
        self._connection_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._connected and self._websocket is not None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._disconnect()
        await super().cancel(frame)

    async def _connect(self) -> bool:
        async with self._connection_lock:
            if self._connected:
                return True
            
            if self._connecting:
                return False
            
            self._connecting = True
            
            for attempt in range(self._reconnect_attempts):
                try:
                    logger.info(f"Connecting to LLM server at {self._ws_url} (attempt {attempt + 1})")
                    
                    self._websocket = await websockets.connect(
                        self._ws_url,
                        ping_interval=20,
                        ping_timeout=20,
                    )
                    
                    await self._websocket.send(json.dumps({
                        "type": "connect",
                        "caller_id": self._caller_id,
                    }))
                    
                    response = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "connected":
                        self._connected = True
                        self._connecting = False
                        logger.info(f"Connected to LLM server as {self._caller_id}")
                        return True
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Connection timeout (attempt {attempt + 1})")
                except Exception as e:
                    logger.error(f"Connection failed (attempt {attempt + 1}): {e}")
                
                if attempt < self._reconnect_attempts - 1:
                    await asyncio.sleep(self._reconnect_delay)
            
            self._connecting = False
            return False

    async def _disconnect(self):
        async with self._connection_lock:
            self._connected = False
            if self._websocket:
                try:
                    await self._websocket.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                finally:
                    self._websocket = None
                    logger.info("Disconnected from GPT-2 server")

    async def _ensure_connected(self) -> bool:
        if self.connected:
            return True
        return await self._connect()

    def _extract_user_message(self, messages: List[Dict[str, Any]]) -> tuple[str, str]:
        """Extract system prompt and latest user message."""
        system_prompt = ""
        user_message = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user" and content.strip():
                user_message = content.strip()
        
        return system_prompt, user_message

    async def _generate(self, user_message: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
        """Send generation request and yield response tokens."""
        if not await self._ensure_connected():
            logger.error("Cannot generate: not connected to LLM server")
            return
        
        if not user_message:
            logger.warning("Empty user message, skipping generation")
            return
        
        request_id = str(uuid.uuid4())
        
        request = {
            "type": "generate",
            "request_id": request_id,
            "caller_id": self._caller_id,
            "text": user_message,
            "system": system_prompt,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "streaming": True,
        }
        
        try:
            logger.debug(f"Generation started for request {request_id}")
            await self._websocket.send(json.dumps(request))
            
            previous_text = ""
            
            while True:
                try:
                    raw = await asyncio.wait_for(self._websocket.recv(), timeout=60.0)
                    data = json.loads(raw)
                    event_type = data.get("type")
                    
                    if event_type == "started":
                        logger.debug(f"Server acknowledged generation start for {request_id}")
                        
                    elif event_type == "partial":
                        full_text = data.get("text", "")
                        if len(full_text) > len(previous_text):
                            new_text = full_text[len(previous_text):]
                            previous_text = full_text
                            if new_text:
                                yield new_text
                        
                    elif event_type == "completed":
                        final_text = data.get("text", "")
                        if len(final_text) > len(previous_text):
                            new_text = final_text[len(previous_text):]
                            if new_text:
                                yield new_text
                        logger.debug(f"Generation completed for request {request_id}")
                        break
                        
                    elif event_type == "error":
                        logger.error(f"LLM generation error: {data.get('error')}")
                        break
                        
                except asyncio.TimeoutError:
                    logger.error("Generation timeout")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            logger.error("WebSocket connection closed during generation")
            self._connected = False
        except Exception as e:
            logger.error(f"Error during generation: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
            messages = context.get_messages()
            await self._process_messages(messages)
            
        elif isinstance(frame, LLMMessagesFrame):
            await self._process_messages(frame.messages)
            
        else:
            await self.push_frame(frame, direction)

    async def _process_messages(self, messages: List[Dict[str, Any]]):
        if not messages:
            logger.warning("No messages to process")
            return
        
        system_prompt, user_message = self._extract_user_message(messages)
        
        logger.debug(f"Processing - System: {system_prompt[:50]}... User: {user_message[:50] if user_message else 'EMPTY'}...")
        
        if not user_message:
            logger.warning("No user message found, skipping generation")
            return
        
        await self.push_frame(LLMFullResponseStartFrame())
        
        generated_text = ""
        
        try:
            async for token in self._generate(user_message, system_prompt):
                generated_text += token
                await self.push_frame(LLMTextFrame(text=token))
            
            if generated_text:
                logger.debug(f"Generated response: {generated_text[:100]}...")
            else:
                logger.warning("Empty response generated")
                
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))
            
        finally:
            await self.push_frame(LLMFullResponseEndFrame())


__all__ = ["GPT2WebSocketLLMService", "GPT2InputParams"]