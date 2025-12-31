# import os
# import sys

# from dotenv import load_dotenv
# from loguru import logger
# from pipecat.audio.vad.silero import SileroVADAnalyzer
# from pipecat.frames.frames import LLMRunFrame
# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.pipeline.runner import PipelineRunner
# from pipecat.pipeline.task import PipelineParams, PipelineTask
# from pipecat.transports.base_transport import TransportParams
# from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
# from pipecat.processors.aggregators.llm_response import (
#     LLMUserContextAggregator,
#     LLMAssistantContextAggregator,
# )
# from pipecat.services.vosk.stt import VoskWebSocketSTT
# from pipecat.services.supertonic.tts import InterruptibleCustomTTSService
# from pipecat.services.gpt2.llm import GPT2WebSocketLLMService


# load_dotenv(override=True)

# SYSTEM_INSTRUCTION = f"""You are a Mental Health Support VOICE BOT , a friendly, helpful robot."""


# async def run_bot(webrtc_connection):
#     pipecat_transport = SmallWebRTCTransport(
#         webrtc_connection=webrtc_connection,
#         params=TransportParams(
#             audio_in_enabled=True,
#             audio_out_enabled=True,
#             vad_analyzer=SileroVADAnalyzer(),
#             audio_out_10ms_chunks=2,
#         ),
#     )
    
#     stt = VoskWebSocketSTT(url="ws://0.0.0.0:8761",emit_interim=False,)
                  
    
#     llm = GPT2WebSocketLLMService(
#         ws_url="ws://localhost:8763",
#         caller_id=12345,
#     )
#     tts = InterruptibleCustomTTSService(
#         url="ws://localhost:8764",
#     )
#     messages = [
#         {
#             "role": "system",
#             "content": SYSTEM_INSTRUCTION,
#         },
#     ]

#     context = OpenAILLMContext(messages)
    
#     # Create context aggregator pair manually
#     class ContextAggregatorPair:
#         def __init__(self, user_agg, assistant_agg):
#             self._user = user_agg
#             self._assistant = assistant_agg
        
#         def user(self):
#             return self._user
        
#         def assistant(self):
#             return self._assistant
    
#     user_aggregator = LLMUserContextAggregator(context)
#     assistant_aggregator = LLMAssistantContextAggregator(context)
#     context_aggregator = ContextAggregatorPair(user_aggregator, assistant_aggregator)


#     pipeline = Pipeline(
#         [
#             pipecat_transport.input(),  # Transport user input
#             stt,  # STT
#             context_aggregator.user(),  # User responses
#             llm,  # LLM
#             tts,  # TTS
#             pipecat_transport.output(),  # Transport bot output
#             context_aggregator.assistant(),  # Assistant spoken responses
#         ]
#     )


#     task = PipelineTask(
#         pipeline,
#         params=PipelineParams(
#             enable_metrics=True,
#             enable_usage_metrics=True,
#         ),
#     )

#     @pipecat_transport.event_handler("on_client_connected")
#     async def on_client_connected(transport, client):
#         logger.info("Pipecat Client connected")
#         # Kick off the conversation.
#         await task.queue_frames([LLMRunFrame()])

#     @pipecat_transport.event_handler("on_client_disconnected")
#     async def on_client_disconnected(transport, client):
#         logger.info("Pipecat Client disconnected")
#         await task.cancel()

#     runner = PipelineRunner(handle_sigint=False)

#     await runner.run(task)


import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
)
from pipecat.services.vosk.stt import VoskWebSocketSTT
from pipecat.services.supertonic.tts import InterruptibleCustomTTSService
from pipecat.services.gpt2.llm import GPT2WebSocketLLMService


load_dotenv(override=True)

SYSTEM_INSTRUCTION = """You are a Mental Health Support VOICE BOT, a friendly, helpful robot."""

# Storage directory for conversations
CONVERSATIONS_DIR = Path("./conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)


async def save_conversation(session_id: str, messages: list, metadata: dict = None):
    """Save conversation to JSON file."""
    conversation = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "metadata": metadata or {},
        "stats": {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m.get("role") == "user"]),
            "assistant_messages": len([m for m in messages if m.get("role") == "assistant"]),
        }
    }
    
    filepath = CONVERSATIONS_DIR / f"{session_id}.json"
    with open(filepath, "w") as f:
        json.dump(conversation, f, indent=2)
    
    logger.info(f"Conversation saved to {filepath}")
    return filepath


async def analyze_conversation(messages: list) -> dict:
    """
    Basic conversation analysis.
    Expand this based on your needs.
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    
    # Combine all user text for analysis
    user_text = " ".join(m.get("content", "") for m in user_messages).lower()
    
    # Simple keyword detection (expand as needed)
    concerns = {
        "anxiety": ["anxious", "worried", "nervous", "panic", "stress", "overwhelmed"],
        "depression": ["sad", "depressed", "hopeless", "tired", "worthless", "empty"],
        "anger": ["angry", "frustrated", "annoyed", "furious", "mad"],
        "loneliness": ["alone", "lonely", "isolated", "no one", "nobody"],
    }
    
    detected = {}
    for concern, keywords in concerns.items():
        count = sum(1 for kw in keywords if kw in user_text)
        if count > 0:
            detected[concern] = count
    
    return {
        "turn_count": len(user_messages),
        "detected_concerns": detected,
        "primary_concern": max(detected, key=detected.get) if detected else None,
        "total_user_words": len(user_text.split()),
    }


async def run_bot(webrtc_connection):
    # Generate unique session ID for this conversation
    session_id = str(uuid.uuid4())
    session_start = datetime.now()
    
    logger.info(f"Starting session: {session_id}")
    
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )
    
    stt = VoskWebSocketSTT(url="ws://0.0.0.0:8761", emit_interim=False)
    
    llm = GPT2WebSocketLLMService(
        ws_url="ws://localhost:8763",
        caller_id=12345,
    )
    
    tts = InterruptibleCustomTTSService(
        url="ws://localhost:8764",
    )
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
    ]

    context = OpenAILLMContext(messages)
    
    class ContextAggregatorPair:
        def __init__(self, user_agg, assistant_agg):
            self._user = user_agg
            self._assistant = assistant_agg
        
        def user(self):
            return self._user
        
        def assistant(self):
            return self._assistant
    
    user_aggregator = LLMUserContextAggregator(context)
    assistant_aggregator = LLMAssistantContextAggregator(context)
    context_aggregator = ContextAggregatorPair(user_aggregator, assistant_aggregator)

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected - Session: {session_id}")
        await task.queue_frames([LLMRunFrame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected - Session: {session_id}")
        
        # === CAPTURE CONVERSATION HERE ===
        try:
            # Get all messages from the context
            conversation_messages = context.get_messages()
            
            # Calculate session duration
            session_duration = (datetime.now() - session_start).total_seconds()
            
            # Analyze the conversation
            analysis = await analyze_conversation(conversation_messages)
            
            # Save with metadata
            await save_conversation(
                session_id=session_id,
                messages=conversation_messages,
                metadata={
                    "duration_seconds": session_duration,
                    "analysis": analysis,
                    "started_at": session_start.isoformat(),
                    "ended_at": datetime.now().isoformat(),
                }
            )
            
            logger.info(f"Session {session_id} analysis: {analysis}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
        
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)