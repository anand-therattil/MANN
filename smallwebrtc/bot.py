import os
import sys

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

SYSTEM_INSTRUCTION = f"""You are a Mental Health Support VOICE BOT , a friendly, helpful robot."""


async def run_bot(webrtc_connection):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )
    
    stt = VoskWebSocketSTT(url="ws://0.0.0.0:8761",emit_interim=False,)
                  
    
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
    
    # Create context aggregator pair manually
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
            pipecat_transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            pipecat_transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)