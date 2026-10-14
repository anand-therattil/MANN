# MANN

Mental-health triage over a voice call. A voice agent talks to the caller, and the goal is for it to judge how urgently that person needs to speak to a professional and show the result on a live dashboard.

This is a prototype built on top of [Bolna Bhai](https://github.com/anand-therattil/Bolna_bhai). It is not a clinical tool and hasn't been validated by clinicians, so don't use it to make real decisions about someone's care. A person in crisis needs a human and emergency services, not this repo.

## How it works

The caller connects over WebRTC. Their speech goes to Vosk for transcription, the text goes to an LLM with a mental-health support prompt, and the reply is spoken back with Supertonic TTS. Pipecat runs the pipeline and each model sits behind its own WebSocket server.

```
caller -> SmallWebRTC -> Vosk (ASR) -> LLM -> Supertonic (TTS) -> caller
                          :8761       :8763      :8764
```

The LLM server in `llm/gpt_model/` runs Phi-3-mini on vLLM. `ws_server1.py` is a llama.cpp version for Macs.

After a call ends, the bot saves the conversation as JSON in `conversations/` and runs a basic keyword check for anxiety, depression, anger and loneliness.

## What works and what doesn't yet

Working: the voice loop above, and the saved conversations.

Not built yet:
- Urgency scoring. The keyword check is a placeholder and nothing rates how urgent a call is.
- The dashboard showing `caller_number | priority` in real time.
- The "semi-therapist" mode for longer emotional conversations.

## Layout

```
asr/vosk/             Vosk ASR server and Pipecat client
llm/gpt_model/        LLM server and client (convert_gguf.py merges a LoRA adapter and converts to GGUF)
tts/supertonic/       Supertonic TTS server and client
smallwebrtc/          WebRTC server and the bot
config/config.yaml    URLs, ports, Vosk model path
```

## Running

Set `asr.vosk.model_path` in `config/config.yaml` to a Vosk model on your machine (`vosk-model-small-en-in-0.4` works), then install the requirements in each folder and start the servers.

```bash
python asr/vosk/ws_server.py
python llm/gpt_model/ws_server.py
python tts/supertonic/server/py/server.py
python smallwebrtc/server.py
```

The ports above are set in `smallwebrtc/bot.py`.
