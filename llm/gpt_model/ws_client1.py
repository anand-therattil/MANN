#!/usr/bin/env python3
"""Simple test client for the Phi-3 WebSocket server."""

import asyncio
import json
import websockets

async def test():
    uri = "ws://localhost:8763"
    
    async with websockets.connect(uri) as ws:
        # Connect
        await ws.send(json.dumps({"type": "connect", "caller_id": "test-client"}))
        response = await ws.recv()
        print(f"Connected: {response}")
        
        # Generate
        await ws.send(json.dumps({
            "type": "generate",
            "request_id": "test-1",
            "caller_id": "test-client",
            "text": "I've been feeling anxious lately. Can you help?",
            "max_tokens": 150
        }))
        
        # Receive streaming responses
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "partial":
                print(f"\r{data['text']}", end="", flush=True)
            elif data["type"] == "completed":
                print(f"\n\n--- Completed ---\n{data['text']}")
                break
            elif data["type"] == "error":
                print(f"Error: {data}")
                break
            else:
                print(f"Received: {data}")

if __name__ == "__main__":
    asyncio.run(test())