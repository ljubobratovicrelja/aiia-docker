"""Stage 1 GUI — the only published entry point.

Gradio chat UI that forwards each message to the inference container's
OpenAI-compatible endpoint. Pure stdlib for the HTTP call, so the only
dependency is gradio itself.
"""

import json
import os
import urllib.request

import gradio as gr

API_URL = os.getenv("API_URL", "http://inference:8080/v1")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "aiia-local")


def chat(message, history):
    messages = [{"role": "user", "content": message}]
    payload = json.dumps(
        {
            "model": MODEL_ALIAS,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            # Qwen3.5 is a thinking model: left on, it spends its whole budget
            # on reasoning_content and never fills `content`.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{API_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"]


demo = gr.ChatInterface(
    fn=chat,
    title="aiia-docker — local inference",
    description=f"Served by llama-server at {API_URL}",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
