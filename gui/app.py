"""Stage 1 GUI — the only published entry point.

Gradio chat UI that forwards the conversation to the inference container's
OpenAI-compatible endpoint. The HTTP call is stdlib, so gradio is the only
dependency.
"""

import json
import os
import urllib.request

import gradio as gr

API_URL = os.getenv("API_URL", "http://inference:8080/v1")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "aiia-local")


def _as_messages(history):
    """Normalise gradio's history into the API's message list.

    Gradio has passed history as [user, assistant] pairs and as message dicts
    across versions; accept both so a gradio bump does not silently drop the
    conversation.
    """
    messages = []
    for turn in history:
        if isinstance(turn, dict):
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
        else:
            messages.append({"role": "user", "content": turn[0]})
            if turn[1]:
                messages.append({"role": "assistant", "content": turn[1]})
    return messages


def chat(message, history):
    messages = _as_messages(history)
    messages.append({"role": "user", "content": message})
    payload = json.dumps(
        {
            "model": MODEL_ALIAS,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            # Qwen3.5 is a thinking model: left on, it spends the whole budget
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
