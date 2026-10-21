"""aiia-docker — Qwen3.5-2B + stage-4 LoRA, served from a Gradio Space on ZeroGPU.

This is the deployment half of the assignment's last bullet: the fine-tuned model
published on the Hub, served through an online API (a Gradio Space *is* an API —
`gradio_client` and the `/gradio_api/call/...` HTTP endpoints both work).

Three things here are deliberate and are the reason the file is longer than a demo
usually is:

1. **The sampler is the benchmark's sampler.** The model was measured with the Qwen3.5
   card's recommendation for non-thinking mode, text tasks
   (`temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0,
   repetition_penalty=1.0`). Serving it with anything else produces a different model
   than the one that was reported — the measured terseness in particular is partly a
   sampler effect, not only a weight effect (see the model card).

2. **`presence_penalty` is implemented by hand.** `transformers` 5.20 removed it from
   `GenerationConfig`, so it is applied as a `LogitsProcessor` with llama.cpp's
   semantics: subtract the penalty from the logits of tokens already present in the
   sequence. Dropping it silently would have changed the demo's behaviour.

3. **The model is loaded at module level and moved to `cuda` there.** ZeroGPU emulates
   CUDA outside `@spaces.GPU`, so the load costs no GPU quota; the docs explicitly
   discourage loading inside the decorated function. The weights are therefore read
   once, at Space start, and each request only pays for generation.
"""

import os

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

MODEL_ID = os.environ.get("MODEL_ID", "ljubobratovicrelja/qwen3.5-2b-magicoder-lora")

# The Qwen3.5 card's non-thinking / text recommendation, minus presence_penalty, which
# is applied below as a logits processor because transformers no longer accepts it.
SAMPLING = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.0,
)
PRESENCE_PENALTY = 2.0
DEFAULT_MAX_NEW_TOKENS = 320


class PresencePenalty(LogitsProcessor):
    """llama.cpp semantics: penalise every token already present in the sequence."""

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        if self.penalty == 0.0:
            return scores
        for row in range(input_ids.shape[0]):
            scores[row, torch.unique(input_ids[row])] -= self.penalty
        return scores


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
model.eval()


@spaces.GPU(duration=30)
def generate(message: str, history: list, max_new_tokens: int) -> str:
    """One turn of chat. Runs on a ZeroGPU slice; the model itself is already resident."""
    messages = []
    for turn in history or []:
        # gradio may hand us dicts or tuples depending on version
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
        else:
            role, content = ("user", turn[0]) if len(turn) == 2 else (None, None)
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    # Same chat template the served GGUF uses, with thinking off — the mode this stack
    # (and the benchmark) runs in.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=int(max_new_tokens),
            logits_processor=[PresencePenalty(PRESENCE_PENALTY)],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **SAMPLING,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


DESCRIPTION = f"""
**Qwen3.5-2B-Base + the aiia-docker stage-4 LoRA**, merged into one text-only checkpoint and served here
through Gradio's API (`gradio_client`, or the Space's `/gradio_api/call/...` endpoints).

Trained for 2 epochs (374 optimizer steps) on 3 000 records of
`ise-uiuc/Magicoder-OSS-Instruct-75K`. **Honest expectations:** the measured effect of this adapter was a
change of *style* more than of correctness — shorter answers, no more rambling multi-variant replies —
and the benchmark behind it is
[in the repo](https://github.com/ljubobratovicrelja/aiia-docker/blob/main/docs/BENCHMARK_REPORT.md).
Sampling follows the Qwen3.5 card's non-thinking text recommendation, which is the setting the model was
measured with.
"""

demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(32, 1024, value=DEFAULT_MAX_NEW_TOKENS, step=32, label="max new tokens"),
    ],
    title="aiia-docker — Qwen3.5-2B + stage-4 LoRA (merged)",
    description=DESCRIPTION,
    examples=[
        "Write a Python function that returns the sum of the squares of the even numbers in a list.",
        "Implement a fibonacci sequence generator.",
        "Explain in three sentences why the sky is blue.",
    ],
)

if __name__ == "__main__":
    demo.launch()
