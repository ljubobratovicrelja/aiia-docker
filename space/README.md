---
title: aiia-docker — Qwen3.5-2B + stage-4 LoRA
emoji: 🐳
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
models:
  - ljubobratovicrelja/qwen3.5-2b-magicoder-lora
---

# aiia-docker — the fine-tuned model, served through an online API

This Space is the deployment half of the last bullet of the MUIA 43506 *Arquitecturas e
Infraestructuras para IA* assignment: the stage-4 LoRA fine-tune is published on the Hub
(`ljubobratovicrelja/qwen3.5-2b-magicoder-lora`) and served here through Gradio's own API
— `gradio_client`, or the Space's `/gradio_api/call/...` HTTP endpoints. Nothing else is
needed to use it: no local weights, no local GPU.

**What the model is:** `Qwen/Qwen3.5-2B-Base` (pinned revision) with a LoRA SFT adapter
(r=16, alpha=32) trained for 2 epochs / 374 optimizer steps on 3 000 records of
`ise-uiuc/Magicoder-OSS-Instruct-75K`, merged into a standalone text-only checkpoint.

**What it is not:** a demonstrated improvement. The honest measurement is in the project
repo — [docs/BENCHMARK_REPORT.md](https://github.com/ljubobratovicrelja/aiia-docker/blob/main/docs/BENCHMARK_REPORT.md).
150 generations across three model arms found the adapter changed *style* far more than
correctness: answers are ~45 % shorter and stop rambling into multiple variants, but on a
code-review prompt neither the base nor the tuned model found the real bug, one
instruction-following prompt got worse, and held-out perplexity moved slightly the wrong
way. The Space serves the sampler the model was measured with (the Qwen3.5 card's
non-thinking text recommendation, including a hand-rolled `presence_penalty` because
`transformers` 5.20 dropped it) so that what you see here is what was reported.

**Hardware:** a ZeroGPU slice (shared RTX Pro 6000 Blackwell). Free accounts get 5 GPU
minutes per day, and this Space is built to spend them slowly: the weights load at Space
startup — outside the `@spaces.GPU` boundary, where CUDA is emulated and therefore costs
no quota — so only generation is billed.

The rest of the project (the Docker/Compose inference stack this came out of) is at
[github.com/ljubobratovicrelja/aiia-docker](https://github.com/ljubobratovicrelja/aiia-docker).
