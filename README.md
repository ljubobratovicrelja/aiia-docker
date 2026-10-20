# aiia-docker — Assignment 2 (Docker), MUIA 43506

> Part of the MUIA master's (Universidad de Alicante, 2026-27),
> course **43506 — Arquitecturas e Infraestructuras para IA** (resp. David Ortiz Pérez).


Deployment of a Deep Learning architecture for inference inside Docker containers, with the
graphical interface isolated from the inference in separate Compose services.

**Deadline:** 2026-10-25 00:00 (Moodle, *Submission Practical Assignment Docker* dropbox).
**Weight:** 25% of the practical grade of 43506.

## What the assignment asks for

- one container running a pre-trained DL model for inference, with the inference call wrapped in
  a web server (Flask / FastAPI);
- a second container running *only* a very simple GUI (gradio) that requests the first;
- the two isolated via Docker Compose;
- a report, max 6 pages excluding cover/TOC, including the code and the commands used.

Optional (and the max-grade clauses): train the model for a few epochs, publish it to Hugging
Face Spaces for API inference, NVIDIA Container Toolkit + training/inference on GPU, or any other
improvement to the Dockerfile / compose / run parameters.

The full brief is in [`docs/assignment/`](docs/assignment/) (PDF + extracted markdown).

## Layout

```
docs/            assignment brief (pdf + md), project plan, architecture decisions
inference/       container 1: model + web server wrapping the inference call
gui/             container 2: gradio interface, talks to container 1 over the internal network
training/        the "few epochs" stage (dataset prep + fine-tune), outside the serving path
evidence/        run logs, measurements and figures the report is built from
```

## Run

```bash
docker compose up -d --build
docker compose ps
```

GUI: <http://localhost:7860> — the only published port. The inference API is reachable
only from inside the compose network (`http://inference:8080/v1`).

```bash
./scripts/evidence.sh    # bring up, health-check, send a fixed prompt, collect evidence
docker compose down
```

## Status

Pre-development: skeleton and plan only. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).
Work log and findings: [`JOURNAL.md`](JOURNAL.md).
