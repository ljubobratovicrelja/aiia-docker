======================================================================
README.txt — evidence/mint-3090ti-2026-10-24/
======================================================================
machine         : mint (tailscale), relja-linux, x86_64, Linux 5.15.0-198-generic
cpu             : AMD Ryzen 9 7950X3D 16-Core Processor (32 threads)
gpu             : NVIDIA GeForce RTX 3090 Ti 24 GB, driver 580.178.04
engine / compose: Docker 29.9.2 / Compose 5.6.1
collected       : 2026-10-24, 18:55 - 19:25 CEST
repo            : a clean rsync of this repository at 3067837, staged on mint at
                  /var/tmp/aiia-mint-measure (mint's /home is not writable for the
                  Docker CLI, so HOME=/var/tmp/mint-home and
                  DOCKER_CONFIG=/var/tmp/mint-home/.docker)

WHY THIS DIRECTORY EXISTS
  The numbers the course report quotes for the RTX 3090 Ti were measured on this
  host but were never committed; the repository only carried artifacts recorded
  on the aarch64 DGX Spark ('gb10').  Everything here was re-measured on mint on
  2026-10-24.  No figure in this directory was copied from the gb10 artifacts —
  where a gb10 number is mentioned it is named as gb10's and used only for
  comparison.

HOW IT WAS RUN
  The project's own scripts, unmodified, on a clean worktree of this repository:
    ./scripts/image-sizes.sh      ./scripts/bench.sh        ./scripts/bench_gpu.sh
    ./scripts/gpu_evidence.sh     ./scripts/size-experiment.sh
    ./scripts/cache_demo.sh
  Each artifact's own header records the date, arch, engine and compose version
  the script printed; a preamble above that header adds the machine, the CPU/GPU,
  the driver and the exact command, because the scripts' headers only know the
  architecture.  The three artifacts the scripts do not produce (host-env.txt,
  mem-stats.txt, gpu-vram.txt, port-isolation.txt, image-sizes-reconciliation.txt)
  carry the shell pipeline that produced them in the same header.

WHAT IS IN HERE — artifact : the question it answers : the headline number
  host-env.txt
      What machine is this, exactly?  uname, nvidia-smi, docker version, the nvidia
      runtime, cgroup version, git HEAD of the tree measured, disk.
      : RTX 3090 Ti 24 GB (24564 MiB), driver 580.178.04, Engine 29.9.2.

  image-sizes.txt
      Image size and layer breakdown of the two images the stack runs, from the
      project's own script.
      : cpu inference .Size 3 745 056 666 B, gui .Size 642 651 219 B.
        READ WITH image-sizes-reconciliation.txt: on this engine .Size double
        counts compressed layer content.  3.75 GB is not the footprint.

  image-sizes-gpu-and-base.txt
      The same for the GPU target and both upstream llama.cpp base images.
      : gpu inference .Size 9 015 544 582 B; base server 1 211 446 363 B;
        base server-cuda13 6 474 828 317 B.

  image-sizes-reconciliation.txt
      The four numbers an image size can be, measured on all five images, and the
      identity between them.
      : .Size = Σ(history SIZE) + Σ(compressed blobs), i.e. .Size is inflated by
        23-42 % here.  Footprints (file bytes): cpu 2 125 630 063 B,
        gpu 5 365 668 299 B, gui 441 747 285 B.

  bench-cpu.txt
      Fixed-prompt throughput on the CPU path, both routes, 3 runs each.
      : inference median 9.53 tok/s (pooled 9.32), gui median 9.67 (pooled 9.60),
        59 completion tokens per isolated run.

  bench-gpu.txt
      The same request on the CUDA target, 2 executions x 3 runs per route, with
      the CPU-vs-GPU table appended by the project's comparator.
      : inference median 281.51 tok/s (pooled 277.40), 62 completion tokens;
        gui median 258.89 (pooled 262.36).  Executions: 278.97 and 296.74.

  gpu-vram.txt
      Device memory during load, idle, during generation and after — host and inside
      the container — plus llama.cpp's offload decision.
      : 25/25 layers offloaded; CUDA0 model 1211.05 MiB + KV 48.00 + recurrent
        77.06 + compute 56.02 (projected 1392 MiB); /app/llama-server 1696-1698 MiB
        per-process; device-wide 6.4-6.9 GiB loaded, including a desktop using
        ~4.6-4.7 GiB.

  gpu-device-and-offload.txt
      The GPU variant proof (project script): shared model stage, the resolved
      DeviceRequests, nvidia-smi from inside, offload log, docker stats.
      : count: all -> DeviceRequests[0].Count = -1; health from the inherited
        check; inference still unpublished.

  mem-stats.txt
      RSS of both containers under load against the Compose limits that Compose
      actually resolved (.HostConfig).
      : inference limit 4 294 967 296 B / 4e9 NanoCpus, gui 536 870 912 B / 1e9.
        cpu stack: inference 929 MiB -> 1.026 GiB (25.7 % of 4 GiB), gui 201 MiB
        (39.3 % of 512 MiB).  gpu stack: inference 767-807 MiB (19.7 %).

  size-experiment.txt
      One demonstrated size improvement: gui built with and without
      --no-cache-dir, both variants measured, plus a cross-check by file bytes.
      : file bytes 441 747 285 -> 529 711 073 (+88.0 MB, +19.9 %); the extra
        bytes are /root/.cache/pip (88 MB).  The .Size delta (177.4 MB) is the
        inflated figure and is not the cost.

  build-cache.txt
      Layer cache behaviour, all four cases, with assertions.
      : rebuild with no change: every step CACHED.  touch app.py: everything
        CACHED (COPY is keyed on content, not mtime).  real content edit: pip
        layer CACHED, COPY app.py RE-RAN.  runtime ENV change in
        inference/Dockerfile: model layer CACHED (no 1.28 GB re-download).

  port-isolation.txt
      Which container publishes what, on both stacks, from the engine's own view.
      : inference HostConfig.PortBindings {} and NetworkSettings.Ports
        {"8080/tcp": null} on both stacks; gui publishes 7860 on 0.0.0.0 and ::;
        ss shows only the two gradio sockets.

CROSS-CHECK AGAINST gb10 (same quantity, different host)
  cpu inference 2 117 416 629 B -> 2 125 630 063 B  (+0.4 %)
  gpu inference 5 684 548 299 B -> 5 365 668 299 B  (-5.6 %, different CUDA base)
  gui             461 208 892 B -> 441 747 285 B    (-4.2 %)
  --no-cache-dir    84.1 MB      -> 88.0 MB
  The image sizes reproduce; the throughput figures do not match the numbers
  previously quoted for this host (10.0 / 320.7 tok/s): measured here 9.53 and
  281.51 tok/s.  See JOURNAL.md, 2026-10-24.
