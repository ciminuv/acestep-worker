# syntax=docker/dockerfile:1
FROM ghcr.io/ace-step/ace-step-1.5:latest

WORKDIR /app
RUN uv add runpod

RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    uv run --with huggingface_hub \
      hf download ACE-Step/Ace-Step1.5 --local-dir /app/checkpoints \
    && HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
       uv run --with huggingface_hub \
         hf download ACE-Step/acestep-5Hz-lm-0.6B --local-dir /app/checkpoints/acestep-5Hz-lm-0.6B

COPY handler.py /app/handler.py

# Base image sets ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B and ACESTEP_MODE/ACESTEP_INIT_SERVICE
# for its own gradio/api entrypoint (bypassed below). Pin explicitly to what handler.py
# expects and what was actually baked in above, rather than relying on the base image's
# unrelated defaults staying compatible across its releases.
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ENV ACESTEP_LM_BACKEND=pt

ENTRYPOINT []
CMD ["uv", "run", "python", "handler.py"]
