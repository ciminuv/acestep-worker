# syntax=docker/dockerfile:1
FROM ghcr.io/ace-step/ace-step-1.5:latest

WORKDIR /app
RUN uv add runpod

# The main bundle ships the DiT, VAE, text encoder and the 1.7B LM; the LM is
# never loaded by handler.py but check_main_model_exists insists on it being
# present, so it stays in the image rather than being re-downloaded at boot.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    uv run --with huggingface_hub \
      hf download ACE-Step/Ace-Step1.5 --local-dir /app/checkpoints

COPY handler.py /app/handler.py

# Base image sets ACESTEP_MODE/ACESTEP_INIT_SERVICE for its own gradio/api
# entrypoint (bypassed below). Pin the config explicitly to what handler.py
# expects rather than relying on the base image's unrelated defaults staying
# compatible across its releases.
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo

ENTRYPOINT []
CMD ["uv", "run", "--no-sync", "python", "handler.py"]
