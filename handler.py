import os
import base64
import runpod
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


def _load_dit():
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root="/app",
        config_path=os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),
        device="cuda",
    )
    # initialize_service auto-downloads the main DiT model when missing
    # (InitServiceDownloadsMixin._ensure_models_present); failure comes back
    # as (message, False).
    if not ok:
        raise RuntimeError(f"DiT init failed: {msg}")
    return dit


# Loaded once at import, before the worker reports ready, so a broken
# checkpoint fails the container instead of the first job, and so the state
# FlashBoot snapshots at spin-down already holds the weights.
_dit = _load_dit()

# No LM is loaded: this worker is a literal caption/lyrics passthrough (the
# app writes both upstream), and generate_music skips the LM entirely when
# llm_handler is None. The flags below are rejected rather than silently
# ignored so a caller asking for CoT finds out.
LM_ONLY_FLAGS = ("thinking", "use_cot_caption", "use_cot_language", "use_cot_metas")

# Per-request overridable defaults for GenerationParams. task_type is fixed -
# this worker only ever does text2music (no src_audio upload path exists for
# cover/repaint/extract/lego/complete). Everything else here is a tuning
# default the caller can override per job, so a param tweak is a request-body
# change, not an image rebuild.
DIT_DEFAULTS = {
    "caption": "",
    "lyrics": "[Instrumental]",
    "instrumental": False,
    "duration": -1,
    "bpm": None,
    "keyscale": "",
    "timesignature": "",
    "vocal_language": "en",
    "seed": -1,
    "inference_steps": 8,
    "shift": 3.0,  # recommended for turbo models per INFERENCE.md
    "thinking": False,
    "use_cot_caption": False,
    "use_cot_language": False,
    "use_cot_metas": False,
    "guidance_scale": 7.0,  # no-op under turbo (auto-forced to 1.0) but harmless to pass
}

CONFIG_DEFAULTS = {
    "batch_size": 1,
    # mp3, not flac: the whole file rides back base64-encoded inside the /run
    # result, and a flac of a full-length song is far over RunPod's 10 MB
    # result ceiling (the job then reports COMPLETED with the output dropped).
    "audio_format": "mp3",
}


def handler(job):
    p = job["input"]
    out_dir = "/tmp/acestep-out"
    os.makedirs(out_dir, exist_ok=True)

    requested_lm = [k for k in LM_ONLY_FLAGS if p.get(k)]
    if requested_lm:
        raise ValueError(f"this worker has no LM loaded; unsupported flags: {requested_lm}")

    # task_type is a fixed worker constant, not a per-request override (see
    # DIT_DEFAULTS comment); config addresses the separate GenerationConfig
    # dataclass. Every other key passes straight through to GenerationParams -
    # an unrecognized key raises a loud TypeError from the dataclass
    # constructor, which is fine since the only caller is our own app backend,
    # never raw end-user input.
    overrides = {k: v for k, v in p.items() if k not in ("config", "task_type")}
    dit_kwargs = {**DIT_DEFAULTS, **overrides}
    dit_kwargs["lyrics"] = p.get("lyrics") or "[Instrumental]"

    params = GenerationParams(task_type="text2music", **dit_kwargs)
    config = GenerationConfig(**{**CONFIG_DEFAULTS, **(p.get("config") or {})})
    result = generate_music(_dit, None, params, config, save_dir=out_dir)

    if not result.success or not result.audios:
        raise RuntimeError(result.error or "generation failed")

    audio = result.audios[0]
    with open(audio["path"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return {
        "audio_base64": b64,
        "audio_format": config.audio_format,
        "sample_rate": audio["sample_rate"],
        "seed": audio["params"]["seed"],
    }


runpod.serverless.start({"handler": handler})
