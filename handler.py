import os
from pathlib import Path
import base64
import runpod

_dit = None
_llm = None

def _init():
    global _dit, _llm
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.model_downloader import ensure_lm_model

    _dit = AceStepHandler()
    msg, ok = _dit.initialize_service(
        project_root="/app",
        config_path=os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),
        device="cuda",
    )
    # initialize_service auto-downloads the main DiT model when missing
    # (InitServiceDownloadsMixin._ensure_models_present); failure comes back
    # as (message, False).
    if not ok:
        raise RuntimeError(f"DiT init failed: {msg}")

    # LLMHandler.initialize does NOT download - it errors when the LM is
    # missing ("5Hz LM model not found"). Fetch it explicitly first.
    checkpoint_dir = "/app/checkpoints"
    lm_name = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
    dl_ok, dl_msg = ensure_lm_model(model_name=lm_name, checkpoints_dir=Path(checkpoint_dir))
    if not dl_ok:
        raise RuntimeError(f"LM download failed: {dl_msg}")

    _llm = LLMHandler()
    llm_msg, llm_ok = _llm.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_name,
        backend=os.getenv("ACESTEP_LM_BACKEND", "pt"),
        device="cuda",
    )
    if not llm_ok:
        raise RuntimeError(f"LM init failed: {llm_msg}")

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
    # Our app writes final caption/lyrics/language upstream (Step 1) - skip
    # ACE-Step's own LM CoT entirely by default. thinking=False alone is NOT
    # sufficient: use_cot_caption/use_cot_language/use_cot_metas default True
    # on GenerationParams and independently trigger the LM step even when
    # thinking=False (see acestep/inference.py's use_lm/need_lm_for_cot
    # logic), silently overwriting caption/vocal_language with the LM's own
    # rewrite. All three must be off for a literal passthrough.
    "thinking": False,
    "use_cot_caption": False,
    "use_cot_language": False,
    "use_cot_metas": False,
    "guidance_scale": 7.0,  # no-op under turbo (auto-forced to 1.0) but harmless to pass
    "lm_temperature": 0.85,
    "lm_top_p": 0.95,
}

CONFIG_DEFAULTS = {
    "batch_size": 1,
    "audio_format": "flac",
}

def handler(job):
    global _dit, _llm
    if _dit is None:
        _init()

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    p = job["input"]
    out_dir = "/tmp/acestep-out"
    os.makedirs(out_dir, exist_ok=True)

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
    result = generate_music(_dit, _llm, params, config, save_dir=out_dir)

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
        "lm_metadata": result.extra_outputs.get("lm_metadata"),
    }

runpod.serverless.start({"handler": handler})
