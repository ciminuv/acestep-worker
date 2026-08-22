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

def handler(job):
    global _dit, _llm
    if _dit is None:
        _init()

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    p = job["input"]
    out_dir = "/tmp/acestep-out"
    os.makedirs(out_dir, exist_ok=True)

    params = GenerationParams(
        task_type="text2music",
        caption=p.get("caption", ""),
        lyrics=p.get("lyrics") or "[Instrumental]",
        instrumental=bool(p.get("instrumental", False)),
        duration=float(p.get("duration", -1)),
        bpm=p.get("bpm"),
        keyscale=p.get("keyscale", ""),
        timesignature=p.get("timesignature", ""),
        vocal_language=p.get("vocal_language", "en"),
        seed=int(p.get("seed", -1)),
        inference_steps=int(p.get("inference_steps", 8)),
        shift=3.0,  # recommended for turbo models per INFERENCE.md
        thinking=bool(p.get("thinking", False)),  # app prepares inputs upstream; skip CoT by default
    )
    config = GenerationConfig(batch_size=1, audio_format="mp3")
    result = generate_music(_dit, _llm, params, config, save_dir=out_dir)

    if not result.success or not result.audios:
        raise RuntimeError(result.error or "generation failed")

    audio = result.audios[0]
    with open(audio["path"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return {
        "audio_base64": b64,
        "audio_format": "mp3",
        "sample_rate": audio["sample_rate"],
        "seed": audio["params"]["seed"],
        "lm_metadata": result.extra_outputs.get("lm_metadata"),
    }

runpod.serverless.start({"handler": handler})
