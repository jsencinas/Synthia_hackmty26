"""Project paths and runtime settings."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "manifest.csv"
AUDIO_DIR = ROOT / "audio"
ARTIFACT_DIR = ROOT / "models"
TIMING_MODEL_PATH = ARTIFACT_DIR / "timing_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
VOICE_MODEL_PATH = ARTIFACT_DIR / "voice_model.joblib"
STACKER_PATH = ARTIFACT_DIR / "stacker.joblib"
CALIBRATION_PATH = ARTIFACT_DIR / "calibration.json"

REQUEST_BUDGET_S = float(os.getenv("REQUEST_BUDGET_S", "20"))
STT_TIMEOUT_S = float(os.getenv("STT_TIMEOUT_S", "12"))
# STT is consulted only when the fused acoustic decision is this uncertain
# (|p - 0.5| below the band).
STT_UNCERTAINTY_BAND = float(os.getenv("STT_UNCERTAINTY_BAND", "0.15"))
# Set to "0" to disable the STT stage entirely even if a key is configured.
STT_ENABLED = os.getenv("STT_ENABLED", "1") not in {"0", "false", "False"}

ELEVENLABS_MODEL_ID = "scribe_v2"
ELEVENLABS_LANGUAGE = "spa"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TRANSCRIPT_CACHE_DIR = ROOT / "outputs" / "transcripts"
ELEVENLABS_KEYTERMS = [
    "repita",
    "confirme",
    "mande",
    "bueno",
    "este",
    "o sea",
]
