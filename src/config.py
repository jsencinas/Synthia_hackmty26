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
JSON_MODEL_PATH = ARTIFACT_DIR / "timing_model.joblib"
JSON_METADATA_PATH = ARTIFACT_DIR / "metadata.json"
VOICE_MODEL_PATH = ARTIFACT_DIR / "voice_model.json"
STACKER_PATH = ARTIFACT_DIR / "stacker.joblib"
CALIBRATION_PATH = ARTIFACT_DIR / "calibration.json"
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
