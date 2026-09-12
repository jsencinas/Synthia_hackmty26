import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ESCALATION_THRESHOLD_PCT = float(os.getenv("ESCALATION_THRESHOLD_PCT", "75"))
TURNS_DIR = "turns"
AUDIO_DIR = "audio"
JSON_MODEL_PATH = "modelo_detector.joblib"
JSON_METADATA_PATH = "modelo_metadata.json"
VOICE_MODEL_PATH = "modelo_xgboost.json"
ELEVENLABS_MODEL_ID = "scribe_v2"
ELEVENLABS_LANGUAGE = "spa"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TRANSCRIPT_CACHE_DIR = "outputs/transcripts"
ELEVENLABS_KEYTERMS = [
    "repita",
    "confirme",
    "mande",
    "bueno",
    "este",
    "o sea",
]
