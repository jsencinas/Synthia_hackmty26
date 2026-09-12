from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TURNS_DIR = PROJECT_ROOT / "turns"
AUDIO_DIR = PROJECT_ROOT / "audio"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

MANIFEST_PATH = PROJECT_ROOT / "manifest.csv"
FEATURES_PATH = OUTPUTS_DIR / "features.csv"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"

MODEL_PATH = MODELS_DIR / "call_dna_xgboost.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
HUMAN_STATS_PATH = MODELS_DIR / "human_reference_stats.json"

API_VERSION = "v1"
MODEL_VERSION = "0.1.0"
RESPONSE_VERSION = "1.0"

CALLER_CHANNEL = 0
AGENT_CHANNEL = 1

RANDOM_STATE = 42
