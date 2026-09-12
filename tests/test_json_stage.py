import turns
from src.json_stage import run


EXPECTED_KEYS = {
    "anon_id",
    "numero_turnos_caller",
    "duracion_promedio_caller",
    "pausa_promedio_caller",
    "desviacion_estandar_latencia",
    "interrupciones_agente",
    "duracion_total_interrupciones",
    "duracion_promedio_interrupcion",
}


def test_mini_turns_feature_keys(mini_turns_path):
    metricas = turns.analizar_llamada(mini_turns_path)
    assert EXPECTED_KEYS.issubset(metricas.keys())
    assert metricas["numero_turnos_caller"] == 2


def test_missing_model_returns_error(monkeypatch, mini_turns_path):
    monkeypatch.setattr("src.json_stage.JSON_MODEL_PATH", "missing_model.joblib")
    result = run(mini_turns_path)
    assert result.error == "json_model_missing"


def test_missing_turns_returns_error():
    result = run("does_not_exist.json")
    assert result.error in {"json_turns_missing", "json_model_missing"}
