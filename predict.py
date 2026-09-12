#!/usr/bin/env python3
"""
predict.py
Módulo e interfaz de inferencia para predecir si una llamada es 'human' o 'synthetic'.
Soporta predicción por anon_id, archivo turns/*.json, diccionario de métricas o archivo CSV completo.
"""

import os
import sys
import json
import argparse
import statistics
import joblib
import pandas as pd
import numpy as np
from train_model import agregar_features_ingenieria, COLUMNAS_BASE


MODELO_PATH = "modelo_detector.joblib"
METADATA_PATH = "modelo_metadata.json"


def cargar_modelo_y_metadata():
    """Carga el modelo serializado y sus metadatos de configuración."""
    if not os.path.exists(MODELO_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo ({MODELO_PATH}) o metadatos ({METADATA_PATH}). "
            f"Por favor ejecuta primero: python3 train_model.py"
        )

    modelo = joblib.load(MODELO_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return modelo, metadata


def extraer_metricas_de_turns_json(ruta_json: str) -> dict:
    """
    Extrae las 7 métricas base a partir de un archivo turns/<anon_id>.json.
    Replica la lógica de extracción de segmentos y turnos de conversación.
    """
    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    turns = data.get("turns", [])
    channel_0 = [t for t in turns if t["channel"] == 0]
    channel_1 = [t for t in turns if t["channel"] == 1]

    channel_0.sort(key=lambda x: x["start"])
    channel_1.sort(key=lambda x: x["start"])

    # 1. Número de turnos del caller
    numero_turnos_caller = len(channel_0)

    # 2. Duración promedio de turnos del caller
    duraciones = [t["end"] - t["start"] for t in channel_0]
    duracion_promedio_caller = (
        sum(duraciones) / len(duraciones) if duraciones else 0.0
    )

    # 3. Pausas promedio entre turnos consecutivos del caller
    pausas = []
    for i in range(len(channel_0) - 1):
        pausa = channel_0[i + 1]["start"] - channel_0[i]["end"]
        pausas.append(pausa)
    pausa_promedio_caller = sum(pausas) / len(pausas) if pausas else 0.0

    # 4. Latencias entre el fin del turno del agente y el inicio del caller
    latencias = []
    for caller in channel_0:
        ultimo_agente = None
        for agent in channel_1:
            if agent["end"] <= caller["start"]:
                if ultimo_agente is None or agent["end"] > ultimo_agente:
                    ultimo_agente = agent["end"]
        if ultimo_agente is not None:
            latencias.append(caller["start"] - ultimo_agente)

    # 5. Desviación estándar de latencias
    if len(latencias) > 1:
        desviacion_estandar_latencia = statistics.stdev(latencias)
    else:
        desviacion_estandar_latencia = 0.0

    # 6 y 7. Interrupciones del agente sobre el caller
    interrupciones_agente = 0
    duraciones_interrupciones = []
    for caller in channel_0:
        for agent in channel_1:
            inicio_overlap = max(caller["start"], agent["start"])
            final_overlap = min(caller["end"], agent["end"])
            if inicio_overlap < final_overlap:
                interrupciones_agente += 1
                duraciones_interrupciones.append(final_overlap - inicio_overlap)

    duracion_total_interrupciones = sum(duraciones_interrupciones)
    duracion_promedio_interrupcion = (
        (duracion_total_interrupciones / len(duraciones_interrupciones))
        if duraciones_interrupciones
        else 0.0
    )

    anon_id = os.path.splitext(os.path.basename(ruta_json))[0]

    return {
        "anon_id": anon_id,
        "numero_turnos_caller": numero_turnos_caller,
        "duracion_promedio_caller": duracion_promedio_caller,
        "pausa_promedio_caller": pausa_promedio_caller,
        "desviacion_estandar_latencia": desviacion_estandar_latencia,
        "interrupciones_agente": interrupciones_agente,
        "duracion_total_interrupciones": duracion_total_interrupciones,
        "duracion_promedio_interrupcion": duracion_promedio_interrupcion,
    }


def predecir_muestra(metricas: dict, modelo=None, metadata=None, usar_umbral_optimo: bool = True) -> dict:
    """
    Realiza la predicción para un diccionario de métricas.
    Retorna un diccionario con formato compatible con el challenge:
    {"is_synthetic": bool, "confidence": float, "label": str, ...}

    Stage 1 of the escalation cascade calls this with usar_umbral_optimo=False
    so the 0.66 val-tuned threshold is not used as the 75% certainty gate.
    """
    if modelo is None or metadata is None:
        modelo, metadata = cargar_modelo_y_metadata()

    features_esperadas = metadata["features"]
    umbral = metadata["umbral_recomendado"] if usar_umbral_optimo else 0.50

    # Construir DataFrame de una fila
    df_row = pd.DataFrame([metricas])
    df_row = agregar_features_ingenieria(df_row)

    # Asegurar que todas las columnas requeridas existan
    X = df_row[features_esperadas]

    proba_synthetic = float(modelo.predict_proba(X)[0, 1])
    is_synthetic = bool(proba_synthetic >= umbral)
    label = "synthetic" if is_synthetic else "human"
    confidence = proba_synthetic if is_synthetic else (1.0 - proba_synthetic)

    return {
        "anon_id": metricas.get("anon_id", "desconocido"),
        "label": label,
        "is_synthetic": is_synthetic,
        "confidence": round(confidence, 4),
        "probabilidad_synthetic": round(proba_synthetic, 4),
        "umbral_utilizado": umbral,
    }


def predecir_csv(ruta_csv: str, ruta_salida: str = None, modelo=None, metadata=None) -> pd.DataFrame:
    """
    Aplica el modelo a un archivo CSV completo y genera un nuevo dataframe
    con las columnas de predicción.
    """
    if modelo is None or metadata is None:
        modelo, metadata = cargar_modelo_y_metadata()

    df = pd.read_csv(ruta_csv)
    df_feat = agregar_features_ingenieria(df)
    features = metadata["features"]
    umbral = metadata["umbral_recomendado"]

    X = df_feat[features]
    probs_synthetic = modelo.predict_proba(X)[:, 1]
    is_synth = probs_synthetic >= umbral

    df["pred_label"] = np.where(is_synth, "synthetic", "human")
    df["is_synthetic"] = is_synth
    df["prob_synthetic"] = np.round(probs_synthetic, 4)
    df["confidence"] = np.round(
        np.where(is_synth, probs_synthetic, 1.0 - probs_synthetic), 4
    )

    if ruta_salida:
        df.to_csv(ruta_salida, index=False)
        print(f"Predicciones guardadas en: {ruta_salida}")

    return df


def demo_validacion():
    """Ejecuta una demostración de inferencia sobre ejemplos del conjunto de validación."""
    modelo, metadata = cargar_modelo_y_metadata()
    df = pd.read_csv("resultados_turns.csv")
    val_df = df[df["split"] == "val"].copy()

    # Tomar 5 humanos y 5 sintéticos de validación
    muestra_human = val_df[val_df["label"] == "human"].head(5)
    muestra_synth = val_df[val_df["label"] == "synthetic"].head(5)
    muestra = pd.concat([muestra_human, muestra_synth])

    print("=" * 80)
    print("DEMOSTRACIÓN DE PREDICCIONES EN EL CONJUNTO DE VALIDACIÓN")
    print(f"Modelo en uso: {metadata['modelo_seleccionado']} | Umbral: {metadata['umbral_recomendado']}")
    print("=" * 80)

    correctas = 0
    total = len(muestra)

    for _, row in muestra.iterrows():
        res = predecir_muestra(row.to_dict(), modelo, metadata)
        real = row["label"]
        pred = res["label"]
        acierto = "✓ CORRECTO" if real == pred else "✗ ERROR"
        if real == pred:
            correctas += 1

        print(f"ID: {res['anon_id']:20s} | Real: {real:10s} | Pred: {pred:10s} | "
              f"P(Synth): {res['probabilidad_synthetic']:.3f} | Conf: {res['confidence']:.3f} | {acierto}")

    print("-" * 80)
    print(f"Aciertos en muestra de validación: {correctas}/{total} ({correctas/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Predictor Human vs Synthetic para llamadas telefónicas."
    )
    parser.add_argument("--id", type=str, help="anon_id de la llamada a predecir (debe existir en resultados_turns.csv)")
    parser.add_argument("--json", type=str, help="Ruta a un archivo turns/<anon_id>.json para predecir")
    parser.add_argument("--csv", type=str, help="Ruta a un archivo CSV con métricas de turnos")
    parser.add_argument("--output", type=str, help="Ruta para guardar los resultados en CSV (usado con --csv)")
    parser.add_argument("--demo", action="store_true", help="Ejecutar demostración en el split de validación")

    args = parser.parse_args()

    if args.demo:
        demo_validacion()
        return

    if args.json:
        if not os.path.exists(args.json):
            print(f"Error: no existe el archivo {args.json}", file=sys.stderr)
            sys.exit(1)
        metricas = extraer_metricas_de_turns_json(args.json)
        pred = predecir_muestra(metricas)
        print(json.dumps(pred, indent=2, ensure_ascii=False))
        return

    if args.id:
        df = pd.read_csv("resultados_turns.csv")
        fila = df[df["anon_id"] == args.id]
        if fila.empty:
            print(f"Error: no se encontró la llamada con id {args.id} en resultados_turns.csv", file=sys.stderr)
            sys.exit(1)
        pred = predecir_muestra(fila.iloc[0].to_dict())
        real = fila.iloc[0].get("label", "desconocido")
        pred["label_real"] = real
        print(json.dumps(pred, indent=2, ensure_ascii=False))
        return

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: no existe el archivo {args.csv}", file=sys.stderr)
            sys.exit(1)
        out = args.output or "predicciones_" + os.path.basename(args.csv)
        df_preds = predecir_csv(args.csv, out)
        print(f"Procesadas {len(df_preds)} llamadas.")
        print(df_preds[["anon_id", "pred_label", "is_synthetic", "confidence"]].head(10).to_string(index=False))
        return

    # Si no se pasó ningún argumento, ejecutar demo por defecto
    demo_validacion()


if __name__ == "__main__":
    main()
