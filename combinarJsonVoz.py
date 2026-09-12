import os
import json
import pandas as pd
import numpy as np
import librosa

from xgboost import XGBClassifier

from predict import (
    cargar_modelo_y_metadata,
    extraer_metricas_de_turns_json,
    predecir_muestra
)


CARPETA_TURNS = "turns"
CARPETA_AUDIO = "audio"

MODELO_AUDIO_PATH = "modelo_xgboost.json"
MANIFEST_PATH = "manifest.csv"

# Si el modelo de JSON tiene esta confianza o más, nos quedamos con su decisión.
# Si no, bajamos un nivel y usamos el modelo de audio.
UMBRAL_WATERFALL = 0.90


def _cargar_turns(ruta_json):
    """
    Carga la lista de turnos desde un archivo JSON, soportando dos formatos:
    - {"turns": [...]}   (formato usado por predict.py y por el generador de resultados_turns.csv)
    - [...]              (lista de turnos directamente)
    """
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    if isinstance(data, dict):
        return data.get("turns", [])

    return data


def extraer_features_audio(ruta_audio, ruta_json):
    """
    Extrae las 4 features acústicas del canal del caller (channel 0):
    mfcc_1_std, shimmer, pitch_delta_std, jitter.
    """

    audio, frecuencia = librosa.load(
        ruta_audio,
        sr=None,
        mono=False
    )

    # Si el audio es mono, librosa devuelve un array 1D en vez de (canales, muestras).
    # audio[0] en ese caso tomaría la primera MUESTRA, no el primer CANAL.
    if audio.ndim == 1:
        caller = audio
    else:
        caller = audio[0]

    turns = _cargar_turns(ruta_json)

    segmentos_caller = []

    for turno in turns:

        if turno["channel"] == 0:

            inicio = int(turno["start"] * frecuencia)
            fin = int(turno["end"] * frecuencia)

            segmento = caller[inicio:fin]

            if len(segmento) > 0:
                segmentos_caller.append(segmento)

    if len(segmentos_caller) == 0:
        raise ValueError("No se encontraron segmentos del caller.")

    caller_completo = np.concatenate(segmentos_caller)

    mfcc = librosa.feature.mfcc(
        y=caller_completo,
        sr=frecuencia,
        n_mfcc=13
    )

    mfcc_1_std = float(np.std(mfcc[0]))

    f0 = librosa.yin(
        caller_completo,
        fmin=70,
        fmax=400,
        sr=frecuencia
    )

    f0_validos = f0[np.isfinite(f0)]

    if len(f0_validos) > 1:

        cambios_pitch = np.diff(f0_validos)

        pitch_delta_std = float(
            np.std(cambios_pitch)
        )

    else:

        pitch_delta_std = 0.0

    if len(f0_validos) > 1:

        periodos = 1.0 / f0_validos

        diferencias_periodo = np.abs(
            np.diff(periodos)
        )

        jitter = float(
            np.mean(diferencias_periodo)
            / np.mean(periodos)
        )

    else:

        jitter = 0.0

    tamano_frame = 1024
    hop = 256

    amplitudes = []

    inicio = 0

    while inicio + tamano_frame <= len(caller_completo):

        frame = caller_completo[
            inicio:inicio + tamano_frame
        ]

        amplitud = np.mean(np.abs(frame))

        amplitudes.append(amplitud)

        inicio += hop

    amplitudes = np.array(amplitudes)

    if len(amplitudes) > 1 and np.mean(amplitudes) > 0:

        diferencias_amplitud = np.abs(
            np.diff(amplitudes)
        )

        shimmer = float(
            np.mean(diferencias_amplitud)
            / np.mean(amplitudes)
        )

    else:

        shimmer = 0.0

    features = {
        "mfcc_1_std": mfcc_1_std,
        "shimmer": shimmer,
        "pitch_delta_std": pitch_delta_std,
        "jitter": jitter
    }

    return features


def predecir_audio(features, modelo_audio):

    columnas = [
        "mfcc_1_std",
        "shimmer",
        "pitch_delta_std",
        "jitter"
    ]

    X = pd.DataFrame(
        [[features[columna] for columna in columnas]],
        columns=columnas
    )

    probabilidad_synthetic = float(
        modelo_audio.predict_proba(X)[0, 1]
    )

    is_synthetic = bool(
        probabilidad_synthetic >= 0.50
    )

    if is_synthetic:

        label = "synthetic"
        confidence = probabilidad_synthetic

    else:

        label = "human"
        confidence = 1.0 - probabilidad_synthetic

    return {
        "label": label,
        "is_synthetic": is_synthetic,
        "confidence": confidence,
        "probabilidad_synthetic": probabilidad_synthetic
    }


def predecir_waterfall(ruta_json, ruta_audio, modelo_json, metadata_json, modelo_audio):
    """
    Nivel 1: modelo de turnos/JSON.
    Si su confianza >= UMBRAL_WATERFALL, esa es la decisión final.
    Si no, Nivel 2: modelo de audio (XGBoost con features acústicas).
    """

    metricas_json = extraer_metricas_de_turns_json(
        ruta_json
    )

    resultado_json = predecir_muestra(
        metricas_json,
        modelo_json,
        metadata_json
    )

    if resultado_json["confidence"] >= UMBRAL_WATERFALL:

        return {
            "anon_id": resultado_json["anon_id"],
            "label": resultado_json["label"],
            "is_synthetic": resultado_json["is_synthetic"],
            "confidence": resultado_json["confidence"],
            "nivel": "JSON",
            "probabilidad_synthetic": resultado_json[
                "probabilidad_synthetic"
            ]
        }

    features_audio = extraer_features_audio(
        ruta_audio,
        ruta_json
    )

    resultado_audio = predecir_audio(
        features_audio,
        modelo_audio
    )

    nombre_archivo = os.path.basename(ruta_json)

    anon_id = os.path.splitext(
        nombre_archivo
    )[0]

    return {
        "anon_id": anon_id,
        "label": resultado_audio["label"],
        "is_synthetic": resultado_audio["is_synthetic"],
        "confidence": resultado_audio["confidence"],
        "nivel": "AUDIO_XGBOOST",
        "probabilidad_synthetic": resultado_audio[
            "probabilidad_synthetic"
        ]
    }


def procesar_todas_las_llamadas():

    modelo_json, metadata_json = cargar_modelo_y_metadata()

    modelo_audio = XGBClassifier()

    modelo_audio.load_model(
        MODELO_AUDIO_PATH
    )

    archivos_json = os.listdir(
        CARPETA_TURNS
    )

    resultados = []

    for i in range(len(archivos_json)):

        archivo_json = archivos_json[i]

        if not archivo_json.endswith(".json"):
            continue

        nombre_base = os.path.splitext(
            archivo_json
        )[0]

        ruta_json = os.path.join(
            CARPETA_TURNS,
            archivo_json
        )

        ruta_audio = os.path.join(
            CARPETA_AUDIO,
            nombre_base + ".wav"
        )

        print("Procesando:", nombre_base)

        if not os.path.exists(ruta_audio):

            print("No existe el audio correspondiente.")
            print()

            continue

        try:

            resultado = predecir_waterfall(
                ruta_json,
                ruta_audio,
                modelo_json,
                metadata_json,
                modelo_audio
            )

            resultados.append(resultado)

            print(
                "Resultado:",
                resultado["label"]
            )

            print(
                "Confianza:",
                round(
                    resultado["confidence"],
                    4
                )
            )

            print(
                "Nivel:",
                resultado["nivel"]
            )

            print()

        except Exception as error:

            print(
                "Error:",
                error
            )

            print()

    if len(resultados) > 0:

        df_resultados = pd.DataFrame(
            resultados
        )

        df_resultados.to_csv(
            "resultados_waterfall.csv",
            index=False
        )

        print("Proceso terminado.")
        print("Resultados guardados en:")
        print("resultados_waterfall.csv")

        calcular_accuracy(df_resultados)

    else:

        print("No se procesaron llamadas.")


def calcular_accuracy(df_resultados):
    """
    Cruza los resultados del waterfall contra las etiquetas reales del
    manifest y calcula accuracy global y por nivel (JSON vs AUDIO_XGBOOST).
    """

    if not os.path.exists(MANIFEST_PATH):

        print()
        print("No se encontró", MANIFEST_PATH, "- no se puede calcular accuracy.")

        return

    manifest = pd.read_csv(MANIFEST_PATH)

    manifest_renombrado = manifest[["anon_id", "label"]].rename(
        columns={"label": "label_real"}
    )

    df_comparado = df_resultados.merge(
        manifest_renombrado,
        on="anon_id",
        how="left"
    )

    df_comparado = df_comparado.dropna(subset=["label_real"])

    if len(df_comparado) == 0:

        print()
        print("Ningún anon_id de los resultados coincide con el manifest.")

        return

    df_comparado["correcto"] = (
        df_comparado["label"] == df_comparado["label_real"]
    )

    print()
    print("=" * 60)
    print("ACCURACY DEL MODELO COMBINADO (WATERFALL)")
    print("=" * 60)

    accuracy_global = df_comparado["correcto"].mean()

    print(
        "Accuracy global:",
        round(accuracy_global, 4),
        f"({df_comparado['correcto'].sum()}/{len(df_comparado)})"
    )

    print()
    print("Accuracy por nivel del waterfall:")

    for nivel, grupo in df_comparado.groupby("nivel"):

        acc_nivel = grupo["correcto"].mean()

        print(
            f"  {nivel:15s}: {round(acc_nivel, 4)} "
            f"({grupo['correcto'].sum()}/{len(grupo)} llamadas)"
        )


if __name__ == "__main__":

    procesar_todas_las_llamadas()