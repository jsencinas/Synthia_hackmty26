import json
import os
import numpy as np
import pandas as pd
import librosa

CARPETA_AUDIO = "audio"
CARPETA_TURNS = "turns"
MANIFEST = "manifest.csv"


def analizar_llamada(ruta_audio, ruta_json):

    audio, frecuencia = librosa.load(
        ruta_audio,
        sr=None,
        mono=False
    )

    caller_audio = audio[0]

    with open(ruta_json, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    turns = data["turns"]

    segmentos_caller = []

    for turn in turns:

        if turn["channel"] == 0:

            inicio = int(turn["start"] * frecuencia)
            final = int(turn["end"] * frecuencia)

            segmento = caller_audio[inicio:final]

            if len(segmento) > 0:
                segmentos_caller.append(segmento)

    if len(segmentos_caller) == 0:
        return None

    caller_completo = np.concatenate(segmentos_caller)


    # ============================================================
    # PITCH
    # ============================================================

    f0 = librosa.yin(
        caller_completo,
        fmin=70,
        fmax=400,
        sr=frecuencia
    )

    f0_validos = f0[np.isfinite(f0)]

    if len(f0_validos) > 1:

        pitch_deltas = np.abs(np.diff(f0_validos))

        pitch_delta_std = np.std(pitch_deltas)

    else:

        pitch_delta_std = 0


    # ============================================================
    # JITTER
    # ============================================================

    if len(f0_validos) > 1:

        periodos = 1 / f0_validos

        diferencias_periodo = np.abs(
            np.diff(periodos)
        )

        jitter = (
            np.mean(diferencias_periodo)
            / np.mean(periodos)
        )

    else:

        jitter = 0


    # ============================================================
    # SHIMMER
    # ============================================================

    if len(caller_completo) >= 1024:

        frames = librosa.util.frame(
            caller_completo,
            frame_length=1024,
            hop_length=256
        )

        amplitudes = np.mean(
            np.abs(frames),
            axis=0
        )

        if len(amplitudes) > 1:

            diferencias_amplitud = np.abs(
                np.diff(amplitudes)
            )

            promedio_amplitud = np.mean(
                amplitudes
            )

            if promedio_amplitud != 0:

                shimmer = (
                    np.mean(diferencias_amplitud)
                    / promedio_amplitud
                )

            else:

                shimmer = 0

        else:

            shimmer = 0

    else:

        shimmer = 0


    # ============================================================
    # MFCC 1
    # ============================================================

    mfcc = librosa.feature.mfcc(
        y=caller_completo,
        sr=frecuencia,
        n_mfcc=13
    )

    mfcc_1 = mfcc[0]

    mfcc_1_std = np.std(mfcc_1)


    # ============================================================
    # IDENTIFICADOR
    # ============================================================

    nombre_archivo = os.path.basename(ruta_audio)

    anon_id = os.path.splitext(nombre_archivo)[0]


    # ============================================================
    # RESULTADOS
    # ============================================================

    return {
        "anon_id": anon_id,

        "mfcc_1_std": mfcc_1_std,
        "shimmer": shimmer,
        "pitch_delta_std": pitch_delta_std,
        "jitter": jitter
    }


# ================================================================
# CARGAR MANIFEST
# ================================================================

manifest = pd.read_csv(MANIFEST)

print("Manifest cargado.")
print("Número de llamadas:", len(manifest))


# ================================================================
# ANALIZAR TODAS LAS LLAMADAS
# ================================================================

resultados = []

archivos = os.listdir(CARPETA_AUDIO)

for archivo in archivos:

    if not archivo.endswith(".wav"):
        continue

    ruta_audio = os.path.join(
        CARPETA_AUDIO,
        archivo
    )

    anon_id = os.path.splitext(archivo)[0]

    ruta_json = os.path.join(
        CARPETA_TURNS,
        anon_id + ".json"
    )

    if not os.path.exists(ruta_json):

        print(
            "No existe JSON para:",
            archivo
        )

        continue

    try:

        resultado = analizar_llamada(
            ruta_audio,
            ruta_json
        )

        if resultado is not None:

            resultados.append(resultado)

            print(
                "Analizado:",
                archivo
            )

    except Exception as e:

        print(
            "ERROR en",
            archivo,
            ":",
            e
        )


# ================================================================
# CREAR DATAFRAME
# ================================================================

df_resultados = pd.DataFrame(resultados)


# ================================================================
# AGREGAR LABEL Y SPLIT
# ================================================================

df_resultados = df_resultados.merge(
    manifest[
        [
            "anon_id",
            "label",
            "split"
        ]
    ],
    on="anon_id",
    how="left"
)


# ================================================================
# MOSTRAR RESULTADOS
# ================================================================

print("\n")
print("=" * 100)
print("CARACTERÍSTICAS SELECCIONADAS")
print("=" * 100)

print(
    df_resultados.to_string(index=False)
)


# ================================================================
# PROMEDIOS HUMAN VS SYNTHETIC
# ================================================================

columnas_audio = [
    "mfcc_1_std",
    "shimmer",
    "pitch_delta_std",
    "jitter"
]

comparacion = df_resultados.groupby(
    "label"
)[columnas_audio].mean()


print("\n")
print("=" * 100)
print("PROMEDIO DE LAS CARACTERÍSTICAS")
print("=" * 100)

print(
    comparacion.to_string()
)


# ================================================================
# GUARDAR CSV
# ================================================================

df_resultados.to_csv(
    "caracteristicas_finales.csv",
    index=False
)


print("\n")
print("=" * 100)
print("ARCHIVO GUARDADO")
print("=" * 100)

print(
    "Se creó: caracteristicas_finales.csv"
)