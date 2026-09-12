import json
import os
import pandas as pd


# ============================================
# CONFIGURACIÓN
# ============================================

CARPETA_TURNS = "turns"
MANIFEST = "manifest.csv"


# ============================================
# ANALIZAR UNA LLAMADA
# ============================================

def analizar_llamada(ruta_json):

    # Leer JSON
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    turns = data["turns"]

    # Separar canales
    channel_0 = [
        turn for turn in turns
        if turn["channel"] == 0
    ]

    channel_1 = [
        turn for turn in turns
        if turn["channel"] == 1
    ]

    # Ordenar cronológicamente el caller para el cálculo de pausas
    channel_0.sort(key=lambda x: x["start"])


    # ========================================
    # CHANNEL 0
    # DURACIÓN DE CADA INTERVENCIÓN
    # ========================================

    duraciones_caller = [turn["end"] - turn["start"] for turn in channel_0]

    # Promedio
    if duraciones_caller:
        promedio_duracion_caller = (
            sum(duraciones_caller)
            / len(duraciones_caller)
        )
    else:
        promedio_duracion_caller = 0

    # ========================================
    # PAUSAS ENTRE INTERVENCIONES DEL CALLER
    # ========================================

    pausas = [
        channel_0[i + 1]["start"] - channel_0[i]["end"]
        for i in range(len(channel_0) - 1)
    ]

    if pausas:
        promedio_pausa = (
            sum(pausas)
            / len(pausas)
        )
    else:
        promedio_pausa = 0

    # ========================================
    # INTERRUPCIONES / OVERLAPS
    # ========================================

    interrupciones = []

    for caller in channel_0:
        for agent in channel_1:
            inicio_overlap = max(
                caller["start"],
                agent["start"]
            )
            final_overlap = min(
                caller["end"],
                agent["end"]
            )

            # Si se están superponiendo
            if inicio_overlap < final_overlap:
                interrupciones.append(
                    final_overlap - inicio_overlap
                )

    # Promedio de interrupción
    if interrupciones:
        promedio_interrupcion = (
            sum(interrupciones)
            / len(interrupciones)
        )
    else:
        promedio_interrupcion = 0

    # ========================================
    # RESULTADO
    # ========================================

    anon_id = os.path.splitext(os.path.basename(ruta_json))[0]

    return {
        "anon_id": anon_id,
        "intervenciones_caller": len(channel_0),
        "duracion_promedio_caller": promedio_duracion_caller,
        "pausa_promedio": promedio_pausa,
        "interrupciones": len(interrupciones),
        "interrupcion_promedio": promedio_interrupcion,
    }


# ============================================
# LEER MANIFEST
# ============================================

manifest = pd.read_csv(MANIFEST)

print("Manifest cargado.")
print("Número de llamadas:", len(manifest))


# ============================================
# ANALIZAR TODOS LOS JSON
# ============================================

resultados = []

archivos = os.listdir(CARPETA_TURNS)

for archivo in archivos:

    # Solo JSON
    if not archivo.endswith(".json"):
        continue

    ruta_json = os.path.join(
        CARPETA_TURNS,
        archivo
    )

    try:

        resultado = analizar_llamada(ruta_json)

        resultados.append(resultado)

        print(
            f"Analizado: {archivo}"
        )

    except Exception as e:

        print(
            f"ERROR en {archivo}: {e}"
        )


# ============================================
# CREAR DATAFRAME
# ============================================

df_resultados = pd.DataFrame(resultados)


# ============================================
# AGREGAR LABEL DEL MANIFEST
# ============================================

df_resultados = df_resultados.merge(
    manifest[["anon_id", "label"]],
    on="anon_id",
    how="left"
)


# ============================================
# GUARDAR RESULTADOS
# ============================================

df_resultados.to_csv(
    "resultados_turns.csv",
    index=False
)


# ============================================
# MOSTRAR RESULTADOS
# ============================================

print("\n")
print("=" * 60)
print("ANÁLISIS COMPLETO")
print("=" * 60)

print(
    "JSON analizados:",
    len(df_resultados)
)

print(
    "Human:",
    len(
        df_resultados[
            df_resultados["label"] == "human"
        ]
    )
)

print(
    "Synthetic:",
    len(
        df_resultados[
            df_resultados["label"] == "synthetic"
        ]
    )
)


# ============================================
# PROMEDIOS HUMAN VS SYNTHETIC
# ============================================

comparacion = df_resultados.groupby("label")[
    [
        "intervenciones_caller",
        "duracion_promedio_caller",
        "pausa_promedio",
        "interrupciones",
        "interrupcion_promedio"
    ]
].mean()


print("\n")
print("=" * 60)
print("HUMAN VS SYNTHETIC")
print("=" * 60)

print(
    comparacion.to_string()
)


# ============================================
# DIFERENCIA ENTRE HUMAN Y SYNTHETIC
# ============================================

print("\n")
print("=" * 60)
print("DIFERENCIAS")
print("=" * 60)

print(comparacion.T)