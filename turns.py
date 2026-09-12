import json
import os
import statistics
import pandas as pd


# ============================================
# CONFIGURACIÓN
# ============================================

CARPETA_TURNS = "turns"
MANIFEST = "manifest.csv"
OUTPUT_CSV = "resultados_turns.csv"


# ============================================
# ANALIZAR UNA LLAMADA
# ============================================

def analizar_llamada(ruta_json: str) -> dict:
    """
    Extrae las 7 características fundamentales de conversación
    a partir de un archivo JSON de turnos:
      1. numero_turnos_caller
      2. duracion_promedio_caller
      3. pausa_promedio_caller
      4. desviacion_estandar_latencia
      5. interrupciones_agente
      6. duracion_total_interrupciones
      7. duracion_promedio_interrupcion
    """
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    turns = data.get("turns", [])

    # Separar canales: channel 0 = caller, channel 1 = agente
    channel_0 = [t for t in turns if t["channel"] == 0]
    channel_1 = [t for t in turns if t["channel"] == 1]

    # Ordenar cronológicamente
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
    pausas = [
        channel_0[i + 1]["start"] - channel_0[i]["end"]
        for i in range(len(channel_0) - 1)
    ]
    pausa_promedio_caller = (
        sum(pausas) / len(pausas) if pausas else 0.0
    )

    # 4. Latencias entre fin de turno del agente e inicio del caller
    latencias = []
    for caller in channel_0:
        ultimo_agente = None
        for agent in channel_1:
            if agent["end"] <= caller["start"]:
                if ultimo_agente is None or agent["end"] > ultimo_agente:
                    ultimo_agente = agent["end"]
        if ultimo_agente is not None:
            latencias.append(caller["start"] - ultimo_agente)

    # 5. Desviación estándar de latencias del caller
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

            # Si el agente habló mientras el caller seguía hablando
            if inicio_overlap < final_overlap:
                interrupciones_agente += 1
                duraciones_interrupciones.append(final_overlap - inicio_overlap)

    duracion_total_interrupciones = (
        sum(duraciones_interrupciones) if duraciones_interrupciones else 0.0
    )
    duracion_promedio_interrupcion = (
        duracion_total_interrupciones / len(duraciones_interrupciones)
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


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

def main():
    manifest = pd.read_csv(MANIFEST)
    print("Manifest cargado.")
    print("Número de llamadas:", len(manifest))

    resultados = []
    archivos = sorted(os.listdir(CARPETA_TURNS))

    for archivo in archivos:
        if not archivo.endswith(".json"):
            continue

        ruta_json = os.path.join(CARPETA_TURNS, archivo)
        try:
            resultado = analizar_llamada(ruta_json)
            resultados.append(resultado)
            print(f"Analizado: {archivo}")
        except Exception as e:
            print(f"ERROR en {archivo}: {e}")

    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)

    # Agregar label y split del manifest
    df_resultados = df_resultados.merge(
        manifest[["anon_id", "label", "split"]],
        on="anon_id",
        how="left"
    )

    # Guardar CSV
    df_resultados.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 110)
    print(f"ARCHIVO GUARDADO: {OUTPUT_CSV} ({len(df_resultados)} llamadas)")
    print("=" * 110)

    # Promedios Human vs Synthetic
    columnas_comparacion = [
        "numero_turnos_caller",
        "duracion_promedio_caller",
        "pausa_promedio_caller",
        "desviacion_estandar_latencia",
        "interrupciones_agente",
        "duracion_total_interrupciones",
        "duracion_promedio_interrupcion",
    ]

    comparacion = df_resultados.groupby("label")[columnas_comparacion].mean()

    print("\n" + "=" * 110)
    print("PROMEDIOS: HUMAN VS SYNTHETIC")
    print("=" * 110)
    print(comparacion.to_string())
    print("\n" + "=" * 110)
    print("DIFERENCIAS TRANSPUESTAS")
    print("=" * 110)
    print(comparacion.T)


if __name__ == "__main__":
    main()