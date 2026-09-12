import json
import os
import pandas as pd
import statistics


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


    # ========================================
    # SEPARAR CALLER Y AGENTE
    # ========================================

    channel_0 = []
    channel_1 = []

    for turn in turns:

        if turn["channel"] == 0:
            channel_0.append(turn)

        else:
            channel_1.append(turn)


    # Ordenar cronológicamente
    channel_0.sort(key=lambda x: x["start"])
    channel_1.sort(key=lambda x: x["start"])


    # ========================================
    # 1. NÚMERO DE TURNOS DEL CALLER
    # ========================================

    numero_turnos_caller = len(channel_0)


    # ========================================
    # 2. DURACIÓN PROMEDIO DE LOS TURNOS
    # ========================================

    duraciones = []

    for turn in channel_0:

        duracion = turn["end"] - turn["start"]

        duraciones.append(duracion)


    if len(duraciones) > 0:

        duracion_promedio_caller = (
            sum(duraciones) / len(duraciones)
        )

    else:

        duracion_promedio_caller = 0


    # ========================================
    # 3. PAUSA PROMEDIO ENTRE TURNOS DEL CALLER
    # ========================================

    pausas = []

    for i in range(len(channel_0) - 1):

        fin_actual = channel_0[i]["end"]

        inicio_siguiente = channel_0[i + 1]["start"]

        pausa = inicio_siguiente - fin_actual

        pausas.append(pausa)


    if len(pausas) > 0:

        pausa_promedio_caller = (
            sum(pausas) / len(pausas)
        )

    else:

        pausa_promedio_caller = 0


    # ========================================
    # 4. LATENCIAS DEL CALLER
    # ========================================

    latencias = []

    for caller in channel_0:

        anterior_agente = None

        # Buscar el último turno del agente
        # que terminó antes de que el caller empezara

        for agent in channel_1:

            if agent["end"] <= caller["start"]:

                if anterior_agente is None:

                    anterior_agente = agent["end"]

                elif agent["end"] > anterior_agente:

                    anterior_agente = agent["end"]


        if anterior_agente is not None:

            latencia = (
                caller["start"] - anterior_agente
            )

            latencias.append(latencia)


    # ========================================
    # 5. DESVIACIÓN ESTÁNDAR DE LA LATENCIA
    # ========================================

    if len(latencias) > 1:

        desviacion_estandar_latencia = statistics.stdev(
            latencias
        )

    else:

        desviacion_estandar_latencia = 0


    # ========================================
    # 6. INTERRUPCIONES DEL AGENTE
    # ========================================

    interrupciones_agente = 0

    duraciones_interrupciones = []


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


            # Si hay overlap significa que
            # el agente habló mientras
            # el caller todavía estaba hablando

            if inicio_overlap < final_overlap:

                duracion_overlap = (
                    final_overlap - inicio_overlap
                )

                interrupciones_agente += 1

                duraciones_interrupciones.append(
                    duracion_overlap
                )


    # ========================================
    # 7. DURACIÓN TOTAL DE INTERRUPCIONES
    # ========================================

    if len(duraciones_interrupciones) > 0:

        duracion_total_interrupciones = sum(
            duraciones_interrupciones
        )

    else:

        duracion_total_interrupciones = 0


    # ========================================
    # 8. DURACIÓN PROMEDIO DE INTERRUPCIONES
    # ========================================

    if len(duraciones_interrupciones) > 0:

        duracion_promedio_interrupcion = (
            duracion_total_interrupciones
            / len(duraciones_interrupciones)
        )

    else:

        duracion_promedio_interrupcion = 0


    # ========================================
    # ID DEL ARCHIVO
    # ========================================

    nombre_archivo = os.path.basename(ruta_json)

    anon_id = os.path.splitext(nombre_archivo)[0]


    # ========================================
    # RESULTADO
    # ========================================

    return {

        "anon_id":
            anon_id,

        "numero_turnos_caller":
            numero_turnos_caller,

        "duracion_promedio_caller":
            duracion_promedio_caller,

        "pausa_promedio_caller":
            pausa_promedio_caller,

        "desviacion_estandar_latencia":
            desviacion_estandar_latencia,

        "interrupciones_agente":
            interrupciones_agente,

        "duracion_total_interrupciones":
            duracion_total_interrupciones,

        "duracion_promedio_interrupcion":
            duracion_promedio_interrupcion
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

    # Solo analizar archivos JSON
    if not archivo.endswith(".json"):
        continue


    ruta_json = os.path.join(
        CARPETA_TURNS,
        archivo
    )


    try:

        resultado = analizar_llamada(ruta_json)

        resultados.append(resultado)

        print("Analizado:", archivo)


    except Exception as e:

        print(
            "ERROR en",
            archivo,
            ":",
            e
        )


# ============================================
# CREAR DATAFRAME
# ============================================

df_resultados = pd.DataFrame(resultados)


# ============================================
# AGREGAR LABEL Y SPLIT
# ============================================

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


# ============================================
# MOSTRAR TABLA
# ============================================

print("\n")
print("=" * 110)
print("RESULTADOS DE TODAS LAS LLAMADAS")
print("=" * 110)

print(
    df_resultados.to_string(index=False)
)


# ============================================
# PROMEDIOS HUMAN VS SYNTHETIC
# ============================================

comparacion = df_resultados.groupby("label")[
    [
        "numero_turnos_caller",
        "duracion_promedio_caller",
        "pausa_promedio_caller",
        "desviacion_estandar_latencia",
        "interrupciones_agente",
        "duracion_total_interrupciones",
        "duracion_promedio_interrupcion"
    ]
].mean()


# ============================================
# MOSTRAR PROMEDIOS
# ============================================

print("\n")
print("=" * 110)
print("PROMEDIOS: HUMAN VS SYNTHETIC")
print("=" * 110)

print(
    comparacion.to_string()
)


# ============================================
# GUARDAR CSV
# ============================================

df_resultados.to_csv(
    "resultados_turns.csv",
    index=False
)


print("\n")
print("=" * 110)
print("ARCHIVO GUARDADO")
print("=" * 110)

print(
    "Se creó: resultados_turns.csv"
)