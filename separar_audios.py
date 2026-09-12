import os
import soundfile as sf


# ==========================================
# CONFIGURACIÓN
# ==========================================

CARPETA_AUDIO = "audio"
CARPETA_SALIDA = "audio_separado"


# ==========================================
# CREAR CARPETA DE SALIDA
# ==========================================

os.makedirs(CARPETA_SALIDA, exist_ok=True)


# ==========================================
# PROCESAR CADA AUDIO
# ==========================================

archivos = os.listdir(CARPETA_AUDIO)

contador = 0

for archivo in archivos:

    if not archivo.lower().endswith(".wav"):
        continue

    ruta_audio = os.path.join(CARPETA_AUDIO, archivo)

    try:

        # ------------------------------------------
        # LEER AUDIO
        # ------------------------------------------

        audio, sample_rate = sf.read(ruta_audio)

        print("\nProcesando:", archivo)
        print("Sample rate:", sample_rate)
        print("Dimensiones:", audio.shape)

        # ------------------------------------------
        # VERIFICAR QUE SEA ESTÉREO
        # ------------------------------------------

        if len(audio.shape) != 2 or audio.shape[1] < 2:
            print("ERROR: El audio no tiene dos canales.")
            continue

        # ------------------------------------------
        # SEPARAR CANALES
        # ------------------------------------------

        channel_0 = audio[:, 0]
        channel_1 = audio[:, 1]

        # ------------------------------------------
        # OBTENER ID
        # ------------------------------------------

        anon_id = os.path.splitext(archivo)[0]

        # ------------------------------------------
        # CREAR CARPETA DEL AUDIO
        # ------------------------------------------

        carpeta_call = os.path.join(
            CARPETA_SALIDA,
            anon_id
        )

        os.makedirs(carpeta_call, exist_ok=True)

        # ------------------------------------------
        # RUTAS DE SALIDA
        # ------------------------------------------

        ruta_channel_0 = os.path.join(
            carpeta_call,
            "channel_0_caller.wav"
        )

        ruta_channel_1 = os.path.join(
            carpeta_call,
            "channel_1_agent.wav"
        )

        # ------------------------------------------
        # GUARDAR CHANNEL 0
        # ------------------------------------------

        sf.write(
            ruta_channel_0,
            channel_0,
            sample_rate
        )

        # ------------------------------------------
        # GUARDAR CHANNEL 1
        # ------------------------------------------

        sf.write(
            ruta_channel_1,
            channel_1,
            sample_rate
        )

        print("  Channel 0 →", ruta_channel_0)
        print("  Channel 1 →", ruta_channel_1)

        contador += 1

    except Exception as e:

        print(
            "ERROR procesando",
            archivo,
            ":",
            e
        )


# ==========================================
# RESUMEN
# ==========================================

print("\n" + "=" * 60)
print("PROCESO TERMINADO")
print("=" * 60)

print("Audios procesados:", contador)
print("Salida:", CARPETA_SALIDA)
```
