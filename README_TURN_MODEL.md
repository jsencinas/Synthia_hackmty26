# Cómo funciona el detector

El sistema recibe una llamada grabada y responde: **¿quien llama es una persona o una IA?**

Hay tres comandos importantes:

```bash
python -m src.train      # entrena los modelos
python -m src.evaluate   # mide qué tan bien salieron
uvicorn api.app:app      # pone el detector a escuchar
```

---

## ¿Qué hay en cada llamada?

El audio es estéreo (dos canales), como si cada persona tuviera su propio micrófono:

- **Canal 0**: quien llama. Es a quien hay que clasificar.
- **Canal 1**: el agente del banco. Siempre es el mismo lado.

A veces quien llama es humano. A veces es un robot: entiende lo que dice el banco, piensa una respuesta y la lee con voz sintética.

Los humanos dudan, se tardan distinto cada vez y su voz trepida. Las IAs suelen ser más regulares, más planas y más “perfectas” al repetir datos.

Por eso no miramos una sola cosa. Miramos **el ritmo de la charla**, **cómo suena la voz** y, si hace falta, **qué se dijo**.

---

## Archivos que usa el proyecto

| Archivo | Para qué sirve |
|---|---|
| `manifest.csv` | Lista de llamadas: id, si es humana o sintética, y si es de entrenamiento o de prueba. |
| `audio/<id>.wav` | El audio de esa llamada. |
| `turns/` | Recortes de habla de ejemplo. El sistema **no los usa**: calcula los recortes él mismo a partir del WAV. |

Las llamadas de entrenamiento (`train`) y las de prueba (`val`) no comparten las mismas personas. Así el modelo no “memoriza voces” y luego aparenta ser bueno.

---

## Paso 1: recortar quién habla y cuándo

Antes de clasificar, el sistema busca en el audio los momentos con voz (por energía: dónde hay sonido fuerte y dónde hay silencio).

Eso da una lista de turnos: “el caller habló de 12.4 s a 15.1 s”, “el agente habló de 15.3 s a 18.0 s”, etc.

Con esos recortes se calculan dos grupos de pistas.

### Ritmo de la conversación (timing)

Diez números que resumen **cómo** habla quien llama, no **qué** dice:

- ¿Cuántas veces tomó la palabra?
- ¿Cuánto dura cada intervención?
- ¿Cuánto calla entre una y otra?
- ¿Siempre responde al mismo tiempo, o a veces rápido y a veces piensa?
- ¿Se pisa con el agente? ¿Cuánto duran esos choques?

Las IAs suelen tener tiempos más constantes. Los humanos varían más.

### Sonido de la voz (voice)

Cuatro números sobre el audio del caller:

- ¿Cambia el timbre?
- ¿Temblor de volumen (shimmer)?
- ¿Temblor de tono (jitter)?
- ¿El pitch sube y baja, o se queda plano?

Una voz sintética suele ser más estable. Una voz humana trepida un poco.

---

## Paso 2: entrenar

`python -m src.train` usa **solo** las llamadas marcadas como `train` (hoy 282).

Entrena tres piezas:

1. **Modelo de timing**: mira el ritmo y dice qué tan probable es que sea IA.
2. **Modelo de voz**: mira la acústica y dice lo mismo.
3. **Combinador**: junta las dos opiniones (y también si están de acuerdo o no).

Hay un ajuste extra llamado **temperatura**: suaviza o endurece la probabilidad del modelo de ritmo para que el “qué tan seguro estoy” sea más honesto.

Las llamadas de `val` no se tocan en este paso. Sirven después, para medir de verdad.

Al terminar, guarda todo en `models/`.

---

## Paso 3: decidir en una llamada nueva

Cuando llega un WAV, el detector hace esto:

```
1. Recorta quién habla y cuándo
2. Pregunta al modelo de ritmo
3. Pregunta al modelo de voz
4. Combina las dos respuestas
```

Si ritmo y voz **coinciden** (los dos dicen humano, o los dos dicen IA), se queda con esa respuesta combinada y termina.

Si **no coinciden**, o si uno falló, pide una tercera opinión: transcribe la llamada (ElevenLabs) y mira el texto.

En el texto busca cosas simples:

- Si el banco pide repetir un número y el caller lo suelta **igualito**, eso parece IA.
- Si el caller dice “mande”, “cómo”, “no tengo”, o usa muletillas (“bueno”, “este”, “o sea”), eso parece humano.

Si no hay clave de ElevenLabs, o la transcripción falla, se usa la combinación de ritmo + voz.

La respuesta final es:

```json
{"is_synthetic": true, "confidence": 0.87}
```

- `is_synthetic`: `true` si cree que es IA, `false` si cree que es persona.
- `confidence`: qué tan seguro está de esa decisión, de 0.5 (casi a ciegas) a 1.0 (muy seguro).

El corte es el 50%: arriba es IA, abajo es humano.

---

## Cómo correrlo

Hace falta Python 3.11+ y el audio en `audio/`.

```bash
python -m pip install -r requirements.txt
python -m src.train
python -m src.evaluate
uvicorn api.app:app
```

`evaluate` no vuelve a entrenar. Solo prueba los modelos ya guardados con las llamadas de `val` y muestra qué tan bien clasifican.

La API (`POST /detect`) recibe el WAV en base64 y exige audio estéreo a 8 kHz. Si faltan los modelos, responde error.

Para transcribir (la tercera opinión) hay que definir `ELEVENLABS_API_KEY`.

---

## Dónde está cada cosa

| Archivo | Qué hace |
|---|---|
| `src/features.py` | Recorta el habla y calcula ritmo + voz. |
| `src/timing.py` | Modelo del ritmo. |
| `src/voice.py` | Modelo de la voz. |
| `src/stt.py` | Transcribe y mira el texto. |
| `src/detect.py` | Junta las tres opiniones. |
| `src/train.py` | Entrena con `train`. |
| `src/evaluate.py` | Mide con `val`. |
| `api/app.py` | Endpoint `/detect`. |
| `models/` | Modelos ya entrenados. |
