# Memory Leak AI Detector

Este proyecto recibe una llamada grabada y determina si quien llama es una **persona o una IA**.

El modelo analiza principalmente **cómo se desarrolla la conversación y cómo llega el audio**, en lugar de depender solamente de reconocer una voz. Para esto utiliza información de timing, características de voz, propiedades del canal de audio y, cuando realmente es necesario, una transcripción.

En las 71 llamadas de validación disponibles actualmente, el modelo obtuvo **100% de balanced accuracy y 1.000 de AUC**.

> Este resultado significa 71 de 71 llamadas clasificadas correctamente. No significa que el modelo sea perfecto en cualquier llamada o dataset.

---

## ¿Cómo funciona?

El audio es estéreo y contiene dos canales:

* **Canal 0:** caller, que es la persona que queremos clasificar.
* **Canal 1:** agente.

El sistema primero detecta **cuándo habla cada lado** usando los archivos de ```\turns```. Para hacerlo calcula el nivel de ruido de cada canal y detecta como voz las partes que están aproximadamente 12 dB por encima de ese nivel. También une pausas muy pequeñas y elimina fragmentos demasiado cortos.

Esto produce los turnos de la conversación, por ejemplo:

```text
Caller: 12.4s → 15.1s
Agent:  15.3s → 18.0s
Caller: 20.1s → 22.4s
```

El audio se procesa una sola vez y de ahí se obtienen todas las características.

---

## Características del modelo

El modelo utiliza **51 características de timing y 58 características de voz y canal**.

### Timing

Estas características describen **cómo ocurre la conversación**:

* Duración y cantidad de turnos.
* Promedio, mediana y variación de las intervenciones.
* Tiempo que tarda el caller en responder.
* Qué tan variable es ese tiempo.
* Respuestas rápidas y lentas.
* Tiempo que tarda el agente en responder.
* Pausas dentro de una intervención.
* Interrupciones y solapamientos.
* Porcentaje de la llamada que habla cada lado.

Una de las características más importantes es la **latencia de respuesta**: cuánto tarda el caller en comenzar a hablar después de que termina el agente.

En los datos disponibles, las llamadas de IA suelen tardar más porque deben pasar por procesos como:

```text
Audio
 ↓
Speech-to-Text
 ↓
Modelo de lenguaje
 ↓
Text-to-Speech
 ↓
Respuesta
```

Esto hace que el comportamiento temporal sea una señal muy útil.

### Voz y canal

También se analiza cómo suena la llamada:

* Piso de ruido.
* Nivel de voz.
* Relación señal/ruido.
* Silencios.
* Fugas del audio del agente hacia el caller.
* Pitch y su variación.
* Jitter.
* Shimmer.
* Variabilidad de volumen.
* Energía por frecuencia.
* Centroide y ancho de banda espectral.
* Rolloff.
* Planitud espectral.
* Cruces por cero.
* Flujo espectral.
* Variación de MFCC.
* Ritmo de habla.

Una característica especialmente útil es la distribución de energía en frecuencias altas, incluyendo la zona alrededor de **3.4 kHz**. El modelo no usa esto como una regla única, sino como una de muchas señales.

Los MFCC se utilizan principalmente para medir **cómo cambia el timbre**, en lugar de memorizar el timbre promedio de una persona. Esto ayuda a que el modelo sea más útil cuando aparecen voces que no estaban en el entrenamiento.

---

## Arquitectura del modelo

El sistema genera dos opiniones independientes:

```text
Características de timing
          ↓
    Modelo de timing

Características de voz/canal
          ↓
     Modelo de voz
          ↓
     Combinador
          ↓
     Resultado final
```

Cada modelo principal combina:

* **XGBoost pequeño**, para detectar relaciones y umbrales.
* **Regresión logística**, para producir una decisión más suave y generalizable.

Las clases se balancean durante el entrenamiento porque el conjunto de datos no tiene exactamente la misma cantidad de llamadas humanas y sintéticas.

Después, un **combinador de regresión logística** junta las dos opiniones. Este combinador se entrena utilizando predicciones *out-of-fold*, para evitar aprender directamente de resultados que los modelos ya memorizaron.

También se aplica una calibración de confianza que solo puede hacer la probabilidad más conservadora.

---

## Tercera opinión: Speech-to-Text

La mayoría de las llamadas se pueden clasificar solamente utilizando el audio.

Cuando el resultado está cerca de 50/50, el sistema puede pedir una tercera opinión utilizando **ElevenLabs Scribe v2**.

La transcripción busca patrones como:

* Muletillas y dudas.
* Palabras cortadas.
* Formas naturales de responder.
* Repeticiones exactas de información proporcionada por el agente.

Esta información **no reemplaza al modelo acústico**. Solamente modifica su resultado con un peso menor.

Además, la transcripción tiene un tiempo máximo de **12 segundos** para mantener controlado el tiempo total de la petición.

---

## ¿Por qué puede funcionar tan bien?

Las llamadas humanas y sintéticas pueden presentar diferencias no solo en la voz, sino también en el **proceso que genera la llamada**.

Una IA que responde por voz normalmente tiene que:

```text
escuchar → transcribir → generar respuesta → sintetizar voz
```

Ese proceso puede producir patrones diferentes en:

* Tiempo de respuesta.
* Regularidad de la conversación.
* Variación del pitch.
* Variación del volumen.
* Espectro del audio.
* Forma en que llega la señal al canal.

El modelo combina todas estas pistas en lugar de tomar una decisión basándose en una sola característica.

En pruebas internas, incluso eliminando las dos pistas más fuertes —latencia de respuesta y bandas espectrales— el modelo mantiene aproximadamente **97% de rendimiento en validación cruzada sobre `train`**. Esto indica que existen varias señales útiles y no una sola regla determinante.

---

## Datos y validación

El entrenamiento utiliza únicamente las llamadas marcadas como `train`.

Actualmente hay aproximadamente:

```text
282 llamadas de entrenamiento
169 IA
113 humanas
```

Las llamadas de `val` se utilizan para medir el rendimiento del modelo.

Los speakers de `train` y `val` no son los mismos, por lo que el modelo no depende únicamente de memorizar voces.

Los archivos de `turns/` sirven como referencia para desarrollar y comprobar el detector de voz, pero **no se utilizan para clasificar directamente una llamada nueva**.

El resultado actual en `val` es:

| Modelo    | Balanced Accuracy |       AUC |
| --------- | ----------------: | --------: |
| Timing    |             94.5% |     0.990 |
| Voz       |              100% |     1.000 |
| Combinado |          **100%** | **1.000** |

El resultado de 100% debe interpretarse como **71 de 71 llamadas correctamente clasificadas**, no como una garantía de 100% para cualquier conjunto futuro.

---

## API

La API expone dos endpoints:

```text
POST /detect
GET  /health
```

`POST /detect` recibe el audio en Base64 y devuelve:

```json
{
  "is_synthetic": true,
  "confidence": 0.87
}
```

`is_synthetic` indica si el modelo considera que el caller es una IA.

`confidence` representa la confianza de la decisión.

La API también:

* Acepta diferentes formatos de canal y frecuencia.
* Normaliza el audio antes de analizarlo.
* Carga los modelos al iniciar el servidor.
* Hace una inferencia de calentamiento para reducir la latencia inicial.
* Procesa el audio directamente en memoria.
* Mantiene un presupuesto de tiempo para cada petición.
* Devuelve una respuesta válida incluso cuando ocurre un error de procesamiento.

---

## Cómo ejecutar

```bash
python -m pip install -r requirements.txt

python -m src.train

python -m src.evaluate

uvicorn api.app:app
```

`train` entrena los modelos utilizando `train`.

`evaluate` mide el rendimiento utilizando `val`.

---

## Estructura principal

| Archivo           | Función                                        |
| ----------------- | ---------------------------------------------- |
| `src/features.py` | Procesamiento del audio, VAD y características |
| `src/timing.py`   | Modelo de timing                               |
| `src/voice.py`    | Modelo de voz y canal                          |
| `src/stt.py`      | Speech-to-Text                                 |
| `src/detect.py`   | Flujo completo de detección                    |
| `src/models.py`   | Carga, combinación y calibración de modelos    |
| `src/train.py`    | Entrenamiento                                  |
| `src/evaluate.py` | Evaluación                                     |
| `api/app.py`      | API                                            |
| `models/`         | Modelos entrenados                             |
| `audio/`          | Llamadas grabadas                              |
| `turns/`          | Turnos de referencia                           |
| `manifest.csv`    | Información y división de las llamadas         |

## Resumen

El detector no intenta simplemente reconocer una “voz de IA”. Analiza **el comportamiento de la conversación, las características de la voz y la forma en que llega el audio**.

La combinación de estas señales permite distinguir entre llamadas humanas y sintéticas con un rendimiento muy alto en el conjunto de datos disponible.
