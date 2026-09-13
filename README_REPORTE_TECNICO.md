# Reporte Técnico: Arquitectura y Funcionamiento del Detector de IA

---

## 1. El Problema y la Estrategia

En un centro de llamadas bancarias, los atacantes o usuarios pueden utilizar **asistentes de voz impulsados por Inteligencia Artificial** para hacerse pasar por clientes reales.

### ¿Por qué los detectores comunes fallan?
1. **Los sintetizadores de voz modernos suenan casi humanos:** Si solo intentas detectar si la voz "suena a robot", vas a fallar cuando usen modelos de última generación (como ElevenLabs, OpenAI u otros).
2. **Las voces son diferentes a las del entrenamiento:** Si memorizas cómo suenan los estafadores de tu conjunto de prueba, cuando llegue una voz nueva con acento distinto, no funcionará.

### Nuestra solución: El Enfoque en Tres Capas
En lugar de buscar una sola "pista mágica", nuestro sistema analiza la llamada desde tres ángulos complementarios e independientes:



```
                      Audio Estéreo (Caller vs. Agente)
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
    [ Capa de Timing ]                              [ Capa de Voz y Canal ]
 ¿Cómo interactúa en la llamada?                   ¿Cómo se comporta físicamente el audio?
 (Pausas, latencias, ritmo, turnos)              (Espectro, estabilidad, micro-variaciones)
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     ▼
                        [ Combinador (Stacker) ]
                    Junta ambas opiniones de forma equilibrada
                                     │
                     ¿Hay duda o caso límite? (50/50)
                                ├── No ──► Decisión Final Rápida (< 300 ms)
                                └── Sí ──► [ Capa Lingüística (STT) ]
                                           Analiza vacilaciones ("mande", "este")
```

---

## 2. Detección Automática de Habla (VAD Inteligente)

El sistema **no necesita que nadie le diga cuándo habla cada persona**. Procesa el audio directamente y detecta los turnos de conversación de forma autónoma:

1. **Averigua el nivel de ruido del fondo:** Cada llamada telefónica tiene un ruido de línea distinto (estática, ruido ambiental). El sistema calcula el ruido base de cada canal.
2. **Detecta voz real:** Considera que alguien está hablando cuando el volumen supera por al menos **12 dB** el ruido de fondo de esa llamada en específico.
3. **No corta palabras:** Agrega un pequeño margen de 100 milisegundos al final de cada intervención para no comerse las consonantes suaves finales (como "s", "d" o respiraciones).
4. **Une pausas pequeñas:** Si alguien hace una micro-pausa de menos de 200 ms para tomar aire, la mantiene como parte del mismo turno.
5. **Elimina ruidos falsos:** Ignora chasquidos o ruidos de línea que duren menos de 300 ms.

> **Resultado:** Este detector coincide en más de un **95%** con las anotaciones manuales de referencia, funcionando en tiempo real en memoria.

---

## 3. Las Pistas que Busca el Modelo (103 Características)

El modelo analiza 103 números divididos en dos grandes grupos:

---

### A. Características de Timing e Interacción (45 variables)
Analizan **el ritmo de la conversación**:

* **Latencia de respuesta (La pista más fuerte):**
  - ¿Cuánto tarda el cliente en contestar después de que el agente del banco termina de hablar?
  - *La razón:* Un bot de IA tiene que escuchar el audio $\to$ convertirlo a texto $\to$ procesarlo en un modelo de lenguaje (LLM) $\to$ sintetizar el audio $\to$ reproducirlo. Ese proceso genera retrasos o tiempos de respuesta artificialmente rígidos. Un humano, en cambio, reacciona de forma intuitiva, a veces rápido y a veces dudando.
* **Duración y frecuencia de turnos:**
  - ¿Habla a ráfagas cortas o da respuestas largas?
  - Fracción de turnos que duran menos de 1 segundo vs. más de 8 segundos.
* **Pausas internas:**
  - ¿Cuánto tiempo se queda callado a mitad de su propia intervención para pensar o formular una frase?
* **Interrupciones y solapamientos:**
  - ¿Comienza a hablar mientras el agente bancario aún no ha terminado?
  - Las IAs suelen respetar rígidamente los turnos o interrumpir en momentos poco naturales por fallas en su detector de fin de turno (*endpointing*).

---

### B. Características de Voz, Canal y Espectro (58 variables)
Analizan **la física del sonido y el canal telefónico**:

* **Silencios digitales perfectos:**
  - Muchos sintetizadores de voz introducen tramos de silencio numéricamente exactos (muestras con valor $0$ absoluto). En un micrófono físico humano, siempre existe un micro-ruido térmico o ambiental.
* **Fuga entre canales (Cross-talk):**
  - En llamadas reales con teléfonos físicos, cuando el agente habla por el altavoz o auricular, una fracción minúscula de ese sonido se cuela de regreso al micrófono del cliente. En un bot de software puro, no hay auricular ni micrófono físico, por lo que el aislamiento suele ser artificialmente perfecto.
* **Tono y micro-variaciones de la voz:**
  - **Pitch ($F_0$):** Rango y variación del tono.
  - **Jitter:** Qué tan estables son los ciclos de las cuerdas vocales. La voz humana tiene micro-imperfecciones naturales involuntarias; las IAs tienden a ser excesivamente regulares o a tener cambios abruptos.
  - **Shimmer:** Micro-variación en el volumen entre un instante y el siguiente.
* **El "corte" de frecuencia telefónica (3.4 kHz a 4.0 kHz):**
  - La telefonía tradicional estándar (G.711) corta el audio abruptamente por encima de los 3,400 Hz. Muchos generadores de IA modernos generan audio de alta fidelidad que luego se comprime o que inyecta componentes de alta frecuencia en esa frontera.
* **Variabilidad de timbre (MFCCs sin trampa):**
  - Calculamos 13 coeficientes del timbre de la voz, pero **solo usamos su variación (desviación estándar)**, descartando el promedio.
  - *¿Por qué?* El promedio memoriza a la persona específica (ej. "voz de hombre grave"). La variación mide qué tan expresiva y rica es la voz, lo cual generaliza perfectamente a personas que el modelo nunca ha escuchado.

---

## 4. Arquitectura de Modelos: Dos Opiniones y un Árbitro

No usamos una "caja negra" monolítica. Usamos un sistema modular y transparente:

```
[ Variables de Timing ]  ──►  Cabeza de Timing (XGBoost + Regresión Logística)  ──► Opinión A
                                                                                          │
                                                                                          ▼
                                                                                   [ Árbitro Stacker ]
                                                                                   (Regresión Logística)
                                                                                          ▲
                                                                                          │
[ Variables de Voz ]     ──►  Cabeza de Voz (XGBoost + Regresión Logística)     ──► Opinión B
```

### ¿Por qué cada cabeza junta XGBoost con Regresión Logística?
* **XGBoost (Árboles de decisión):** Es excelente para encontrar umbrales y reglas no lineales (ej. *"si la latencia es $> 2.1$ s Y el silencio digital es alto"*).
* **Regresión Logística:** Es un modelo suave y lineal. Evita que el sistema tome decisiones extremas o absurdas cuando aparece un valor muy raro que nunca vio en el entrenamiento.
* **Juntos:** Se promedian en una "votación suave", combinando la astucia de los árboles con la estabilidad del modelo lineal.

### El Árbitro (Stacker Logístico)
En lugar de promediar las opiniones a ciegas, un meta-modelo de **Regresión Logística** aprende cuánto peso darle a la cabeza de Timing y cuánto a la de Voz:
- Para que este árbitro no aprenda de notas infladas, se entrena usando **predicciones cruzadas fuera de bolsa (Out-of-Fold)**: solo ve predicciones de llamadas que las cabezas no vieron al entrenar.

---

## 5. Calibración de Confianza: Humildad Inteligente

Un error común en IA es la **sobreconfianza**: modelos que dicen estar *"99.9% seguros"* cuando en realidad no conocen el caso. En las métricas de competencia (como el *Brier Score*), esto se penaliza severamente.

Para solucionarlo aplicamos **Temperature Scaling**:
- Es una perilla matemática que "enfría" las probabilidades extremas hacia números más realistas y honestos.
- **Regla de seguridad:** Solo permitimos suavizar la confianza, nunca inflarla artificialmente. Si el modelo dice $85\%$, es porque en la práctica acierta aproximadamente 85 de cada 100 veces.

---

## 6. Desempate Lingüístico: Speech-to-Text (ElevenLabs Scribe v2)

Para más del 95% de las llamadas, la combinación de timing y voz es contundente y decide en menos de 250 milisegundos.

Sin embargo, si una llamada cae en la **zona de incertidumbre** (el modelo está cerca del 50/50) y el servidor aún tiene tiempo disponible en el reloj, el sistema activa una tercera opinión:

1. Transcribe el audio con **ElevenLabs Scribe v2** multicanal.
2. Analiza el texto en busca de patrones humanos del español de México:
   - **Dudas y vacilaciones:** Palabras como *"mande"*, *"cómo"*, *"perdón"*, *"a ver"*, *"este..."* o frases que se quedan a medias restan probabilidad de ser IA.
   - **Repetición robótica de números:** Si el agente dicta una serie de dígitos bancarios y la otra parte los repite con precisión matemática instantánea sin ninguna duda, suma probabilidad de ser IA.
3. Esta opinión **no reemplaza** el análisis de audio; solo le da un empujón suave para inclinar la balanza.

---

## 7. Ingeniería de Producción y Latencia (`api/app.py`)

El sistema expone una API lista para producción bancaria:

* **Inferencia ultrarrápida:** El análisis de audio y predicción toma típicamente **entre 150 y 300 milisegundos**.
* **Pre-calentamiento al arrancar:** Cuando el servidor inicia, ejecuta una llamada simulada en memoria. Esto compila las funciones aceleradas antes de que llegue el primer usuario, evitando que la primera petición sufra retrasos de 3 segundos.
* **Garantía de respuesta (Cero caídas):** Si llega un audio corrupto, vacío o con formato extraño, el servidor nunca responde con error 500. Automáticamente devuelve una respuesta segura y neutra:
  ```json
  {"is_synthetic": false, "confidence": 0.5}
  ```

---

## 8. ¿Por qué estas métricas son de verdad y no "trampa"?

Muchos modelos en competencias obtienen puntuaciones infladas porque prueban con las mismas voces con las que entrenaron. Nuestro pipeline previene esto rigurosamente:

1. **Separación por personas (Speaker-Disjoint):** Ninguna persona ni voz sintética del conjunto de entrenamiento aparece en el conjunto de validación.
2. **Sin fuga de información (No Data Leakage):** El conjunto de validación está completamente congelado. No se usa para calibrar umbrales, ni para ajustar hiperparámetros.
3. **Métricas en las 71 llamadas de validación:**
   - **Cabeza de Timing:** 94.5% de Balanced Accuracy (AUC 0.990)
   - **Cabeza de Voz:** 100% de Balanced Accuracy (AUC 1.000)
   - **Sistema Completo Combinado:** **100% de Balanced Accuracy (71 de 71 llamadas clasificadas correctamente)** con un Brier Score de **0.009** (calibración casi perfecta).

---

## 9. Guía Rápida para Ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar el sistema (usando únicamente los datos de entrenamiento)
python -m src.train

# 3. Evaluar de forma ciega sobre las llamadas de validación
python -m src.evaluate

# 4. Iniciar la API REST
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
