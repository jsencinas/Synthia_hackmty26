# Synthia AI Detector

> **Detección de llamadas humanas vs. sintéticas en telefonía bancaria**
> Proyecto desarrollado para **HackMTY 2026 — Reto Altur**.

Memory Leak AI Detector recibe una llamada telefónica grabada y determina si la persona que llama es **humana o una IA**.

En lugar de intentar identificar únicamente si una voz “suena a IA”, el sistema analiza **cómo se desarrolla la conversación, cómo se comporta la voz y cómo llega el audio por el canal telefónico**. Cuando estas señales no son suficientes, utiliza una tercera opinión mediante Speech-to-Text.

En las **71 llamadas del conjunto de validación**, el sistema obtuvo:

- **100% Balanced Accuracy**
- **1.000 ROC-AUC**
- **0.0092 Brier Score**
- **71/71 llamadas clasificadas correctamente**

Este resultado corresponde únicamente al conjunto de validación disponible. **No significa que el modelo sea perfecto en cualquier llamada o dataset futuro.**

---



## ¿Cómo funciona?

El sistema sigue un proceso escalonado:

```mermaid
flowchart TD
    WAV["Audio Estéreo WAV"] --> VAD["VAD Adaptativo"]
    
    VAD --> Time["Timing & Turn-taking<br/>51 features"]
    VAD --> Voice["Voz, Acústica & Canal<br/>58 features"]
    
    Time --> TimeHead["Timing Head<br/>XGBoost + LogReg"]
    Voice --> VoiceHead["Voice Head<br/>XGBoost + LogReg"]
    
    TimeHead --> Stack["Stacker Logístico"]
    VoiceHead --> Stack
    
    Stack --> Calib["Calibración de Temperatura"]
    Calib --> Gate{"¿Resultado incierto?"}
    
    Gate -- No --> Final["Resultado Final"]
    Gate -- Sí --> Scribe["ElevenLabs Scribe v2"]
    
    Scribe --> Heuristics["Análisis Lingüístico"]
    Heuristics --> Blend["Fusión ponderada"]
    Blend --> Final
```



La idea principal es utilizar varias señales independientes en lugar de depender de una sola característica.

# 1. El audio y la detección de turnos

Las llamadas son audio estéreo y normalmente contienen dos canales:

```text
Canal 0 → Caller
Canal 1 → Agente
```

Durante el desarrollo utilizamos los archivos de `turns/` como referencia para comprobar que la detección de voz fuera correcta.

En producción, el sistema **no necesita turnos precalculados**. Los obtiene directamente del WAV utilizando un VAD adaptativo.

### VAD adaptativo

Para cada canal:

1. El audio se divide en ventanas de aproximadamente **20 ms**.
2. Se calcula el nivel RMS.
3. Se estima el piso de ruido mediante el **percentil 20**.
4. Se considera voz una señal aproximadamente **12 dB por encima del ruido**.
5. Se eliminan fragmentos demasiado cortos.
6. Se extienden los segmentos aproximadamente 100 ms para evitar cortar sonidos suaves.
7. Se unen pausas pequeñas de menos de aproximadamente 200 ms.

Esto genera los turnos necesarios para analizar la conversación:

```text
Caller: 12.4s → 15.1s
Agent:  15.3s → 18.0s
Caller: 20.1s → 22.4s
```

Durante la validación del VAD, se obtuvo aproximadamente:

- **0.95 IoU para caller**
- **0.97 IoU para agente**

comparando los segmentos detectados con las anotaciones de referencia.

Una vez obtenido el VAD, **el audio se procesa una sola vez** y se reutiliza para extraer todas las características.

---



# 2. Características del modelo

El modelo utiliza **109 características en total**:

```text
51 características de Timing
+
58 características de Voz y Canal
=
109 características
```

Estas características se dividen en dos grupos independientes.

---



## 2.1 Timing e interacción

Las características de timing intentan describir **cómo ocurre la conversación**, no cómo suena una persona.

Entre ellas se encuentran:

- Duración de los turnos.
- Número de turnos.
- Promedio, mediana y variación de las intervenciones.
- Tiempo de respuesta del caller.
- Variabilidad del tiempo de respuesta.
- Respuestas extremadamente rápidas o lentas.
- Tiempo de respuesta del agente.
- Pausas internas.
- Interrupciones.
- Solapamientos.
- Distribución temporal de la conversación.
- Porcentaje de tiempo hablado por cada lado.



### La latencia de respuesta

Una de las señales más importantes es:

> **¿Cuánto tarda el caller en comenzar a responder después de que termina el agente?**

Una IA de voz normalmente necesita pasar por varias etapas:

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

Cada una puede introducir una pequeña cantidad de latencia.

En los datos disponibles, las llamadas sintéticas presentan patrones temporales diferentes a los humanos.

![Latencia de respuesta](assets/01_kde_latencia_respuesta.png)

*La distribución muestra cómo los tiempos de respuesta de humanos y sistemas sintéticos presentan comportamientos diferentes. Los humanos tienden a concentrarse en respuestas más rápidas y variables, mientras que las llamadas sintéticas presentan una distribución desplazada.*

---



## 2.2 Voz, acústica y canal

La segunda cabeza analiza **cómo llega y cambia físicamente el audio**.

Se utilizan características relacionadas con:

- Piso de ruido.
- Energía de la señal.
- Relación señal/ruido.
- Silencios.
- Fugas entre canales.
- Pitch.
- Variación del pitch.
- Jitter.
- Shimmer.
- Variación del volumen.
- Energía por frecuencia.
- Centroide espectral.
- Ancho de banda.
- Rolloff.
- Planitud espectral.
- Cruces por cero.
- Flujo espectral.
- Variación de MFCC.
- Ritmo de habla.



### Características del canal

Una señal interesante es el **cross-talk**.

En una llamada telefónica física, parte de la señal del agente puede filtrarse ligeramente hacia el canal del caller. Cuando una llamada es generada completamente por software, este comportamiento puede ser diferente.

También analizamos los **silencios digitales exactos**, donde existen segmentos con valores numéricos perfectamente iguales a cero.

Estas señales no se utilizan como reglas individuales. Se combinan con el resto de características para que el modelo no dependa de una sola pista.

### Frecuencias altas

También analizamos la energía alrededor de la zona de **3.4 kHz** y otras propiedades espectrales.

La telefonía tradicional limita gran parte de la información de frecuencias altas, mientras que determinados sistemas sintéticos pueden generar patrones distintos en estas regiones.

De nuevo, esto es solamente una característica entre muchas otras.

### MFCC

Los MFCC se utilizan principalmente para analizar **cómo cambia el timbre**, no para memorizar el timbre promedio de un individuo.

En lugar de depender únicamente del valor medio, se analiza su variabilidad. Esto ayuda a reducir la dependencia del modelo respecto a voces específicas del entrenamiento.

![Características acústicas](assets/02_boxplots_caracteristicas_acusticas.png)

*Las distribuciones muestran diferencias entre llamadas humanas y sintéticas en diferentes propiedades acústicas y del canal.*

---



# 3. Arquitectura de los modelos

El sistema genera dos opiniones independientes:

```text
             ┌─────────────────────┐
             │ Timing features     │
             └──────────┬──────────┘
                        ↓
                 Timing Model
                        │
                        │ P_timing
                        │
                        ├─────────────┐
                                      ↓
                                Stacker
                                      ↑
                        ┌─────────────┘
                        │ P_voice
                 Voice Model
                        ↑
             ┌──────────┴──────────┐
             │ Voice / Channel     │
             │ features            │
             └─────────────────────┘
```

Cada cabeza combina dos modelos complementarios:

### XGBoost

Utilizamos un XGBoost pequeño y regularizado para detectar:

- Relaciones no lineales.
- Umbrales.
- Interacciones entre características.

Se mantienen árboles pequeños y regularización para reducir el sobreajuste.

### Regresión logística

La regresión logística proporciona una decisión más suave y estable.

Esto es útil para evitar depender completamente de fronteras complejas y para generalizar mejor ante valores que no aparecieron exactamente durante el entrenamiento.

### Soft voting

Las dos predicciones se combinan para producir la probabilidad de cada cabeza.

---



# 4. El Stacker

Después de obtener las predicciones independientes, un tercer modelo las combina.

El **Stacker Logístico** recibe las probabilidades de:

```text
Timing Head
     +
Voice Head
     ↓
Stacker
```

Antes de combinarlas se proyectan a espacio logit:

$$
z = \ln\left(\frac{p}{1-p}\right)
$$

El stacker se entrena utilizando **predicciones Out-of-Fold de 5 folds**.

Esto es importante porque evita entrenar el combinador utilizando predicciones que fueron producidas sobre los mismos ejemplos que los modelos utilizaron para aprender.

En otras palabras, el sistema intenta evitar que el stacker aprenda una confianza artificialmente optimista.

![Stacker](assets/03_logits_stacker.png)

*Visualización de las predicciones en espacio logit y cómo las diferentes cabezas contribuyen a la decisión final.*

---



# 5. Calibración de confianza

Después del stacker se aplica una calibración de temperatura.

La calibración permite que una predicción como:

```text
0.99
```

no sea interpretada automáticamente como una certeza absoluta.

La temperatura únicamente puede hacer la confianza **más conservadora**:

$$
T \geq 1
$$

Esto evita que el sistema incremente artificialmente probabilidades extremas.

Además de mejorar la interpretación de la confianza, esta etapa ayuda a obtener un mejor **Brier Score**.

---



# 6. Tercera opinión: Speech-to-Text

La mayoría de las llamadas pueden clasificarse solamente con las señales acústicas.

Por eso, la transcripción **no se utiliza para todas las llamadas**.

Primero se obtiene una decisión acústica.

Si el resultado cae dentro de una zona de incertidumbre:

$$
|p-0.5| \leq 0.12
$$

y todavía existe suficiente tiempo disponible, el sistema puede pedir una tercera opinión.

```text
Audio
 ↓
Timing + Voice
 ↓
Resultado
 ↓
¿Incierto?
 ├── No → Final
 └── Sí
       ↓
 ElevenLabs Scribe v2
       ↓
 Análisis lingüístico
       ↓
 Ajuste de probabilidad
       ↓
 Final
```

La transcripción se realiza utilizando **ElevenLabs Scribe v2**.

El sistema busca señales como:

### Patrones humanos

Ejemplos:

```text
"mande"
"cómo?"
"perdón"
"a ver"
"este..."
```

También se consideran palabras cortadas, dudas y otras formas de respuesta espontánea.

### Patrones sintéticos

Se presta atención, entre otras cosas, a repeticiones demasiado exactas o inmediatas de información proporcionada por el agente.

### Peso limitado

La información lingüística **no reemplaza la predicción acústica**.

Su contribución tiene un peso menor:

# $$
\text{logit final}

\text{logit acústico}
+
0.5 \times \text{logit STT}
$$

Además, la transcripción tiene un presupuesto máximo de aproximadamente **12 segundos** para mantener controlada la latencia total.

---



# 7. ¿Por qué funciona?

La idea principal del proyecto es que una llamada sintética puede diferenciarse de una humana en más de una dimensión.

Una IA de voz normalmente tiene que:

```text
Escuchar
   ↓
Transcribir
   ↓
Generar respuesta
   ↓
Sintetizar voz
   ↓
Hablar
```

Ese proceso puede producir diferencias en:

- Tiempo de respuesta.
- Regularidad de los turnos.
- Variación del pitch.
- Variación del volumen.
- Espectro.
- Ruido.
- Propiedades del canal.
- Forma de las pausas.
- Forma en que responde a información del agente.

El modelo intenta combinar todas estas señales.

En pruebas internas, incluso eliminando dos de las señales más fuertes —**latencia de respuesta y bandas espectrales**— el modelo mantiene aproximadamente **97% de rendimiento en validación cruzada sobre** `train`.

Esto sugiere que el modelo no depende completamente de una sola regla.

---



# 8. Datos y validación

El entrenamiento utiliza únicamente las llamadas del split `train`.

Actualmente el conjunto de entrenamiento contiene aproximadamente:

```text
282 llamadas
├── 169 sintéticas
└── 113 humanas
```

El conjunto `val` se mantiene separado para evaluar el sistema.

Además, los speakers de `train` y `val` son diferentes:

```text
TRAIN
  ↓
Aprendizaje

VAL
  ↓
Evaluación
```

Esto reduce el riesgo de que el modelo simplemente memorice determinadas voces.

Los archivos dentro de `turns/` se utilizan como referencia para desarrollar y comprobar el detector de voz, pero **no son necesarios para clasificar una llamada nueva en producción**.

---



# 9. Resultados

En las **71 llamadas de validación** disponibles:


| Modelo            | Balanced Accuracy | ROC-AUC    | Brier Score |
| ----------------- | ----------------- | ---------- | ----------- |
| Timing Head       | 94.5%             | 0.9902     | 0.0541      |
| Voice Head        | 100.0%            | 1.0000     | 0.0094      |
| Fused Model       | **100.0%**        | **1.0000** | **0.0092**  |
| Pipeline completo | **100.0%**        | **1.0000** | **0.0092**  |


Esto representa:

```text
71 llamadas
71 correctas
0 incorrectas
```

El stacker tuvo que resolver discrepancias entre las dos cabezas en aproximadamente **2.8% de las llamadas**, equivalentes a 2 de 71.

En este conjunto de validación, ninguna llamada terminó necesitando el desempate lingüístico porque las señales acústicas fueron suficientemente claras.

**Importante:** 100% en estas 71 llamadas no significa 100% de precisión garantizada para llamadas nuevas.

![Matriz de confusión y ROC](assets/04_matriz_confusion_y_roc.png)

*Resultados del modelo sobre el conjunt*