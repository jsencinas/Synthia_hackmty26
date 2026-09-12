# 📖 Explicación Completa de `train_turn_model.py`

Este documento explica de forma sencilla y detallada el funcionamiento del script [`train_turn_model.py`](train_turn_model.py), diseñado para **detectar si quien llama por teléfono a un centro de atención es una persona real o un agente de Inteligencia Artificial (voz sintética)**, analizando únicamente la dinámica de los turnos de la conversación.

---

## 🎯 1. ¿Cuál es el objetivo de este código?

Imagina que un banco recibe cientos de llamadas al día. En algunas, un cliente humano habla con el asistente virtual del banco. En otras, quien llama es un **robot de IA** (un modelo de lenguaje conectado a un sintetizador de voz) simulando ser cliente.

El objetivo de este script es:
> **Aprender patrones matemáticos de cómo conversan los humanos vs. las IAs**, entrenar varios modelos de Machine Learning (aprendizaje automático) y guardar el mejor de ellos para predecir futuras llamadas.

### 💡 La idea clave: No escuchamos la voz, medimos el ritmo
A diferencia de los modelos acústicos (que analizan el timbre o tono de voz), este código **no necesita escuchar el audio**. Solo analiza **el ritmo y los turnos de la conversación**:
- ¿Cuánto tarda en contestar después de que el agente del banco termina de hablar? (Latencia).
- ¿Interrumpe frecuentemente al agente o espera su turno?
- ¿Habla de corrido o hace pausas naturales para respirar y pensar?
- ¿Cuánto duran sus intervenciones?

Las IAs suelen responder con latencias muy constantes o interrumpir en momentos inusuales, mientras que los humanos dudan, hacen pausas irregulares y reaccionan de manera distinta.

---

## 📊 2. Las Variables (Features): ¿Qué datos analiza el modelo?

El archivo de entrada contiene métricas calculadas a partir de los turnos de cada llamada.

### A. Variables base extraídas de la llamada:
1. `numero_turnos_caller`: Cuántas veces tomó la palabra quien llama.
2. `duracion_promedio_caller`: Cuántos segundos duró en promedio cada intervención de quien llama.
3. `pausa_promedio_caller`: Tiempo en silencio entre una intervención y la siguiente de la misma persona.
4. `desviacion_estandar_latencia`: Qué tan variable es el tiempo que tarda en empezar a hablar tras terminar el agente. *(Un humano suele tener variación alta: a veces responde rápido y a veces piensa; un bot suele tener tiempos de respuesta más robóticos y predecibles).*
5. `interrupciones_agente`: Número de veces que quien llama habló al mismo tiempo que el agente del banco.
6. `duracion_total_interrupciones`: Segundos acumulados en los que ambas partes hablaron al mismo tiempo.
7. `duracion_promedio_interrupcion`: Duración media de cada interrupción.

### B. Ingeniería de Características (`agregar_features_ingenieria`):
El código crea 3 variables adicionales combinando las anteriores para darle más pistas al modelo:
1. `tasa_interrupciones`: Interrupciones totales divididas por el número de turnos. *(¿Interrumpe en casi todos sus turnos o solo esporádicamente?)*.
2. `duracion_interrupciones_por_turno`: Cuánto tiempo de choque de voces hay distribuido por turno.
3. `ratio_habla_pausa`: Duración del habla entre duración de las pausas. *(¿Pasa más tiempo hablando o en silencio?)*.

---

## 🛡️ 3. División de Datos sin Trampa (`cargar_datos`)

Uno de los problemas más comunes en Machine Learning es el **autoengaño (data leakage u overfitting)**: si pruebas un modelo con las mismas preguntas con las que estudió, parecerá infalible, pero fallará con llamadas nuevas.

Para evitar esto, el código divide los datos en **tres conjuntos estrictamente separados**:

```
Dataset Total
 ├── Train (Llamadas de entrenamiento)
 │    ├── train_fit (~252 llamadas) ──► Usado para ENTRENAR y comparar modelos (5-Fold CV).
 │    └── test_holdout (30 llamadas) ──► GUARDADO BAJO LLAVE. No se toca hasta el final.
 └── Val (71 llamadas) ────────────────► Usado EXCLUSIVAMENTE para calibrar el umbral óptimo.
```

1. **`train_fit`**: El conjunto de estudio donde los modelos aprenden los patrones.
2. **`val` (Validación)**: Se usa únicamente para afinar el punto de corte (umbral de probabilidad).
3. **`test_holdout` (Prueba ciega de 30 llamadas)**: Llamadas que ningún modelo vio durante el entrenamiento ni durante la calibración. Es la prueba de fuego real.

---

## 🤖 4. El Catálogo de Modelos (`construir_modelos`)

El script no se confía de un solo algoritmo; crea un torneo entre **7 modelos diferentes**:

| Modelo | ¿Cómo funciona en palabras simples? |
|---|---|
| **LogisticRegression** | Traza una frontera lineal simple. Usa un `StandardScaler` para normalizar los números. Es el modelo base de referencia. |
| **RandomForest** | Un "bosque" de 150 árboles de decisión que votan en equipo. Muy estable y resistente al ruido. |
| **ExtraTrees** | Similar al Random Forest, pero elige los puntos de corte de forma aleatoria, lo que a veces reduce el sobreajuste. |
| **GradientBoosting** | Árboles entrenados secuencialmente, donde cada árbol nuevo intenta corregir los errores cometidos por los anteriores. |
| **LightGBM** | Implementación ultra-rápida y eficiente de Gradient Boosting desarrollada por Microsoft. |
| **XGBoost** | Uno de los algoritmos más potentes y usados en competencias de ciencia de datos para datos tabulares. |
| **VotingEnsemble** | Un "comité de sabios": combina las probabilidades de RandomForest, XGBoost, GradientBoosting y LogisticRegression mediante votación suave (`soft voting`). |

---

## 🔬 5. Comparación y Selección de Modelos (Validación Cruzada 5-Fold)

Para saber cuál modelo es el mejor **sin hacer trampa**:
1. El script toma `train_fit` y lo divide en 5 partes (folds).
2. Entrena con 4 partes y prueba con la 1 restante, repitiendo el proceso 5 veces (`cross_val_score`).
3. Evalúa con la métrica **ROC-AUC** (Área bajo la curva ROC):
   - Mide qué tan bien el modelo distingue entre humanos (0) y sintéticos (1).
   - Un valor de `0.50` es como lanzar una moneda al aire.
   - Un valor cercano a `1.00` representa una separación perfecta.
4. **Criterio de elección:** El modelo con el mayor `CV Train AUC` promedio es coronado como el **Mejor Modelo**.

---

## ⚖️ 6. Calibración del Umbral (`calibrar_umbral`)

Por defecto, los clasificadores dicen: *"Si la probabilidad es ≥ 0.50 (50%), es sintético; si no, es humano"*.

Sin embargo, el 50% no siempre es el punto óptimo. Por ejemplo, si los datos están desbalanceados o queremos maximizar tanto el **Accuracy** como el **F1-Score**:
- La función prueba umbrales desde `0.30` hasta `0.70` en pasos de `0.02` usando el conjunto `val`.
- Encuentra el umbral que logra el mejor balance entre aciertos globales y detección precisa de llamadas sintéticas.

---

## 🏁 7. Evaluación Final sobre `test_holdout`

Una vez que el modelo fue elegido y el umbral fue calibrado, se abre el conjunto reservado de 30 llamadas (`test_holdout`) para medir el rendimiento real:

1. **Reporte con umbral estándar (0.50)** y **con umbral calibrado**.
2. **Métricas reportadas:**
   - **Accuracy**: Porcentaje de llamadas clasificadas correctamente.
   - **Precision**: De las que el modelo dijo que eran sintéticas, ¿cuántas lo eran de verdad?
   - **Recall**: De todas las llamadas sintéticas que existían, ¿cuántas logró atrapar?
   - **F1-Score**: La media armónica entre Precision y Recall.
3. **Matriz de Confusión**:
   - Muestra cuántos humanos se predijeron como humanos y cuántos como sintéticos, y lo mismo para las IAs.

---

## 💾 8. Exportación de Artefactos

Al finalizar, el código guarda dos archivos esenciales en el directorio raíz:

1. **`turn_model.joblib`**:
   - El archivo binario con el modelo entrenado y listo para ponerse en producción (hacer inferencias en milisegundos).
2. **`metadata_turn_model.json`**:
   - Ficha técnica en JSON que contiene:
     - Nombre del modelo ganador.
     - Lista exacta de variables que espera recibir.
     - Umbral recomendado a aplicar en producción.
     - Métricas reales obtenidas en validación y en el test holdout.
     - Importancia relativa de cada variable (cuál variable pesó más en las decisiones).

---

## 🚀 9. ¿Cómo se ejecuta?

Asegúrate de estar en el entorno virtual del proyecto con sus dependencias instaladas:

```bash
python train_turn_model.py
```

Al ejecutarse, verás en la terminal:
1. El conteo de muestras de cada partición (`train_fit`, `val`, `test_holdout`).
2. La tabla comparativa de los 7 modelos y sus puntajes de validación cruzada.
3. La selección del modelo ganador.
4. El umbral óptimo encontrado.
5. El reporte final de métricas en las llamadas nunca antes vistas.
6. El ranking de qué variables fueron las más determinantes para detectar las llamadas sintéticas.
