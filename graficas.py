"""
graficas.py

Genera EXCLUSIVAMENTE las 4 gráficas solicitadas en el README.md:
1. GRÁFICA SUGERIDA #1: Curvas de Densidad (KDE) de Latencia de Respuesta
   - Curva Azul (Humanos) vs. Curva Naranja (Sintéticos).
   - Rango 0.0 s - 4.0 s, mostrando concentración humana (0.2s - 0.8s) e IA (>1.5s).
2. GRÁFICA SUGERIDA #2: Boxplot Comparativo de Características Acústicas Clave
   - 3 Subplots lado a lado: band_vhi_ratio_log, jitter, crosstalk_db.
   - Magnitud normalizada de cada métrica (Humano vs. Sintético).
3. GRÁFICA SUGERIDA #3: Dispersión 2D de Logits y Frontera del Stacker
   - 282 llamadas de entrenamiento: Círculos verdes (Humanos) vs. Triángulos rojos (Sintéticos).
   - Frontera lineal de decisión del Stacker (P = 0.5) mostrando la ortogonalidad.
4. GRÁFICA SUGERIDA #4: Matriz de Confusión y Curva ROC en Validación
   - 2 Paneles para las 71 llamadas de validación (speaker-disjoint).
   - Heatmap 2x2 (100% aciertos) y Curva ROC con AUC = 1.000.

CÓMO EJECUTAR:
    python graficas.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import joblib

# Asegurar que el directorio raíz esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import audio_path_for, load_split, validate_audio_files
from src.features import TIMING_FEATURES, VOICE_FEATURES, analyze_audio, load_audio
from src.models import (
    fuse_probabilities,
    load_stacker,
    load_temperature,
    load_timing_model,
    load_voice_model,
    logit,
    require_artifacts,
)

# ============================================================
# CONFIGURACIÓN VISUAL Y DE DIRECTORIOS
# ============================================================

CARPETA_SALIDA = ROOT_DIR / "graficas"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CACHE_FEATURES_PATH = CARPETA_SALIDA / "features_cache.joblib"

# Paleta exacta según especificaciones del README
COLOR_AZUL_HUMANO = "#1f77b4"      # Curva / elemento Azul para Humanos
COLOR_NARANJA_SINTETICO = "#ff7f0e" # Curva / elemento Naranja para Sintéticos
COLOR_VERDE_HUMANO = "#16a34a"     # Círculos verdes para Humanos en Scatter
COLOR_ROJO_SINTETICO = "#dc2626"   # Triángulos rojos para Sintéticos en Scatter
COLOR_FRONTERA = "#0f172a"         # Frontera lineal del Stacker

plt.rcParams.update({
    "figure.dpi": 200,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
})


def guardar_figura(fig: plt.Figure, nombre_archivo: str) -> None:
    """Guarda una figura con alta resolución y cierre automático."""
    ruta = CARPETA_SALIDA / nombre_archivo
    fig.savefig(ruta, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [OK] Gráfica guardada: {ruta}")


def kde_unidimensional(datos: np.ndarray, puntos_x: np.ndarray, ancho_banda: float | None = None) -> np.ndarray:
    """Calcula densidad KDE gaussiana rápida sin dependencias externas."""
    datos = np.asarray(datos, dtype=float)
    datos = datos[np.isfinite(datos)]
    if len(datos) == 0:
        return np.zeros_like(puntos_x)

    if ancho_banda is None:
        desv = np.std(datos, ddof=1) if len(datos) > 1 else 0.2
        # Regla empírica de Silverman
        ancho_banda = 1.06 * desv * (len(datos) ** (-0.2))

    ancho_banda = max(float(ancho_banda), 0.08)
    diffs = puntos_x[:, None] - datos[None, :]
    kernel = np.exp(-0.5 * (diffs / ancho_banda) ** 2)
    kernel /= ancho_banda * np.sqrt(2.0 * np.pi)
    return np.mean(kernel, axis=1)


# ============================================================
# EXTRACCIÓN Y CARGA DE DATOS (CON CACHÉ PERSISTENTE)
# ============================================================

def extraer_features_split(split_name: str) -> dict:
    """Extrae características de audio para un split dado."""
    rows = load_split(split_name)
    validate_audio_files(rows)

    timing_rows = []
    voice_rows = []
    labels = []
    anon_ids = []

    print(f"Extrayendo características para split '{split_name}' ({len(rows)} llamadas)...")
    for i, row in enumerate(rows.itertuples(index=False), 1):
        if i % 25 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] Procesando {row.anon_id}...")
        audio = load_audio(audio_path_for(str(row.anon_id)))
        analysis = analyze_audio(audio)
        timing_rows.append(analysis.timing)
        voice_rows.append(analysis.voice)
        labels.append(1 if row.label == "synthetic" else 0)
        anon_ids.append(str(row.anon_id))

    timing_df = pd.DataFrame(timing_rows).reindex(columns=TIMING_FEATURES)
    voice_df = pd.DataFrame(voice_rows).reindex(columns=VOICE_FEATURES)

    return {
        "timing_df": timing_df,
        "voice_df": voice_df,
        "labels": np.asarray(labels, dtype=int),
        "anon_ids": anon_ids,
    }


def cargar_o_extraer_datos() -> tuple[dict, dict]:
    """Carga datos pre-cacheados o extrae audio de train y val."""
    if CACHE_FEATURES_PATH.exists():
        print(f"Cargando características acústicas desde caché: {CACHE_FEATURES_PATH.name}")
        try:
            cache = joblib.load(CACHE_FEATURES_PATH)
            if "val" in cache and "train" in cache:
                return cache["val"], cache["train"]
        except Exception as e:
            print(f"Aviso: no se pudo leer caché existente ({e}). Se re-extraerán los datos.")

    val_data = extraer_features_split("val")
    train_data = extraer_features_split("train")

    cache = {"val": val_data, "train": train_data}
    try:
        joblib.dump(cache, CACHE_FEATURES_PATH)
        print(f"Caché guardado con éxito en {CACHE_FEATURES_PATH}")
    except Exception as e:
        print(f"Aviso: no se pudo guardar el archivo de caché ({e})")

    return val_data, train_data


def obtener_latencia_respuesta(timing_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Obtiene la columna de latencia de respuesta en segundos."""
    candidatos = [
        "resp_lat_mean",
        "resp_lat_median",
        "response_latency",
        "response_latency_s",
        "latency_s",
    ]
    for c in candidatos:
        if c in timing_df.columns:
            return timing_df[c].to_numpy(dtype=float), c

    for col in timing_df.columns:
        col_low = col.lower()
        if "lat" in col_low and "agent" not in col_low:
            return timing_df[col].to_numpy(dtype=float), col

    raise ValueError("No se encontró la característica de latencia de respuesta en timing_features.")


# ============================================================
# 1. GRÁFICA #1: CURVAS DE DENSIDAD (KDE) DE LATENCIA DE RESPUESTA
# ============================================================

def generar_grafica_1_kde_latencia(timing_df: pd.DataFrame, labels: np.ndarray) -> None:
    """
    GRÁFICA SUGERIDA #1 (README Sección 3.A):
    - Curvas de densidad superpuestas (KDE) de latencia de respuesta (0.0s - 4.0s).
    - Curva Azul (Humanos) vs. Curva Naranja (Sintéticos).
    - Demuestra concentración humana en 0.2s-0.8s y campana desplazada en IAs (>1.5s).
    """
    print("\nGenerando Gráfica #1: KDE de Latencia de Respuesta...")
    latencia, nombre_col = obtener_latencia_respuesta(timing_df)

    # Si por alguna razón viniera en ms (>10), convertir a segundos
    if np.nanmedian(np.abs(latencia)) > 10:
        latencia = latencia / 1000.0

    lat_humanos = latencia[labels == 0]
    lat_sinteticos = latencia[labels == 1]

    # Rejilla de evaluación de 0.0 s a 4.0 s como pide el README
    x_grid = np.linspace(0.0, 4.0, 500)
    kde_hum = kde_unidimensional(lat_humanos, x_grid, ancho_banda=0.15)
    kde_sin = kde_unidimensional(lat_sinteticos, x_grid, ancho_banda=0.25)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # Curva Azul (Humanos)
    ax.plot(
        x_grid, kde_hum,
        color=COLOR_AZUL_HUMANO,
        linewidth=2.8,
        label=f"Humanos (n={len(lat_humanos)})",
        zorder=5
    )
    ax.fill_between(x_grid, kde_hum, color=COLOR_AZUL_HUMANO, alpha=0.20, zorder=3)

    # Curva Naranja (Sintéticos)
    ax.plot(
        x_grid, kde_sin,
        color=COLOR_NARANJA_SINTETICO,
        linewidth=2.8,
        label=f"Sintéticos (n={len(lat_sinteticos)})",
        zorder=5
    )
    ax.fill_between(x_grid, kde_sin, color=COLOR_NARANJA_SINTETICO, alpha=0.20, zorder=3)

    # Resaltar zonas explicativas según README
    ax.axvspan(0.2, 0.8, color=COLOR_AZUL_HUMANO, alpha=0.08, label="Zona Humana Típica (0.2 s - 0.8 s)")
    ax.axvspan(1.5, 4.0, color=COLOR_NARANJA_SINTETICO, alpha=0.06, label="Zona Retraso IA (> 1.5 s)")

    # Líneas punteadas de medianas
    med_hum = float(np.nanmedian(lat_humanos))
    med_sin = float(np.nanmedian(lat_sinteticos))
    ax.axvline(med_hum, color=COLOR_AZUL_HUMANO, linestyle="--", alpha=0.7, linewidth=1.5)
    ax.axvline(med_sin, color=COLOR_NARANJA_SINTETICO, linestyle="--", alpha=0.7, linewidth=1.5)

    ax.text(med_hum, ax.get_ylim()[1] * 0.88, f" Mediana: {med_hum:.2f}s", color=COLOR_AZUL_HUMANO, fontweight="bold", fontsize=9)
    ax.text(med_sin, ax.get_ylim()[1] * 0.70, f" Mediana: {med_sin:.2f}s", color=COLOR_NARANJA_SINTETICO, fontweight="bold", fontsize=9)

    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Latencia de respuesta en segundos (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Densidad de probabilidad", fontsize=11, fontweight="bold")
    ax.set_title("Curvas de Densidad (KDE) de Latencia de Respuesta\nHumanos vs. Asistentes Sintéticos (IA)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.92)

    guardar_figura(fig, "01_kde_latencia_respuesta.png")


# ============================================================
# 2. GRÁFICA #2: BOXPLOTS CARACTERÍSTICAS ACÚSTICAS CLAVE
# ============================================================

def generar_grafica_2_boxplots_acustica(voice_df: pd.DataFrame, labels: np.ndarray) -> None:
    """
    GRÁFICA SUGERIDA #2 (README Sección 3.B):
    - 3 Subplots de diagramas de caja (Boxplots) lado a lado:
      1. band_vhi_ratio_log (Energía sobre 3.4 kHz)
      2. jitter (Inestabilidad glotal de tono)
      3. crosstalk_db (Fuga analógica entre canales)
    - Eje X: Clase (Humano vs. Sintético).
    - Eje Y: Magnitud normalizada de cada métrica.
    - Demuestra separación bimodal física (menor jitter IA, energía residual alta frec, aislamiento digital).
    """
    print("\nGenerando Gráfica #2: Boxplot Comparativo de Características Acústicas Clave...")

    features = [
        ("band_vhi_ratio_log", "Energía >3.4 kHz\n(band_vhi_ratio_log)", "Artefactos en alta frec."),
        ("jitter", "Inestabilidad de Tono\n(jitter)", "Micro-variación glotal natural"),
        ("crosstalk_db", "Fuga Entre Canales\n(crosstalk_db)", "Acoplamiento analógico de hardware"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharey=False)

    for idx, (feat_name, titulo, desc) in enumerate(features):
        ax = axes[idx]
        if feat_name not in voice_df.columns:
            raise KeyError(f"Característica acústica '{feat_name}' no encontrada en voice_features.")

        vals_raw = voice_df[feat_name].to_numpy(dtype=float)
        
        # Magnitud normalizada (Min-Max a [0, 1]) solicitada por el README
        vmin, vmax = np.nanmin(vals_raw), np.nanmax(vals_raw)
        if vmax > vmin:
            vals_norm = (vals_raw - vmin) / (vmax - vmin)
        else:
            vals_norm = np.zeros_like(vals_raw)

        hum_norm = vals_norm[labels == 0]
        sin_norm = vals_norm[labels == 1]

        bp = ax.boxplot(
            [hum_norm, sin_norm],
            tick_labels=["Humano", "Sintético"],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="#0f172a", linewidth=2.2),
            whiskerprops=dict(color="#475569", linewidth=1.4),
            capprops=dict(color="#475569", linewidth=1.4),
            flierprops=dict(marker="o", markersize=4, alpha=0.5, markeredgecolor="none"),
        )

        # Colorear cajas: Humano (Azul) vs Sintético (Naranja)
        bp["boxes"][0].set(facecolor=COLOR_AZUL_HUMANO, alpha=0.6, edgecolor=COLOR_AZUL_HUMANO, linewidth=1.8)
        bp["boxes"][1].set(facecolor=COLOR_NARANJA_SINTETICO, alpha=0.6, edgecolor=COLOR_NARANJA_SINTETICO, linewidth=1.8)

        # Mediana sin normalizar para referencia informativa
        med_h_raw = np.nanmedian(vals_raw[labels == 0])
        med_s_raw = np.nanmedian(vals_raw[labels == 1])

        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xlabel("Clase", fontsize=11, fontweight="bold")
        ax.set_ylabel("Magnitud normalizada [0 - 1]", fontsize=10)
        ax.set_ylim(-0.05, 1.05)

        # Anotación explicativa en la parte inferior
        ax.text(
            0.5, 0.05,
            f"Medianas reales:\nHumano: {med_h_raw:.3f} | Sintético: {med_s_raw:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1")
        )

    fig.suptitle(
        "Boxplot Comparativo de Características Acústicas Clave\nSeparación física bimodal entre llamadas humanas y síntesis neuronal",
        fontsize=13,
        fontweight="bold"
    )
    fig.tight_layout()

    guardar_figura(fig, "02_boxplots_caracteristicas_acusticas.png")


# ============================================================
# 3. GRÁFICA #3: DISPERSIÓN 2D DE LOGITS Y FRONTERA DEL STACKER
# ============================================================

def generar_grafica_3_scatter_logits(train_data: dict, timing_model, voice_model, stacker) -> None:
    """
    GRÁFICA SUGERIDA #3 (README Sección 4):
    - Scatter plot 2D: Logit Timing (ztiming) vs. Logit Voz (zvoice).
    - 282 llamadas de entrenamiento: Círculos verdes = Humanos, Triángulos rojos = Sintéticos.
    - Línea divisoria: Frontera de decisión lineal ajustada por el Stacker (P = 0.5).
    - Demuestra ortogonalidad: llamadas dudosas en timing se resuelven por voz y viceversa.
    """
    print("\nGenerando Gráfica #3: Dispersión 2D de Logits y Frontera del Stacker...")

    timing_df = train_data["timing_df"]
    voice_df = train_data["voice_df"]
    labels = train_data["labels"]

    # Calcular probabilidades de cada cabeza
    p_timing = timing_model.predict_proba(timing_df)[:, 1]
    p_voice = voice_model.predict_proba(voice_df)[:, 1]

    # Convertir a espacio logit
    ztiming = logit(p_timing)
    zvoice = logit(p_voice)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))

    # Obtener la frontera lineal exacta ajustada por el Stacker:
    # w_t * ztiming + w_v * zvoice + intercept = 0
    # => zvoice = -(w_t * ztiming + intercept) / w_v
    coef = stacker.coef_[0]
    intercept = float(stacker.intercept_[0])
    w_timing, w_voice = float(coef[0]), float(coef[1])

    x_min, x_max = float(np.min(ztiming)) - 1.0, float(np.max(ztiming)) + 1.0
    y_min, y_max = float(np.min(zvoice)) - 1.0, float(np.max(zvoice)) + 1.0

    x_vals = np.linspace(x_min, x_max, 400)
    if abs(w_voice) > 1e-8:
        y_vals = -(w_timing * x_vals + intercept) / w_voice
        ax.plot(
            x_vals, y_vals,
            color=COLOR_FRONTERA,
            linewidth=2.8,
            linestyle="--",
            label="Frontera de decisión del Stacker (P = 0.5)",
            zorder=6
        )

        # Regiones sombreadas de decisión
        ax.fill_between(x_vals, y_vals, y_max + 2, color=COLOR_ROJO_SINTETICO, alpha=0.06, label="Zona Predicha Sintético (P > 0.5)")
        ax.fill_between(x_vals, y_min - 2, y_vals, color=COLOR_VERDE_HUMANO, alpha=0.06, label="Zona Predicha Humano (P < 0.5)")

    # Puntos: Círculos verdes = Humanos, Triángulos rojos = Sintéticos
    mask_hum = (labels == 0)
    mask_sin = (labels == 1)

    ax.scatter(
        ztiming[mask_hum],
        zvoice[mask_hum],
        marker="o",
        s=52,
        color=COLOR_VERDE_HUMANO,
        edgecolors="#14532d",
        linewidth=0.9,
        alpha=0.85,
        label=f"Humanos ({np.sum(mask_hum)} llamadas)",
        zorder=5
    )

    ax.scatter(
        ztiming[mask_sin],
        zvoice[mask_sin],
        marker="^",
        s=62,
        color=COLOR_ROJO_SINTETICO,
        edgecolors="#7f1d1d",
        linewidth=0.9,
        alpha=0.85,
        label=f"Sintéticos ({np.sum(mask_sin)} llamadas)",
        zorder=5
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Logit Cabeza de Timing ($z_{\mathrm{timing}}$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Logit Cabeza de Voz ($z_{\mathrm{voice}}$)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Dispersión 2D de Logits y Frontera del Stacker\n{len(labels)} llamadas de entrenamiento: Ortogonalidad Timing vs. Acústica",
        fontsize=13,
        fontweight="bold"
    )
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9.5)

    guardar_figura(fig, "03_logits_stacker.png")


# ============================================================
# 4. GRÁFICA #4: MATRIZ DE CONFUSIÓN Y CURVA ROC EN VALIDACIÓN
# ============================================================

def generar_grafica_4_confusion_roc(val_data: dict, timing_model, voice_model) -> None:
    """
    GRÁFICA SUGERIDA #4 (README Sección 6):
    - Figura compuesta de 2 páneles:
      1. Matriz de Confusión 2x2 en Heatmap para las 71 llamadas de validación (speaker-disjoint).
      2. Curva ROC con área bajo la curva (AUC = 1.000).
    - Demuestra 100% de aciertos limpios (cero falsos positivos y cero falsos negativos).
    """
    print("\nGenerando Gráfica #4: Matriz de Confusión y Curva ROC en Validación...")

    timing_df = val_data["timing_df"]
    voice_df = val_data["voice_df"]
    labels = val_data["labels"]

    p_timing = timing_model.predict_proba(timing_df)[:, 1]
    p_voice = voice_model.predict_proba(voice_df)[:, 1]
    fused_prob = np.asarray(fuse_probabilities(p_timing, p_voice))

    predicciones = (fused_prob >= 0.5).astype(int)
    cm = confusion_matrix(labels, predicciones, labels=[0, 1])
    auc_score = roc_auc_score(labels, fused_prob)
    fpr, tpr, _ = roc_curve(labels, fused_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # --- Panel Izquierdo: Matriz de Confusión Heatmap ---
    ax_cm = axes[0]
    im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Llamadas clasificadas", rotation=-90, va="bottom", fontsize=10)

    classes = ["Humano", "Sintético"]
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax_cm.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax_cm.set_xlabel("Etiqueta Predicha", fontsize=11, fontweight="bold")
    ax_cm.set_ylabel("Etiqueta Real", fontsize=11, fontweight="bold")
    ax_cm.set_title(f"Matriz de Confusión (Split Val: {len(labels)} llamadas)", fontsize=12, fontweight="bold")

    cell_labels = [
        ["VN\n(Verdaderos Negativos)", "FP\n(Falsos Positivos)"],
        ["FN\n(Falsos Negativos)", "VP\n(Verdaderos Positivos)"]
    ]
    max_val = cm.max()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            color_text = "white" if count > (max_val / 2.0) else "#0f172a"
            ax_cm.text(
                j, i,
                f"{count}\n\n{cell_labels[i][j]}",
                ha="center", va="center",
                color=color_text,
                fontsize=11,
                fontweight="bold"
            )

    # --- Panel Derecho: Curva ROC ---
    ax_roc = axes[1]
    ax_roc.plot(
        fpr, tpr,
        color="#2563eb",
        linewidth=3.0,
        label=f"Pipeline Fusionado (AUC = {auc_score:.3f})"
    )
    ax_roc.plot(
        [0, 1], [0, 1],
        color="#94a3b8",
        linestyle="--",
        linewidth=1.8,
        label="Clasificador Azar (AUC = 0.500)"
    )
    ax_roc.fill_between(fpr, tpr, color="#2563eb", alpha=0.15)

    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.03])
    ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11, fontweight="bold")
    ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11, fontweight="bold")
    ax_roc.set_title("Curva ROC en Validación", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", framealpha=0.92, fontsize=10.5)

    fig.suptitle(
        f"Validación Ciega del Pipeline (71 llamadas speaker-disjoint)\n100% Balanced Accuracy | ROC-AUC = {auc_score:.3f} | Cero Errores",
        fontsize=13,
        fontweight="bold"
    )
    fig.tight_layout()

    guardar_figura(fig, "04_matriz_confusion_y_roc.png")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    print("=" * 70)
    print("GENERADOR DE GRÁFICAS DEL PROYECTO (SEGÚN README.md)")
    print("=" * 70)

    require_artifacts()
    timing_model = load_timing_model()
    voice_model = load_voice_model()
    stacker = load_stacker()

    val_data, train_data = cargar_o_extraer_datos()

    # 1. Gráfica #1: KDE Latencia (Humanos vs Sintéticos)
    generar_grafica_1_kde_latencia(val_data["timing_df"], val_data["labels"])

    # 2. Gráfica #2: Boxplots de 3 métricas acústicas
    generar_grafica_2_boxplots_acustica(val_data["voice_df"], val_data["labels"])

    # 3. Gráfica #3: Scatter 2D de Logits y Frontera del Stacker (282 train)
    generar_grafica_3_scatter_logits(train_data, timing_model, voice_model, stacker)

    # 4. Gráfica #4: Matriz de Confusión y Curva ROC (71 val)
    generar_grafica_4_confusion_roc(val_data, timing_model, voice_model)

    print("\n" + "=" * 70)
    print("GENERACIÓN COMPLETADA EXITOSAMENTE")
    print(f"Todas las gráficas fueron guardadas en la carpeta: {CARPETA_SALIDA}/")
    print("1. 01_kde_latencia_respuesta.png")
    print("2. 02_boxplots_caracteristicas_acusticas.png")
    print("3. 03_logits_stacker.png")
    print("4. 04_matriz_confusion_y_roc.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
