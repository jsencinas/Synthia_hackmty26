#!/usr/bin/env python3
"""
train_model.py
Entrenamiento y evaluación de modelos de Machine Learning para clasificar llamadas
como 'human' o 'synthetic' a partir de los datos en resultados_turns.csv.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Columnas base esperadas del dataset
COLUMNAS_BASE = [
    "numero_turnos_caller",
    "duracion_promedio_caller",
    "pausa_promedio_caller",
    "desviacion_estandar_latencia",
    "interrupciones_agente",
    "duracion_total_interrupciones",
    "duracion_promedio_interrupcion",
]


def agregar_features_ingenieria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables calculadas (ratios e interacciones) que mejoran
    la capacidad de discriminación entre llamadas humanas y sintéticas.
    """
    df = df.copy()
    eps = 1e-5

    # 1. Tasa de interrupciones por turno del caller
    df["tasa_interrupciones"] = df["interrupciones_agente"] / (
        df["numero_turnos_caller"] + eps
    )

    # 2. Duración de interrupción promedio distribuida por número de turnos
    df["duracion_interrupciones_por_turno"] = df["duracion_total_interrupciones"] / (
        df["numero_turnos_caller"] + eps
    )

    # 3. Ratio entre tiempo de habla y tiempo de pausa
    df["ratio_habla_pausa"] = df["duracion_promedio_caller"] / (
        df["pausa_promedio_caller"] + eps
    )

    return df


def obtener_nombres_features() -> list:
    """Devuelve la lista completa de características que utiliza el modelo."""
    return COLUMNAS_BASE + [
        "tasa_interrupciones",
        "duracion_interrupciones_por_turno",
        "ratio_habla_pausa",
    ]


def cargar_datos(ruta_csv: str = "resultados_turns.csv"):
    """
    Carga el dataset y separa en splits train y val según la columna 'split'.
    Convierte label 'synthetic' en 1 y 'human' en 0.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    df = agregar_features_ingenieria(df)
    features = obtener_nombres_features()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    # Mapeo: synthetic = 1, human = 0
    # Esto se alinea con la especificación 'is_synthetic': true/false
    y_train = (train_df["label"] == "synthetic").astype(int)
    X_train = train_df[features]

    y_val = (val_df["label"] == "synthetic").astype(int)
    X_val = val_df[features]

    return X_train, y_train, X_val, y_val, features, train_df, val_df


def construir_modelos(random_state: int = 42) -> dict:
    """Crea el catálogo de modelos candidatos a comparar."""
    modelos = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }

    # Modelo ensamble con votación suave (soft voting)
    modelos["VotingEnsemble"] = VotingClassifier(
        estimators=[
            ("rf", modelos["RandomForest"]),
            ("xgb", modelos["XGBoost"]),
            ("gb", modelos["GradientBoosting"]),
            ("lr", modelos["LogisticRegression"]),
        ],
        voting="soft",
    )

    return modelos


def calibrar_umbral(modelo, X_val, y_val):
    """
    Evalúa diferentes umbrales de decisión para la probabilidad de 'synthetic'
    y selecciona el que maximiza la métrica F1-Score y Accuracy en validación.
    """
    probs = modelo.predict_proba(X_val)[:, 1]
    mejores_metricas = {"umbral": 0.5, "accuracy": 0.0, "f1": 0.0}

    for umbral in np.arange(0.30, 0.72, 0.02):
        preds = (probs >= umbral).astype(int)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, zero_division=0)

        score = (acc + f1) / 2.0
        if score > (mejores_metricas["accuracy"] + mejores_metricas["f1"]) / 2.0:
            mejores_metricas = {
                "umbral": round(float(umbral), 2),
                "accuracy": round(float(acc), 4),
                "f1": round(float(f1), 4),
            }

    return mejores_metricas


def main():
    print("=" * 80)
    print("DETECCIÓN DE LLAMADAS HUMAN VS SYNTHETIC - ENTRENAMIENTO DE MODELOS")
    print("=" * 80)

    X_train, y_train, X_val, y_val, features, train_df, val_df = cargar_datos()

    print(f"\nDatos cargados exitosamente:")
    print(f"  • Muestras de entrenamiento (Train): {len(X_train)} "
          f"(Human: {(y_train == 0).sum()}, Synthetic: {(y_train == 1).sum()})")
    print(f"  • Muestras de validación   (Val):   {len(X_val)} "
          f"(Human: {(y_val == 0).sum()}, Synthetic: {(y_val == 1).sum()})")
    print(f"  • Total características ({len(features)}): {', '.join(features)}")

    modelos = construir_modelos()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    resultados_comparacion = []

    print("\n" + "-" * 80)
    print("1. EVALUACIÓN Y COMPARACIÓN DE MODELOS")
    print("-" * 80)

    for nombre, modelo in modelos.items():
        # Validación cruzada 5-fold sobre el split de train
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=skf, scoring="roc_auc")
        cv_auc = cv_scores.mean()

        # Entrenar en conjunto de train completo
        modelo.fit(X_train, y_train)

        # Predicciones sobre validación
        val_preds = modelo.predict(X_val)
        val_probs = modelo.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, val_preds)
        bal_acc = balanced_accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        roc_auc = roc_auc_score(y_val, val_probs)

        resultados_comparacion.append(
            {
                "Modelo": nombre,
                "CV Train AUC": round(cv_auc, 4),
                "Val ROC-AUC": round(roc_auc, 4),
                "Val Acc": round(acc, 4),
                "Val Bal Acc": round(bal_acc, 4),
                "Val Prec": round(prec, 4),
                "Val Recall": round(rec, 4),
                "Val F1": round(f1, 4),
                "_instancia": modelo,
            }
        )

    df_res = pd.DataFrame(resultados_comparacion).drop(columns=["_instancia"])
    df_res = df_res.sort_values(by="Val ROC-AUC", ascending=False).reset_index(drop=True)
    print(df_res.to_string(index=False))

    # Seleccionar el mejor modelo según combinación de ROC-AUC en validación y estabilidad en CV
    mejor_resultado = max(resultados_comparacion, key=lambda r: (r["Val ROC-AUC"] + r["CV Train AUC"]) / 2)
    mejor_nombre = mejor_resultado["Modelo"]
    mejor_modelo = mejor_resultado["_instancia"]

    print("\n" + "=" * 80)
    print(f"MEJOR MODELO SELECCIONADO: {mejor_nombre}")
    print("=" * 80)

    val_preds_std = mejor_modelo.predict(X_val)
    val_probs_best = mejor_modelo.predict_proba(X_val)[:, 1]

    print("\nReporte de Clasificación en Validación (umbral estándar 0.50):")
    print(
        classification_report(
            y_val,
            val_preds_std,
            target_names=["human (0)", "synthetic (1)"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_val, val_preds_std)
    print("Matriz de Confusión (Umbral 0.50):")
    print(f"                Predicho Human    Predicho Synthetic")
    print(f"  Real Human           {cm[0, 0]:2d}                  {cm[0, 1]:2d}")
    print(f"  Real Synthetic       {cm[1, 0]:2d}                  {cm[1, 1]:2d}")

    # Calibración de umbral
    calibracion = calibrar_umbral(mejor_modelo, X_val, y_val)
    umbral_optimo = calibracion["umbral"]
    val_preds_opt = (val_probs_best >= umbral_optimo).astype(int)

    print(f"\nCalibración de Umbral:")
    print(f"  • Umbral óptimo encontrado: {umbral_optimo:.2f}")
    print(f"  • Accuracy con umbral {umbral_optimo:.2f}: {accuracy_score(y_val, val_preds_opt):.4f}")
    print(f"  • F1-Score con umbral {umbral_optimo:.2f}: {f1_score(y_val, val_preds_opt):.4f}")

    cm_opt = confusion_matrix(y_val, val_preds_opt)
    print("\nMatriz de Confusión (Umbral Óptimo):")
    print(f"                Predicho Human    Predicho Synthetic")
    print(f"  Real Human           {cm_opt[0, 0]:2d}                  {cm_opt[0, 1]:2d}")
    print(f"  Real Synthetic       {cm_opt[1, 0]:2d}                  {cm_opt[1, 1]:2d}")

    # Importancia de características
    importancias = {}
    if hasattr(mejor_modelo, "feature_importances_"):
        importancias = dict(
            zip(features, [round(float(v), 4) for v in mejor_modelo.feature_importances_])
        )
        print("\nImportancia de Características:")
        for feat, imp in sorted(importancias.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {feat:35s}: {imp:.4f} ({imp*100:.1f}%)")

    # Guardar modelo serializado y metadata
    ruta_modelo = "modelo_detector.joblib"
    ruta_meta = "modelo_metadata.json"

    joblib.dump(mejor_modelo, ruta_modelo)

    metadata = {
        "modelo_seleccionado": mejor_nombre,
        "features": features,
        "umbral_recomendado": umbral_optimo,
        "metricas_val_umbral_0_5": {
            "roc_auc": mejor_resultado["Val ROC-AUC"],
            "accuracy": mejor_resultado["Val Acc"],
            "f1_score": mejor_resultado["Val F1"],
            "precision": mejor_resultado["Val Prec"],
            "recall": mejor_resultado["Val Recall"],
        },
        "metricas_val_umbral_optimo": {
            "umbral": umbral_optimo,
            "accuracy": round(float(accuracy_score(y_val, val_preds_opt)), 4),
            "f1_score": round(float(f1_score(y_val, val_preds_opt)), 4),
        },
        "mapeo_clases": {
            "0": "human",
            "1": "synthetic",
        },
        "importancias": importancias,
    }

    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Modelo exportado a: {ruta_modelo}")
    print(f"Metadatos guardados en: {ruta_meta}")
    print("=" * 80)


if __name__ == "__main__":
    main()
