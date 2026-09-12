#!/usr/bin/env python3
"""
train_turn_model.py
Entrenamiento y evaluación de modelos de Machine Learning para clasificar llamadas
como 'human' o 'synthetic' a partir de los turnos de conversación.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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


def cargar_datos(ruta_csv: str = "resultados_turns.csv", n_test_holdout: int = 30):
    """
    Carga el dataset y separa en splits train y val según la columna 'split'.

    De 'train' se separan n_test_holdout llamadas (estratificadas por clase)
    para formar un 'test_holdout' que nunca se usa para entrenar ni para
    elegir modelo - solo se toca una vez, al final, para reportar la métrica
    real de generalización.

    'val' se deja completo (sin partir) y se usa únicamente para calibrar
    el umbral de decisión.

    Convierte label 'synthetic' en 1 y 'human' en 0.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    df = agregar_features_ingenieria(df)
    features = obtener_nombres_features()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    y_train_completo = (train_df["label"] == "synthetic").astype(int)
    X_train_completo = train_df[features]

    # Separamos n_test_holdout llamadas de TRAIN (no de val) para el reporte final.
    # Al venir del split más grande, apenas reduce los datos de entrenamiento
    # y deja val completo disponible para calibrar el umbral.
    (
        X_train_fit,
        X_test_holdout,
        y_train_fit,
        y_test_holdout,
    ) = train_test_split(
        X_train_completo,
        y_train_completo,
        test_size=n_test_holdout,
        stratify=y_train_completo,
        random_state=42,
    )

    y_val = (val_df["label"] == "synthetic").astype(int)
    X_val = val_df[features]

    return (
        X_train_fit,
        y_train_fit,
        X_val,
        y_val,
        X_test_holdout,
        y_test_holdout,
        features,
        train_df,
        val_df,
    )


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


def calibrar_umbral(modelo, X_val_calib, y_val_calib):
    """
    Evalúa diferentes umbrales de decisión para la probabilidad de 'synthetic'
    usando SOLO el subset de calibración (val_calib), y selecciona el que
    maximiza la métrica F1-Score y Accuracy. val_test nunca se toca aquí.
    """
    probs = modelo.predict_proba(X_val_calib)[:, 1]
    mejores_metricas = {"umbral": 0.5, "accuracy": 0.0, "f1": 0.0}

    for umbral in np.arange(0.30, 0.72, 0.02):
        preds = (probs >= umbral).astype(int)
        acc = accuracy_score(y_val_calib, preds)
        f1 = f1_score(y_val_calib, preds, zero_division=0)

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

    (
        X_train_fit,
        y_train_fit,
        X_val,
        y_val,
        X_test_holdout,
        y_test_holdout,
        features,
        train_df,
        val_df,
    ) = cargar_datos()

    print(f"\nDatos cargados exitosamente:")
    print(f"  • Muestras de entrenamiento (train_fit):    {len(X_train_fit)} "
          f"(Human: {(y_train_fit == 0).sum()}, Synthetic: {(y_train_fit == 1).sum()})")
    print(f"  • Muestras de calibración   (val completo): {len(X_val)} "
          f"(Human: {(y_val == 0).sum()}, Synthetic: {(y_val == 1).sum()})")
    print(f"  • Muestras de test final    (test_holdout):  {len(X_test_holdout)} "
          f"(Human: {(y_test_holdout == 0).sum()}, Synthetic: {(y_test_holdout == 1).sum()})")
    print(f"  • Total características ({len(features)}): {', '.join(features)}")
    print("\n  Nota: test_holdout sale de 'train' (no de 'val') y no se usa para")
    print("  entrenar ni para elegir modelo - solo se toca al final, para reportar")
    print("  la métrica real. 'val' queda completo, dedicado solo a calibrar el umbral.")

    modelos = construir_modelos()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    resultados_comparacion = []

    print("\n" + "-" * 80)
    print("1. EVALUACIÓN Y COMPARACIÓN DE MODELOS (solo con cross-validation en train_fit)")
    print("-" * 80)

    for nombre, modelo in modelos.items():
        # Validación cruzada 5-fold sobre train_fit.
        # Esta es la ÚNICA señal que se usa para elegir el modelo.
        cv_scores = cross_val_score(modelo, X_train_fit, y_train_fit, cv=skf, scoring="roc_auc")
        cv_auc = cv_scores.mean()
        cv_auc_std = cv_scores.std()

        # Entrenar en train_fit completo
        modelo.fit(X_train_fit, y_train_fit)

        resultados_comparacion.append(
            {
                "Modelo": nombre,
                "CV Train AUC": round(cv_auc, 4),
                "CV Train AUC std": round(cv_auc_std, 4),
                "_instancia": modelo,
            }
        )

    df_res = pd.DataFrame(resultados_comparacion).drop(columns=["_instancia"])
    df_res = df_res.sort_values(by="CV Train AUC", ascending=False).reset_index(drop=True)
    print(df_res.to_string(index=False))

    # Seleccionar el mejor modelo SOLO según CV Train AUC (sin mirar val ni test_holdout)
    mejor_resultado = max(resultados_comparacion, key=lambda r: r["CV Train AUC"])
    mejor_nombre = mejor_resultado["Modelo"]
    mejor_modelo = mejor_resultado["_instancia"]

    print("\n" + "=" * 80)
    print(f"MEJOR MODELO SELECCIONADO (por CV en train_fit): {mejor_nombre}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # A partir de aquí es la ÚNICA vez que tocamos val y test_holdout.
    # ------------------------------------------------------------------

    # Calibración de umbral usando val completo
    calibracion = calibrar_umbral(mejor_modelo, X_val, y_val)
    umbral_optimo = calibracion["umbral"]

    print(f"\nCalibración de Umbral (usando val completo, {len(X_val)} muestras):")
    print(f"  • Umbral óptimo encontrado: {umbral_optimo:.2f}")
    print(f"  • Accuracy en val con ese umbral: {calibracion['accuracy']:.4f}")
    print(f"  • F1-Score en val con ese umbral: {calibracion['f1']:.4f}")

    # Reporte final: única vez que se usa test_holdout, y solo para reportar
    test_probs = mejor_modelo.predict_proba(X_test_holdout)[:, 1]
    test_preds_std = (test_probs >= 0.50).astype(int)
    test_preds_opt = (test_probs >= umbral_optimo).astype(int)

    print(f"\n" + "-" * 80)
    print(f"REPORTE FINAL SOBRE test_holdout ({len(X_test_holdout)} muestras nunca antes usadas)")
    print("-" * 80)

    print("\nCon umbral estándar (0.50):")
    print(
        classification_report(
            y_test_holdout,
            test_preds_std,
            target_names=["human (0)", "synthetic (1)"],
            digits=4,
        )
    )

    print(f"Con umbral calibrado ({umbral_optimo:.2f}):")
    print(
        classification_report(
            y_test_holdout,
            test_preds_opt,
            target_names=["human (0)", "synthetic (1)"],
            digits=4,
        )
    )

    roc_auc_test = roc_auc_score(y_test_holdout, test_probs)
    acc_test_opt = accuracy_score(y_test_holdout, test_preds_opt)
    f1_test_opt = f1_score(y_test_holdout, test_preds_opt)

    print(f"ROC-AUC en test_holdout: {roc_auc_test:.4f}")
    print(f"Accuracy en test_holdout (umbral calibrado): {acc_test_opt:.4f}")
    print(f"F1-Score en test_holdout (umbral calibrado): {f1_test_opt:.4f}")

    cm_test = confusion_matrix(y_test_holdout, test_preds_opt)
    print("\nMatriz de Confusión en test_holdout (Umbral Calibrado):")
    print(f"                Predicho Human    Predicho Synthetic")
    print(f"  Real Human           {cm_test[0, 0]:2d}                  {cm_test[0, 1]:2d}")
    print(f"  Real Synthetic       {cm_test[1, 0]:2d}                  {cm_test[1, 1]:2d}")

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
    ruta_modelo = "turn_model.joblib"
    ruta_meta = "metadata_turn_model.json"

    joblib.dump(mejor_modelo, ruta_modelo)

    metadata = {
        "modelo_seleccionado": mejor_nombre,
        "features": features,
        "umbral_recomendado": umbral_optimo,
        "seleccion_modelo": {
            "metodo": "cross_val_score (5-fold) sobre train_fit, sin usar val ni test_holdout",
            "cv_train_auc": mejor_resultado["CV Train AUC"],
            "cv_train_auc_std": mejor_resultado["CV Train AUC std"],
        },
        "metricas_calibracion_umbral": {
            "umbral": umbral_optimo,
            "accuracy": calibracion["accuracy"],
            "f1_score": calibracion["f1"],
            "n_muestras": len(X_val),
            "fuente": "val completo",
        },
        "metricas_test_holdout": {
            "umbral": umbral_optimo,
            "roc_auc": round(float(roc_auc_test), 4),
            "accuracy": round(float(acc_test_opt), 4),
            "f1_score": round(float(f1_test_opt), 4),
            "n_muestras": len(X_test_holdout),
            "fuente": "30 llamadas separadas de train, nunca usadas para entrenar ni calibrar",
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