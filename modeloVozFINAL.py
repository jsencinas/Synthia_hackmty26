import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


VOICE_MODEL_PATH = "modelo_xgboost.json"

features = [
    "mfcc_1_std",
    "shimmer",
    "pitch_delta_std",
    "jitter"
]

_modelo = None


def cargar_modelo(ruta=VOICE_MODEL_PATH):
    global _modelo
    if _modelo is None:
        modelo = XGBClassifier()
        modelo.load_model(ruta)
        _modelo = modelo
    return _modelo


def clasificar_llamada(mfcc_1_std, shimmer, pitch_delta_std, jitter):
    modelo = cargar_modelo()

    datos = pd.DataFrame([{
        "mfcc_1_std": mfcc_1_std,
        "shimmer": shimmer,
        "pitch_delta_std": pitch_delta_std,
        "jitter": jitter
    }])

    probabilidad_sintetica = float(modelo.predict_proba(datos)[0][1])

    if probabilidad_sintetica >= 0.5:
        decision = "synthetic"
    else:
        decision = "human"

    confianza = max(
        probabilidad_sintetica,
        1 - probabilidad_sintetica
    )

    return decision, confianza


def main():
    df = pd.read_csv("caracteristicas_finales.csv")

    print("Datos cargados.")
    print("Número de llamadas:", len(df))

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    X_train = train[features]
    X_val = val[features]

    y_train = train["label"].map({
        "human": 0,
        "synthetic": 1
    })

    y_val = val["label"].map({
        "human": 0,
        "synthetic": 1
    })

    print("\nDatos de entrenamiento:", len(X_train))
    print("Datos de validación:", len(X_val))

    modelo = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    print("\nEntrenando XGBoost...")
    modelo.fit(X_train, y_train)
    print("Entrenamiento terminado.")

    predicciones = modelo.predict(X_val)
    probabilidades = modelo.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predicciones)
    precision = precision_score(y_val, predicciones)
    recall = recall_score(y_val, predicciones)
    f1 = f1_score(y_val, predicciones)
    roc_auc = roc_auc_score(y_val, probabilidades)
    matriz = confusion_matrix(y_val, predicciones)

    print("\n==========================================")
    print("RESULTADOS DE XGBOOST")
    print("==========================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nMatriz de confusión:")
    print(matriz)

    modelo.save_model(VOICE_MODEL_PATH)
    print("\nModelo guardado como:")
    print(VOICE_MODEL_PATH)

    global _modelo
    _modelo = modelo

    mfcc_1_std = 80.0
    shimmer = 0.20
    pitch_delta_std = 35.0
    jitter = 0.10

    decision, confianza = clasificar_llamada(
        mfcc_1_std,
        shimmer,
        pitch_delta_std,
        jitter
    )

    print("\n==========================================")
    print("DECISIÓN PARA UNA NUEVA LLAMADA")
    print("==========================================")

    print("Resultado:", decision)
    print("Confianza:", round(confianza, 4))


if __name__ == "__main__":
    main()
