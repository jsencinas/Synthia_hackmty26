from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FEATURES_PATH, PLOTS_DIR
from .feature_engineering import FEATURE_NAMES
from .utils import ensure_directories


PLOT_FEATURES = [
    "numero_turnos_caller",
    "pausa_promedio_caller",
    "latencia_promedio",
    "latencia_std",
    "numero_interrupciones",
    "interruption_rate",
    "overlap_ratio",
]


def run_eda() -> None:
    ensure_directories()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run `python train.py` first."
        )

    df = pd.read_csv(FEATURES_PATH)

    print("\n=== DATASET ===")
    print(f"Calls: {len(df)}")
    print(f"Human: {(df['label'] == 'human').sum()}")
    print(f"Synthetic: {(df['label'] == 'synthetic').sum()}")

    print("\n=== DESCRIPTIVE STATISTICS BY CLASS ===")
    stats = (
        df.groupby("label")[FEATURE_NAMES]
        .agg(["mean", "median", "std", "min", "max"])
        .round(4)
    )
    print(stats.to_string())

    print("\n=== MISSING VALUES ===")
    print(df[FEATURE_NAMES].isna().sum().sort_values(ascending=False))

    for feature in PLOT_FEATURES:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="label", y=feature)
        plt.title(f"Human vs Synthetic — {feature}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{feature}_human_vs_synthetic.png", dpi=150)
        plt.close()

    plt.figure(figsize=(12, 10))
    correlation = df[FEATURE_NAMES].corr(numeric_only=True)
    sns.heatmap(correlation, cmap="coolwarm", center=0)
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    run_eda()
