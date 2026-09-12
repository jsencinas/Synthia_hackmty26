import json
import sys

from src.predict import predict_from_features


if __name__ == "__main__":
    if len(sys.argv) > 1:
        features = json.loads(sys.argv[1])
    else:
        from src.config import FEATURES_PATH
        import pandas as pd

        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                "No features.csv found. Run `python train.py` first."
            )
        features = pd.read_csv(FEATURES_PATH).iloc[0].to_dict()

    print(json.dumps(predict_from_features(features), indent=2))
