"""
Loads your Phase 2 trained ensemble (Voting Classifier) and runs inference
on live feature vectors.
"""

import os
import joblib
import numpy as np

import config


class ModelWrapper:
    def __init__(self, model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Save your Phase 2 model with joblib.dump(voting_clf, '{model_path}') first."
            )
        self.model = joblib.load(model_path)

        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        elif scaler_path:
            print(f"[model_wrapper] No scaler found at {scaler_path} — "
                  f"proceeding without scaling. If you scaled features during "
                  f"training, predictions WILL be wrong without this.")

    def predict(self, feature_vector):
        """
        feature_vector: 1-D numpy array matching training feature order.
        Returns: (predicted_class, class_probabilities_dict)
        """
        x = feature_vector.reshape(1, -1)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        pred_class = self.model.predict(x)[0]

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba_arr = self.model.predict_proba(x)[0]
            classes = self.model.classes_
            proba = {int(c): float(p) for c, p in zip(classes, proba_arr)}

        return pred_class, proba


if __name__ == "__main__":
    # Quick manual test with a dummy feature vector (won't be meaningful,
    # just confirms the model loads and runs)
    import numpy as np
    from features import FEATURE_NAMES

    wrapper = ModelWrapper()
    dummy = np.zeros(len(FEATURE_NAMES))
    pred, proba = wrapper.predict(dummy)
    print("Prediction:", pred)
    print("Probabilities:", proba)
