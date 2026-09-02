import os
import joblib
import numpy as np
import config

class ModelWrapper:

    def __init__(self, model_path=config.MODEL_PATH,
                 scaler_path=config.SCALER_PATH):

        self.model = joblib.load(model_path)

        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # Use only the fitted RF + LightGBM models.
        self.rf = None
        self.lgbm = None

        for name, est in self.model.estimators:
            if name == "rf":
                self.rf = est
            elif name == "lgbm":
                self.lgbm = est

        if self.rf is None:
            raise RuntimeError("Fitted Random Forest not found")

        if self.lgbm is None:
            raise RuntimeError("Fitted LightGBM not found")

        print("[model_wrapper] Using RF + LightGBM")
        print("[model_wrapper] RF:", type(self.rf).__name__)
        print("[model_wrapper] LGBM:", type(self.lgbm).__name__)

    def predict(self, feature_vector):

        x = np.asarray(feature_vector, dtype=float).reshape(1, -1)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        rf_proba = self.rf.predict_proba(x)[0]
        lgbm_proba = self.lgbm.predict_proba(x)[0]

        # Equal-weight soft voting between the two fitted models.
        proba = (rf_proba + lgbm_proba) / 2.0

        pred_class = int(np.argmax(proba))

        classes = self.model.classes_
        probabilities = {
            str(c): float(p)
            for c, p in zip(classes, proba)
        }

        return pred_class, probabilities


if __name__ == "__main__":

    from features import FEATURE_NAMES

    print("===== PHASE 3 MODEL TEST =====")
    print("FEATURE COUNT:", len(FEATURE_NAMES))
    print("FEATURES:", FEATURE_NAMES)

    wrapper = ModelWrapper()

    dummy = np.zeros(len(FEATURE_NAMES), dtype=float)

    pred, proba = wrapper.predict(dummy)

    print("Prediction:", pred)
    print("Probabilities:", proba)
    print("===== TEST PASSED =====")
