import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.hddm_w import HDDM_W

class EnsembleModel:
    def __init__(self, n_models, min_num_instances=5000, out_control_level=3):
        self.n_models = n_models
        self.ddm = DDM(min_num_instances=min_num_instances, out_control_level=out_control_level)
        self.models = []
        for i in range(n_models):
            self.models.append(HoeffdingTreeClassifier())

    def predict(self, X) -> np.ndarray:
        predictions = [self.models[i].predict(X) for i in range(self.n_models)]
        predictions = np.stack(predictions, axis=0)
        predictions = np.sum(predictions, axis=0) 
        predictions = predictions / self.n_models
        predictions = np.around(predictions, decimals=1)
        predictions = predictions.astype('int32')
        return predictions
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, y_error : np.ndarray):
        DATASET_SIZE = len(X)
        
        detected_drift = False
        for i in range(len(y_error)):
            self.ddm.add_element(y_error[i])
            if self.ddm.detected_change():
                print("detected")
                detected_drift = True
        
        if detected_drift:
            self.models.append(HoeffdingTreeClassifier()) # append a new model
            self.n_models += 1

        for i in range(self.n_models):
            start_index = i * (DATASET_SIZE // self.n_models)
            end_index = (i + 1) * (DATASET_SIZE // self.n_models) if i < (self.n_models - 1) else DATASET_SIZE
            X_current, y_current = X[start_index : end_index, :], y[start_index : end_index]
            self.models[i].partial_fit(X_current, y_current)