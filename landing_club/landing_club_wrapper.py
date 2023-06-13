from model_wrapper import ModelWrapper
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#
# Model wrapper for pre- and post-processing. After the preprocessing state is set
# this object needs to be pickled to disk. The pickle file needs to be distributed
# along with the Onnx model export.
#

class Landing_Club_Wrapper(ModelWrapper):
    def __init__(self, scaler_X: StandardScaler, scaler_y: StandardScaler, label_encoder: LabelEncoder):
        super().__init__()
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.label_encoder = label_encoder

    # feature_row has the form [0.1, 0.2, 0.3]
    def preprocess(self, feature_row):
        feature_list = feature_row.copy()
        encoded_home_owner = self.label_encoder.transform([feature_list[1]])[0]
        feature_list[1] = encoded_home_owner
        features = np.asarray(feature_list)

        scaled = Landing_Club_Wrapper.standardize(
            self.scaler_X, [features], False)
        return scaled[0]

    # label_array is a one dimensional array like [0.2]
    def postprocess(self, label_array):
        # inverse_transform expects a 2-dimensional array like [[0.2]]
        return self.scaler_y.inverse_transform([label_array])

    def standardize(scaler, values, fit=False):
        if (fit):
            return scaler.fit_transform(values)
        else:
            return scaler.transform(values)
