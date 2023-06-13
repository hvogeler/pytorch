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

    def preprocess(self, feature_list):
        encoded_home_owner = self.label_encoder.transform([feature_list[1]])[0]
        print(f'encoded homeowner = {encoded_home_owner}')
        feature_list[1] = 0
        print(f'features_list = {feature_list}')
        features = np.asarray(feature_list)

        print(f'Encoded = {features}')
        print(f'np_features = {features.reshape(1, -1)}')
        scaled = Landing_Club_Wrapper.standardize(
            self.scaler_X, [features], False)
        print(f'scaled = {scaled}')
        return

    def postprocess(self, labels):
        return self.scaler_X.inverse_transform([labels])

    def standardize(scaler, values, fit=False):
        if (fit):
            return scaler.fit_transform(values)
        else:
            return scaler.transform(values)
