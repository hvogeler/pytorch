'''
Model wrapper for pre- and post-processing. After the preprocessing state is set
this object needs to be pickled to disk. The pickle file needs to be distributed
along with the Onnx model export.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class LandingClubWrapper:
    '''Wrapper class for pre and post processing methods for this model'''
    def __init__(self,
                 scaler_x: StandardScaler,
                 scaler_y: StandardScaler,
                 label_encoder: LabelEncoder):
        super().__init__()
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.label_encoder = label_encoder

    def preprocess(self, loan_amt: float, home_ownership: str, annual_inc: float) -> list:
        '''Pre process features'''
        feature_list = [loan_amt, home_ownership, annual_inc]

        encoded_home_owner = LandingClubWrapper.standardize(
            self.label_encoder,
            [feature_list[1]]
        )[0]

        feature_list[1] = encoded_home_owner
        features = np.asarray(feature_list)

        scaled = LandingClubWrapper.standardize(
            self.scaler_x,
            [features], False
        )

        return scaled[0].tolist()

    def postprocess(self, prediction: float) -> float:
        '''Post process the model prediction'''
        return self.scaler_y.inverse_transform([[prediction]])[0].item()

    @staticmethod
    def standardize(scaler, values, fit=False):
        '''Standardize features and predicition'''
        if fit:
            return scaler.fit_transform(values)
        else:
            return scaler.transform(values)
