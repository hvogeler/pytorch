'''
Pre- and post-processing. After the preprocessing state is set
this object needs to be pickled to disk. The pickle file needs to be distributed
along with the Onnx model export.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class LandingClubPrePost:
    '''Wrapper class for pre and post processing methods for this model'''
    def __init__(self,
                 scaler_x: StandardScaler,
                 scaler_y: StandardScaler,
                 cat_encoder: OneHotEncoder):
        super().__init__()
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.cat_encoder = cat_encoder

    def preprocess(self, loan_amt: float, home_ownership: str, annual_inc: float) -> list:
        '''Pre process features'''
        feature_list = [loan_amt, home_ownership, annual_inc]
        nparr = np.array([home_ownership]).reshape(-1, 1)
        encoded_home_owner = self.cat_encoder.transform(nparr)

        feature_list = [loan_amt, annual_inc]
        feature_list = feature_list + encoded_home_owner[0].tolist()
        features = np.asarray(feature_list)

        scaled = LandingClubPrePost.standardize(
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
