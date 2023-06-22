# %%
import pickle as p
import pandas as pd
import numpy as np
import onnx
import onnxruntime

from landing_club_prepost import LandingClubPrePost

# %%
# Read Data
SMALL_SAMPLE_DATASET_NAME = 'data/landing_club/accepted_2007_to_2018Q4_small_sample.csv'
acc_raw_df = pd.read_csv(SMALL_SAMPLE_DATASET_NAME)
# COL_NAMES = ['loan_amnt', 'home_ownership', 'annual_inc', 'purpose', 'addr_state', 'term', 'emp_length', 'int_rate']
# COL_NAMES = ['loan_amnt', 'home_ownership', 'annual_inc','term', 'emp_length', 'int_rate']
COL_NAMES = ['loan_amnt', 'home_ownership', 'annual_inc', 'int_rate']
COL_LABEL = 'int_rate'


# %%
# Remove records with nan
acc_df = acc_raw_df.loc[:, COL_NAMES]
print(acc_df)

# %%
# DataFrame to Numpy array
X_df = acc_df.drop(columns=[COL_LABEL])
y_df = acc_df[COL_LABEL]
X = X_df.to_numpy()
y = y_df.to_numpy()

# %%
# Test
model = onnx.load("landing_club_model.onnx")
onnx.checker.check_model(model)

ort_session = onnxruntime.InferenceSession("landing_club_model.onnx")

# %%
wrapper_file = open('landing_club_prepost.pickle', 'rb')
wrapper: LandingClubPrePost = p.load(wrapper_file)

for i in range(2):
    print(X[i])
    prepared_features = np.asarray(wrapper.preprocess(X[i, 0], X[i, 1], X[i, 2])).astype(np.float32)
    predict = ort_session.run(['prediction'], {'features': [prepared_features]})
    post_predict = wrapper.postprocess(predict[0].item())
    print(f'{X[i]} -> {prepared_features} -> {post_predict}')
    print(wrapper.scaler_x.inverse_transform([[-1.2608404,  -0.39100227, -0.9965061,  -0.35781503,  1.255253  ]]))
    print(wrapper.cat_encoder.inverse_transform([[1.80926996e-09, -4.79131558e-09, 1.00000002e+00]]))

# %%
