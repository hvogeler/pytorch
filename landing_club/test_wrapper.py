
import sys
import pickle as p
import numpy as np
from landing_club_wrapper import Landing_Club_Wrapper

wrapper_file = open('landing_club/landing_club_wrapper.pickle', 'rb')
wrapper: Landing_Club_Wrapper = p.load(wrapper_file)
# feature_list = [3600.0, 'RENT', 40000.0]
# print(f'features = {feature_list}')
# p1 = wrapper.preprocess(feature_list)
# print(f'p1: {p1}')

features = [[2.2408, -0.8231,  0.6920]]
int_rate = [0.6436]
inv_features = wrapper.scaler_X.inverse_transform(features)
print(f'loan_amnt = {int(round(inv_features[0, 0], 0))}')
print(f'annual_inc = {int(round(inv_features[0, 2], 0))}')
home_ownership = int(round(inv_features[0, 1], 0))
print(f'home_ownership = {wrapper.label_encoder.inverse_transform([home_ownership])}')

inv_int_rate = wrapper.scaler_y.inverse_transform([int_rate])
print(f'int_rate = {inv_int_rate[0].item():.2f} %')
