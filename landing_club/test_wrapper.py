from model_wrapper import Landing_Club_Wrapper
import sys
import pickle as p
import numpy as np

wrapper_file = open('landing_club_wrapper.pickle', 'rb')
wrapper = p.load(wrapper_file)
feature_list = [3600.0, 'MORTGAGE', 40000.0]
print(f'features = {feature_list}')
p1 = wrapper.preprocess(feature_list)
print(f'p1: {p1}')

out = [-1.37928961, -0.86447492, -0.96778147]
loan_amnt = wrapper.postprocess(out)
print(f'inversed loan_amt = {loan_amnt}')