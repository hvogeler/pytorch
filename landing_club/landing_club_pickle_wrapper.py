import numpy as np
import pickle as p
from landing_club_wrapper import Landing_Club_Wrapper

wrapper_file = open('/disk1/dec/decisioning/py/landing_club_wrapper.pickle', 'rb')
wrapper: Landing_Club_Wrapper = p.load(wrapper_file)

def preprocess(loan_amnt, home_ownership, annual_inc):
    return wrapper.preprocess(loan_amnt, home_ownership, annual_inc)

def postprocess(predict):
    return wrapper.postprocess(predict)
