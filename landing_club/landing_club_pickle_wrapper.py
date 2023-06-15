'''
This wrapper is called as external package from a graph transform function
Needs to be provided with the model
'''
import pickle as p
import os
from landing_club_wrapper import LandingClubWrapper

project_dir = os.getenv('PROJECT_DIR')
wrapper_file = open(
    project_dir + '/py/landing_club_wrapper.pickle', 'rb')
wrapper: LandingClubWrapper = p.load(wrapper_file)

def preprocess(loan_amnt, home_ownership, annual_inc):
    '''Because Ab Initio external packages can not call object methods we need
       to proxy it
    '''
    return wrapper.preprocess(loan_amnt, home_ownership, annual_inc)

def postprocess(predict):
    '''Because Ab Initio external packages can not call object methods we need
       to proxy it
    '''
    return wrapper.postprocess(predict)
