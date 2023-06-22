'''
This loader is called as external package from a graph transform function
Needs to be provided with the model
'''
import pickle as p
import os
from landing_club_prepost import LandingClubPrePost

project_dir = os.getenv('PROJECT_DIR')
wrapper_file = open(
    project_dir + '/py/landing_club_prepost.pickle', 'rb')
wrapper: LandingClubPrePost = p.load(wrapper_file)

def landing_club_pre(loan_amnt, home_ownership, annual_inc):
    '''Because Ab Initio external packages can not call object methods we need
       to proxy it
    '''
    return wrapper.preprocess(loan_amnt, home_ownership, annual_inc)

def landing_club_post(predict):
    '''Because Ab Initio external packages can not call object methods we need
       to proxy it
    '''
    return wrapper.postprocess(predict)
