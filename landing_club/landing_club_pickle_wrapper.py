import pickle as p
from landing_club_wrapper import LandingClubWrapper

wrapper_file = open(
    '/disk1/dec/decisioning/py/landing_club_wrapper.pickle', 'rb')
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
