# %% [markdown]
# Isolated pre processing that can be used for training, validation and inference

# %%
import sys
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

# %%
METADATA = {
    'features': [
        {
            'name': 'loan_amnt',
            'type': 'number'
        },
        {
            'name': 'home_ownership',
            'type': 'string'
        },
        {
            'name': 'annual_inc',
            'type': 'number'
        }
    ],
    'labels': [
        {
            'name': 'int_rate',
            'type': 'number'
        }
    ]
}


def usage():
    print(f'usage: model_wrapper.py api | predict')


def get_raw_features_and_labels():
    return json.dumps(METADATA, indent=3)


if (__name__ == "__main__"):
    if (len(sys.argv) == 2):
        if (sys.argv[1] == 'metadata'):
            print(get_raw_features_and_labels())
            sys.exit(0)

    usage()
    sys.exit(1)


class ModelWrapper:
    def __init__(self):
        pass

    def preprocess(self, loan_amt: float, home_ownership: str, annual_inc: float) -> list:
        print('Please override the preprocess method!')
        sys.exit(1)
        pass

    def postprocess(self, prediction: float) -> float:
        print('Please override the preprocess method!')
        sys.exit(1)
