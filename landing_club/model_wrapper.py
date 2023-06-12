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
    
   def preprocess(self, features):
      pass
    
   def postprocess(self, labels):
      pass


class Landing_Club_Wrapper(ModelWrapper):
   def __init__(self, scaler_X: StandardScaler, scaler_y: StandardScaler, label_encoder: LabelEncoder):
      super().__init__()
      self.scaler_X = scaler_X
      self.scaler_y = scaler_y
      self.label_encoder = label_encoder

   def preprocess(self, feature_list):
      encoded_home_owner = self.label_encoder.transform([feature_list[1]])[0]
      print(f'encoded homeowner = {encoded_home_owner}')
      feature_list[1] = 0
      print(f'features_list = {feature_list}')
      features = np.asarray(feature_list)
   
      print(f'Encoded = {features}')
      print(f'np_features = {features.reshape(1, -1)}')
      scaled = Landing_Club_Wrapper.standardize(self.scaler_X, [features], False)
      print(f'scaled = {scaled}')
      return
   
   def postprocess(self, labels):
      return self.scaler_X.inverse_transform([labels])
    
   def standardize(scaler, values, fit=False):
      if (fit):
         return scaler.fit_transform(values)
      else:
         return scaler.transform(values)
