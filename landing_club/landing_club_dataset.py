import pandas as pd
from pandas import DataFrame
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import torch.onnx
import matplotlib.pyplot as plt
import pickle as p
from landing_club_prepost import LandingClubPrePost


class Landing_Club_Dataset(Dataset):
    def __init__(self, file_path: str, col_label, col_names = []):
        self.sc_x = StandardScaler()
        self.sc_y = StandardScaler()
        self.one_hot_enc = OneHotEncoder(sparse=False)
        self.file_path = file_path
        self.col_label = col_label
        self.col_names = col_names
        self.read_and_preprocess_datafile()
        
    def read_and_preprocess_datafile(self):
        self.df = pd.read_csv(self.file_path, usecols = self.col_names)
        self.clean()
        self.split_features_and_labels() # split feature columns and label column
        self.encode_categorical_columns()
        self.split_train_test_and_standardize(test_size=.2)
        self.to_tensor()
        self.wrapper = LandingClubPrePost(self.sc_x, self.sc_y, self.one_hot_enc)
        self.write_prepost_pickle()
            
    def __getitem__(self, index: int):
        return (self.X_tsor[index], self.y_tsor[index])
    
    def __len__(self):
        return len(self.df)
    
    def to_numpy(self):
        return self.df.to_numpy()
    
    def to_tensor(self, with_grad=False):
        self.X_tsor = torch.tensor(DataFrame.to_numpy(self.X_df), requires_grad=with_grad)
        self.y_tsor = torch.tensor(self.y_df.to_numpy(), requires_grad=with_grad)
    
    def clean(self):
        self.df.dropna(inplace=True)
        return
    
    def encode_categorical_columns(self):
        categorical_cols = list(self.X_df.select_dtypes(include=['object']))
        for col in categorical_cols:
            categories = self.X_df[col].to_numpy().reshape(-1,1)
            oh_encoded = self.one_hot_enc.fit_transform(categories)
            oh_df = pd.DataFrame(oh_encoded, columns=self.one_hot_enc.get_feature_names_out())
            self.X_df = self.X_df.join(oh_df)
            self.X_df.drop(columns=[col], inplace=True)
            print('AAAAAAAA')
            print(self.X_df)
            # label_encoded = self.label_enc.fit_transform(self.X_df[col])
            # self.X_df[col] = label_encoded
            # self.X_df = pd.get_dummies(self.X_df, columns=['home_ownership'])

    def split_features_and_labels(self):
        self.X_df = self.df.drop(columns=[self.col_label])
        self.y_df = self.df[self.col_label]
        print(type(self.y_df))

    def split_train_test_and_standardize(self, test_size):
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = \
            train_test_split(self.X_df.values, self.y_df.values, test_size=test_size, shuffle=False)

        y_train_orig = y_train_orig.reshape(-1, 1)
        y_test_orig = y_test_orig.reshape(-1, 1)

        self.X_train = LandingClubPrePost.standardize(self.sc_x, X_train_orig, True)
        self.X_test = LandingClubPrePost.standardize(self.sc_x, X_test_orig, False)
        self.y_train = LandingClubPrePost.standardize(self.sc_y, y_train_orig, True)
        self.y_test = LandingClubPrePost.standardize(self.sc_y, y_test_orig, False)

    def get_train_dataset(self):
        X_train_tsor = torch.from_numpy(self.X_train.astype(np.float32))
        y_train_tsor = torch.from_numpy(self.y_train.astype(np.float32))
        return (X_train_tsor, y_train_tsor)

    def get_train_dataloader(self, batch_size):
        X_train_tsor = torch.from_numpy(self.X_train.astype(np.float32))
        y_train_tsor = torch.from_numpy(self.y_train.astype(np.float32))
        return DataLoader(TensorDataset(X_train_tsor, y_train_tsor), batch_size=batch_size)

    def get_test_dataset(self):
        X_test_tsor = torch.from_numpy(self.X_test.astype(np.float32))
        y_test_tsor = torch.from_numpy(self.y_test.astype(np.float32))
        return (X_test_tsor, y_test_tsor)

    def write_prepost_pickle(self):    
        wrapper_file = open('landing_club_prepost.pickle', 'wb')
        p.dump(self.wrapper, wrapper_file)
        wrapper_file.close()
    
