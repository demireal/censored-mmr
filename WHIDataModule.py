import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import *

DATA_DIR = '/data/whi/data/main_study/processed/merged_v2.csv'
FEAT_DIR = 'data/whi/whi_features_new.txt'

class WHIDataModule:

    def __init__(self, S=0, fu_thresh = 7 * 365):

        self.S = S  # study indices (integer, 0 reserved for RCT)
        self.fu_thresh = fu_thresh

        self.df = pd.read_csv(DATA_DIR).query(f"S == {self.S}").dropna().reset_index(drop=True)

        self.df.loc[self.df['T'] >  self.fu_thresh, 'T'] =  self.fu_thresh
        self.df['Delta'] = (self.df['T'] <  self.fu_thresh) * self.df['Delta']

        self.cov_list = []
        with open(FEAT_DIR, 'r') as f:
            self.cov_list  = f.read().splitlines()

        self.df = self.df[["S", "A", "Delta", "T"] + self.cov_list]
        self.df_observed = self.df.copy()

    
    def get_df(self):
        return self.df.copy(), self.df_observed.copy()
    
    def get_covs(self):
        return self.cov_list.copy()