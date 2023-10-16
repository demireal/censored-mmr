import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import *


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CONT_COLS = ["bw", "b.head", "preterm",  "birth.o" ,"nnhealth", "momage"]

DISCRETE_COLS = ["sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll",
                 "cig", "first", "booze", "drugs", "work.dur", "prenatal", "ark",
                 "ein", "har", "mia", "pen", "tex", "was", "momwhite", "momblack", "momhisp"]

class IHDPDataModule:
    '''
    Current version generates a dataframe with:
        Covariates from IHDP data
        Logistic propensity scores P(A=1 | X)
        Time-to-event variables from a CoxPH model (both for potential outcomes Y0 Y1 and censoring times C0 C1)

        Different models for data generating (e.g. AFT model) are not implemented yet, but their inregration should be easy. 
    '''
    def __init__(self, save_df=True, d=1, n=100, S=0,
                    px_cols = None,
                    prop_fn='sigmoid', prop_args={'beta': [0, 0]},
                    tte_params =
                    {'model': 'coxph',
                    'hazard': 'weibull',
                    'cox_args': {'Y0': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'Y1': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'C0': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'C1': {'beta': [0,0], 'lambda': 1, 'p': 1},},
                    },
                global_threshold=None,
                seed=None,
                ):

        self.d = d  # covariate dimension (integer)
        self.n = n  # sample size (integer)
        self.S = S  # study indices (integer, 0 reserved for RCT)
        #self.px_dist = px_dist  # covariate distribution (string)
        #self.px_args = px_args  # parameters for covariate distribution (dict)
        self.px_cols = px_cols
        self.prop_fn = prop_fn  # oracle propensity score model, e.g., sigmoid function (string)
        self.prop_args = prop_args  # parameters for the oracle proensity score model (dict)
        self.tte_params = tte_params  # the model that specifies how the oracle time-to-event variables are generated (string)
                                    # (for Cox model, keep first term of beta 0 always to not run into errors later with libraries)
                                    # (same effect can be achieved via lambda&p anyways)
        self.global_thresh = global_threshold # threshold for global censoring ( set censoring value to this if Y > thresh)
        self.seed = seed

        self.df_save_dir = os.path.join(DATA_DIR,f'ihdp/S{self.S}/csv')  # directory to save the DataFrames
        self.fig_save_dir = os.path.join(DATA_DIR,f'ihdp/S{self.S}/figures')  # directory to save the figures
        self.df_load_dir = os.path.join(DATA_DIR,f"ihdp/ihdp.csv")
        self.cov_list = [f'X{i}' for i in range(d+1)]
        
        self.og_df = pd.read_csv(self.df_load_dir)
        self.og_df_len = len(self.og_df)
        
        scaler = MinMaxScaler()
        self.og_df[CONT_COLS] = scaler.fit_transform(self.og_df[CONT_COLS]) 
        self.og_df[DISCRETE_COLS] = self.og_df[DISCRETE_COLS] - 0.5
        
        self.df, self.df_observed = self._generate_data()
        if save_df: self._save_csv()

    
    def summary(self, cov_dim=1, plot=True, save_fig=True):
        summary_df = pd.DataFrame(index=np.arange(1))
        summary_df['S'] = self.S
        summary_df['d'] = self.d
        summary_df['n'] = self.n

        self._ipcw_oracle()

        summary_df['True mean Y0'] = self.df['Y0'].mean()
        summary_df['Impute mean Y0'] = self.df_observed.query('A == 0')['T'].mean()
        summary_df['Drop mean Y0'] = self.df_observed.query('Delta == 1 & A == 0')['T'].mean()
        summary_df['Oracle-IPCW mean Y0'] = self.df['ipcw_oracle_Y0'].mean()

        summary_df['True mean Y1'] = self.df['Y1'].mean()
        summary_df['Impute mean Y1'] = self.df_observed.query('A == 1')['T'].mean()
        summary_df['Drop mean Y1'] = self.df_observed.query('Delta == 1 & A == 1')['T'].mean()
        summary_df['Oracle-IPCW mean Y1'] = self.df['ipcw_oracle_Y1'].mean()

        summary_df['True ATE'] = summary_df['True mean Y1'] - summary_df['True mean Y0']
        summary_df['Impute ATE'] = summary_df['Impute mean Y1'] - summary_df['Impute mean Y0']
        summary_df['Drop ATE'] = summary_df['Drop mean Y1'] - summary_df['Drop mean Y0']
        summary_df['Oracle-IPCW ATE'] = self.df['ipcw_oracle_CATE'].mean()

        if plot: self._plot(save_fig, cov_dim)

        return summary_df

    
    def get_df(self):
        return self.df.copy(), self.df_observed.copy()

    
    def get_covs(self):
        return self.cov_list.copy()
    

    def calc_oracle_prop(self, arr, covariate='X1'):
        X = pd.DataFrame(columns=self.cov_list, index=np.arange(len(arr)))
        X[self.cov_list] = 0
        X['X0'] = 1
        X[covariate] = arr
        if self.prop_fn == 'sigmoid':
            return sigmoid_fn(X, self.prop_args['beta'])
        else:
            raise NotImplementedError(f'Propensity score method <{self.prop_fn}> is not implemented.')  
        

    def get_oracle_surv_curve(self, arr, cov_vals, outcome):
        '''
        Get the survival probabilities for a given set of covariates and an outcome.
        @params:
            arr: array of timesteps to get the survival probabilities over (numpy array)
            cov_vals: covariate values (list)
            outcome: e.g. 'Y0', 'C1' (string)
        '''
        if self.tte_params['model'] == 'coxph':
            if self.tte_params['hazard'] == 'weibull':
                return weibull_oracle_adj_surv(arr, np.array(cov_vals), self.tte_params['cox_args'][outcome])
            
            else:
                raise NotImplementedError(f'Baseline hazard model <{self.tte_params["hazard"]}> is not implemented.')
            
        else:
            raise NotImplementedError(f'Time-to-event model <{self.tte_params["model"]}> is not implemented.')
    

    def _generate_data(self):
        np.random.seed(self.seed)

        df = pd.DataFrame(index=np.arange(self.n))
        df['S'] = self.S
        X = sm.add_constant(self._sample_x())  # add a column of 1's for the bias terms in the linear models
        df[self.cov_list] = X  # generated covariates

        df['prop_score'] = self._calc_propensity(X)  # P(A=1 | X, S=self.S)
        df['A'] = np.array(df['prop_score'] > np.random.uniform(size=self.n), dtype=int)  # sample treatment A

        df['Y0'] = self._sample_tte(X, 'Y0')
        df['Y1'] = self._sample_tte(X, 'Y1')
        df['C0'] = self._sample_tte(X, 'C0')
        df['C1'] = self._sample_tte(X, 'C1')
        
        # the version below leads to crazy high variance in the CDR estimator
#         if self.global_thresh is not None:
#             df.loc[df.Y0>self.global_thresh,'C0'] = self.global_thresh
#             df.loc[df.Y1>self.global_thresh,'C1'] = self.global_thresh

        if self.global_thresh is not None:
            df.loc[df.Y1 > self.global_thresh,'C1'] = self.global_thresh * np.random.rand(len(df.loc[df.Y1 > self.global_thresh]))           
        
        df['Y'] =  df['A'] * df['Y1'] + (1 - df['A']) * df['Y0']  # record the realized potential event time 
        df['C'] =  df['A'] * df['C1'] + (1 - df['A']) * df['C0']  # record the realized potential censoring time

        df['Delta'] = (df['Y'] <= df['C']).astype(int)  # record the censoring indicator
        df['T'] = df['Delta'] * df['Y'] + (1 - df['Delta']) * df['C']   # record the censored time-to-event


        for i in range(len(df)):  # record the oracle survival prob. for the censoring variable
            if df.loc[i, 'A'] == 0:
                df.loc[i, 'sc'] = self._oracle_adj_surv(df.loc[i, 'T'], df.loc[i, self.cov_list], 'C0')
            else:
                df.loc[i, 'sc'] = self._oracle_adj_surv(df.loc[i, 'T'], df.loc[i, self.cov_list], 'C1')

        df_observed = df[self.cov_list + ['S', 'A', 'T', 'Delta', 'Y0', 'Y1']].copy()

        return df, df_observed
    

    def _sample_x(self):   
        
        df_ihdp = self.og_df[self.px_cols].copy()
        
        if self.n <= self.og_df_len:
            return df_ihdp.values[:self.n]
        else:
            return pd.concat([df_ihdp] * int(np.round(self.n/self.og_df_len)), ignore_index=True).values


    def _calc_propensity(self, X):
        if self.prop_fn == 'sigmoid':
            return sigmoid_fn(X, self.prop_args['beta'])
        else:
            raise NotImplementedError(f'Propensity score method <{self.prop_fn}> is not implemented.')      
    

    def _sample_tte(self, X, var_type):
        if self.tte_params['model'] == 'coxph':
            if self.tte_params['hazard'] == 'weibull':
                tte = sample_weibull_tte(self.n, self.tte_params['cox_args'][var_type], X)
                min_val =  self.tte_params['cox_args'][var_type]['UB']
                return  tte
            else:
                raise NotImplementedError(f'Baseline hazard model <{self.tte_params["hazard"]}> is not implemented.')
            
        else:
            raise NotImplementedError(f'Time-to-event model <{self.tte_params["model"]}> is not implemented.')
        

    def _oracle_adj_surv(self, T, x, arg):
        if self.tte_params['model'] == 'coxph':
            if self.tte_params['hazard'] == 'weibull':
                return weibull_oracle_adj_surv(T, x, self.tte_params['cox_args'][arg])
            else:
                raise NotImplementedError(f'Baseline hazard model <{self.tte_params["hazard"]}> is not implemented.')
            
        else:
            raise NotImplementedError(f'Time-to-event model <{self.tte_params["model"]}> is not implemented.')


    def _ipcw_oracle(self):
        '''
        Oracle IPW-signal calculated for a single instance, using the TRUE values of the nuisance functions.
        '''
        for i in range(len(self.df)):
            row = self.df.loc[i]

            if row['Delta'] == 1:
                part1 = row['A'] / row['prop_score']
                part0 = (1 - row['A']) / (1 - row['prop_score'])
                ipcw = row['T'] * (part1 - part0) / row['sc']

            else:
                ipcw = 0

            self.df.loc[i, 'ipcw_oracle_CATE'] = ipcw
            self.df.loc[i, 'ipcw_oracle_Y1'] = row['A'] * ipcw
            self.df.loc[i, 'ipcw_oracle_Y0'] = - (1 - row['A']) * ipcw


    def _plot(self, save_fig=True, cov_dim=1):
        matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
        matplotlib.rcParams['ps.fonttype'] = 42
        if save_fig: os.makedirs(self.fig_save_dir, exist_ok=True)
 
        #self._plot_px(save_fig, cov_dim)
        self._plot_prop_score(save_fig)
        self._plot_outcomes(save_fig) 
    

    def _plot_px(self, save_fig=True, cov_dim=1):
        plt.figure()
        sns.histplot(data=self.df, x=self.cov_list[cov_dim], kde=True, bins='auto')
        plt.title(f'P(X{cov_dim} | S={self.S})')
        if save_fig:
            plt.savefig(os.path.join(self.fig_save_dir, 'px_hist.png'))
        else:
            plt.show()


    def _plot_prop_score(self, save_fig=True):
        plt.figure()
        sns.histplot(data=self.df, x='prop_score', kde=True, bins='auto')
        if self.S == 1:
            plt.xticks([0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1])
        plt.title('Propensity scores')
        plt.xlabel('P(A=1 | X, S={})'.format(self.df['S'][0].astype(int)))
        plt.ylabel('Count')
        if save_fig:
            plt.savefig(os.path.join(self.fig_save_dir, 'propensity_scores.png'))
        else:
            plt.show()


    def _plot_outcomes(self, save_fig=True):

        plt.figure()
        sns.histplot(data=self.df, x='Y0', kde=True, bins='auto', alpha=0.5, label=f'Y(0) | S={self.S}')
        sns.histplot(data=self.df, x='Y1', kde=True, bins='auto', alpha=0.5, label=f'Y(1) | S={self.S}')
        sns.histplot(data=self.df, x='C0', kde=True, bins='auto', alpha=0.5, label=f'C(0) | S={self.S}')
        sns.histplot(data=self.df, x='C1', kde=True, bins='auto', alpha=0.5, label=f'C(1) | S={self.S}')
        x_lim = max(self.df['Y0'].max(), self.df['Y1'].max(), self.df['C0'].max(), self.df['C1'].max())
        x_lim= 30
        plt.xlim(0, x_lim)
        plt.title('Potential outcome and censoring variables marginalized over X')
        plt.xlabel('Time')
        plt.ylabel('Density')
        plt.legend()
        if save_fig: plt.savefig(os.path.join(self.fig_save_dir, 'Y_C_vars.png'))
        else: plt.show()


    def _save_csv(self,):
        os.makedirs(self.df_save_dir, exist_ok=True)
        self.df.to_csv(os.path.join(self.df_save_dir, 'df.csv'))
        self.df_observed.to_csv(os.path.join(self.df_save_dir, 'df_observed.csv'))
    
   


## real-world data is systematically (selectively) biased, e.g., access to healthcare
## how much data is enough especially, e.g., in the context of studying rare diseases from observational data
## using different databases can help resolve the "positivity" issue (will have implications in terms of transportability tho..)
