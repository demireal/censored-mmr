import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from utils import *


class SyntheticDataModule:
    '''
    Current version generates a dataframe with:
        Gaussian covariates X
        Logistic propensity scores P(A=1 | X)
        Time-to-event variables from a CoxPH model (both for potential outcomes Y0 Y1 and censoring times C0 C1)

        Different models for data generating (e.g. AFT model) are not implemented yet, but their inregration should be easy. 
    '''
    def __init__(self, save_df=True, d=1, n=100, S=0,
                    px_dist='Gaussian', px_args={'mean': [0], 'cov': [[1]]},
                    prop_fn='sigmoid', prop_args={'beta': [0, 0]},
                    tte_params =
                    {'model': 'coxph',
                    'hazard': 'weibull',
                    'cox_args': {'Y0': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'Y1': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'C0': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'C1': {'beta': [0,0], 'lambda': 1, 'p': 1},},
                    }
                ):

        self.d = d  # covariate dimension (integer)
        self.n = n  # sample size (integer)
        self.S = S  # study indices (integer, 0 reserved for RCT)
        self.px_dist = px_dist  # covariate distribution (string)
        self.px_args = px_args  # parameters for covariate distribution (dict)
        self.prop_fn = prop_fn  # oracle propensity score model, e.g., sigmoid function (string)
        self.prop_args = prop_args  # parameters for the oracle proensity score model (dict)
        self.tte_params = tte_params  # the model that specifies how the oracle time-to-event variables are generated (string)
                                    # (for Cox model, keep first term of beta 0 always to not run into errors later with libraries)
                                    # (same effect can be achieved via lambda&p anyways)

        self.df_save_dir = f'./data/S{self.S}/csv'  # directory to save the DataFrames
        self.fig_save_dir = f'./data/S{self.S}/figures'  # directory to save the figures
        self.cov_list = [f'X{i}' for i in range(d+1)]

        self.df, self.df_observed = self._generate_data()
        if save_df: self._save_csv()

    
    def summary(self, cov_dim=1, plot=True, save_fig=True):
        true_ate = self.df['Y1'].mean() - self.df['Y0'].mean()
        naive_ate = self.df.query('A == 1')['Y1'].mean() - self.df.query('A == 0')['Y0'].mean()
        impute_ate = self.df_observed.query('A == 1')['T'].mean() - self.df_observed.query('A == 0')['T'].mean()
        drop_ate = self.df_observed.query('Delta == 1 & A == 1')['T'].mean() \
                     - self.df_observed.query('Delta == 1 & A == 0')['T'].mean()
        
        ipw_ate_oracle, _ = get_ate(self.df.copy(), self._ipw_psi_oracle, 'ipw_oracle')
        
        print(f'Study index S: {self.S}\nSample size n: {self.n}\nCovariate dimensionality: {self.d}\n***')
        print(f'True ATE: {true_ate:.3f}\nNo-censoring naive ATE estimate: {naive_ate:.3f}\nCensoring-imputed ATE estimate: {impute_ate:.3f}\nCensoring-dropped ATE estimate: {drop_ate:.3f}\n***')
        print(f'Oracle IPW-estimated ATE: {ipw_ate_oracle:.3f}\n')

        if plot: self._plot(save_fig, cov_dim)

    
    def get_df(self):
        return self.df.copy(), self.df_observed.copy()

    
    def get_covs(self):
        return self.cov_list.copy()


    def get_oracle_outcome_model(self, interval, treatment, outcome):
        if self.tte_params['model'] == 'coxph':
            return self._get_oracle_survival_function(self, interval, treatment, outcome)
        elif self.tte_params['model'] == 'AFT':
            return self._get_oracle_mean_response_surface(self, interval, treatment, outcome)
        

    def _get_oracle_survival_function(self, interval, treatment, outcome):
        # return the oracle survival function in INTERVAL for OUTCOME (C, Y) in TREATMENT arm (0,1) 
        return
    

    def _get_oracle_mean_response_surface(self, interval, treatment, outcome):
        # return the oracle mean response surface signal in INTERVAL for OUTCOME (C, Y) in TREATMENT arm (0,1) 
        return
    

    def _generate_data(self):       
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
        
        df['Y'] =  df['A'] * df['Y1'] + (1 - df['A']) * df['Y0']  # record the realized potential event time 
        df['C'] =  df['A'] * df['C1'] + (1 - df['A']) * df['C0']  # record the realized potential censoring time

        df['Delta'] = (df['Y'] <= df['C']).astype(int)  # record the censoring indicator
        df['T'] = df['Delta'] * df['Y'] + (1 - df['Delta']) * df['C']   # record the censored time-to-event


        for i in range(len(df)):  # record the oracle survival prob. for the censoring variable
            if df.loc[i, 'A'] == 0:
                df.loc[i, 'sc'] = self._oracle_adj_surv(df.loc[i, 'T'], df.loc[i, self.cov_list], 'C0')
            else:
                df.loc[i, 'sc'] = self._oracle_adj_surv(df.loc[i, 'T'], df.loc[i, self.cov_list], 'C1')

        df_observed = df[self.cov_list + ['S', 'A', 'T', 'Delta']].copy()

        return df, df_observed
    

    def _sample_x(self):
        if self.px_dist == 'Gaussian':
            return np.random.multivariate_normal(np.array(self.px_args['mean']), np.array(self.px_args['cov']), size=self.n)
        elif self.px_dist == 'pre-specified':
            # to-be-implemented
            # requires another parameter in the constructor that contains the covariates themselves
            raise NotImplementedError(f'Handle the pre-specification') 
        else:
            raise NotImplementedError(f'Covariate distribution <{self.px_dist}> is not implemented.') 


    def _calc_propensity(self, X):
        if self.prop_fn == 'sigmoid':
            return sigmoid_fn(X, self.prop_args['beta'])
        else:
            raise NotImplementedError(f'Propensity score method <{self.prop_fn}> is not implemented.')      
    

    def _sample_tte(self, X, type):
        if self.tte_params['model'] == 'coxph':
            if self.tte_params['hazard'] == 'weibull':
                return sample_weibull_tte(self.n, self.tte_params['cox_args'][type], X)
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


    def _ipw_psi_oracle(self, row):
        '''
        Oracle IPW-signal calculated for a single instance, using the TRUE values of the nuisance functions.
        '''
        if row['Delta'] == 1:
            part1 = int(row['A']==1) / (row['prop_score'])
            part2 = int(row['A']==0) / (1-row['prop_score'])

            ipw_psi = row['T'] * (part1 - part2) / row['sc']
        else:
            ipw_psi = 0

        return ipw_psi


    def _plot(self, save_fig=True, cov_dim=1):
        matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
        matplotlib.rcParams['ps.fonttype'] = 42
        if save_fig: os.makedirs(self.fig_save_dir, exist_ok=True)
 
        self._plot_px(save_fig, cov_dim)
        self._plot_prop_score(save_fig)
        self._plot_outcomes(save_fig) 
        #self._plot_oracle_surv_curves(save_fig) 
    

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
        plt.xlim(0, 10)
        plt.title('Potential outcome and censoring variables marginalized over X')
        plt.xlabel('Time')
        plt.ylabel('Density')
        plt.legend()
        if save_fig: plt.savefig(os.path.join(self.fig_save_dir, 'Y_C_vars.png'))
        else: plt.show()


    def calc_oracle_prop(self, arr, covariate='X1'):
        X = pd.DataFrame(columns=self.cov_list, index=np.arange(len(arr)))
        X[self.cov_list] = 1
        X[covariate] = arr
        if self.prop_fn == 'sigmoid':
            return sigmoid_fn(X, self.prop_args['beta'])
        else:
            raise NotImplementedError(f'Propensity score method <{self.prop_fn}> is not implemented.')  

    
    # def calc_oracle_surv_curves(self, arr, cov_vals):
    #     tbs_Y0, tbs_Y1, tbs_C0, tbs_C1 = np.zeros(len(arr)), np.zeros(len(arr)), np.zeros(len(arr)), np.zeros(len(arr))
    #     if self.tte_params['model'] == 'coxph':
    #         if self.tte_params['hazard'] == 'weibull':
    #             for i in range(len(arr)):
    #                 tbs_Y0[i] = weibull_oracle_adj_surv(arr[i], np.array(cov_vals), self.tte_params['cox_args']['Y0'])
    #                 tbs_Y1[i] = weibull_oracle_adj_surv(arr[i], np.array(cov_vals), self.tte_params['cox_args']['Y1'])
    #                 tbs_C0[i] = weibull_oracle_adj_surv(arr[i], np.array(cov_vals), self.tte_params['cox_args']['C0'])
    #                 tbs_C1[i] = weibull_oracle_adj_surv(arr[i], np.array(cov_vals), self.tte_params['cox_args']['C1'])

    #             return tbs_Y0, tbs_Y1, tbs_C0, tbs_C1
            
    #         else:
    #             raise NotImplementedError(f'Baseline hazard model <{self.tte_params["hazard"]}> is not implemented.')
            
    #     else:
    #         raise NotImplementedError(f'Time-to-event model <{self.tte_params["model"]}> is not implemented.')


    def calc_oracle_surv_curves(self, arr, cov_vals):
        if self.tte_params['model'] == 'coxph':
            if self.tte_params['hazard'] == 'weibull':
                tbs_Y0 = weibull_oracle_adj_surv(arr, np.array(cov_vals), self.tte_params['cox_args']['Y0'])
                tbs_Y1 = weibull_oracle_adj_surv(arr, np.array(cov_vals), self.tte_params['cox_args']['Y1'])
                tbs_C0 = weibull_oracle_adj_surv(arr, np.array(cov_vals), self.tte_params['cox_args']['C0'])
                tbs_C1 = weibull_oracle_adj_surv(arr, np.array(cov_vals), self.tte_params['cox_args']['C1'])

                return tbs_Y0, tbs_Y1, tbs_C0, tbs_C1
            
            else:
                raise NotImplementedError(f'Baseline hazard model <{self.tte_params["hazard"]}> is not implemented.')
            
        else:
            raise NotImplementedError(f'Time-to-event model <{self.tte_params["model"]}> is not implemented.')


    def _save_csv(self,):
        os.makedirs(self.df_save_dir, exist_ok=True)
        self.df.to_csv(os.path.join(self.df_save_dir, 'df.csv'))
        self.df_observed.to_csv(os.path.join(self.df_save_dir, 'df_observed.csv'))
    
   


## real-world data is systematically (selectively) biased, e.g., access to healthcare
## how much data is enough especially, e.g., in the context of studying rare diseases from observational data
## using different databases can help resolve the "positivity" issue (will have implications in terms of transportability tho..)
