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
        Gaussian covariates 
        Logistic propensity scores
        Time-to-events from CoxPH model (both for potential outcomes and censoring time)

        Different choices of data generating processes are not implemented yet, but the structure easily allows it. 
    '''
    def __init__(self, save_df=True, d=1, n=100, S=0,
                    px_dist='Gaussian', px_args={'mean': [0], 'cov': [[1]]},
                    prop_fn='sigmoid', prop_args={'beta': [0, 0]},
                    outcome_model ='survival_function', hazard_fn='weibull',
                    cox_args={'A0': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'A1': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'cen_A0': {'beta': [0,0], 'lambda': 1, 'p': 1},
                                'cen_A1': {'beta': [0,0], 'lambda': 1, 'p': 1},},
                ):

        self.d = d  # covariate dimension (integer)
        self.n = n  # sample size (integer)
        self.S = S  # study indices (integer, 0 reserved for RCT)
        self.px_dist = px_dist  # covariate distribution (string)
        self.px_args = px_args  # parameters for covariate distribution (dict)
        self.prop_fn = prop_fn  # oracle propensity score model, e.g., sigmoid function (string)
        self.prop_args = prop_args  # parameters for the oracle proensity score model (dict)
        self.hazard_fn = hazard_fn  # baseline hazard function, e.g., Weibull (string)
        self.outcome_model = outcome_model  # the model that specifies how the oracle time-to-event variables are sampled (string)
        self.cox_args = cox_args  # parameters for the Cox PH model (dict) 
                                  # (keep first term of beta 0 always to not run into errors later with libraries)
                                  # (same effect can be achieved via lambda&p anyways)

        self.df_save_dir = f'./data/S{self.S}/csv'  # directory to save the DataFrames
        self.fig_save_dir = f'./data/S{self.S}/figures'  # directory to save the figures
        self.cov_list = [f'X{i}' for i in range(d+1)]

        self.df, self.df_observed = self._generate_data()
        if save_df: self._save_csv()

    
    def summary(self, cov_dim=1, plot=True, save_fig=True):
        true_ate = self.df['Y1'].mean() - self.df['Y0'].mean()
        naive_ate = self.df_observed.query('A == 1')['Y'].mean() - self.df_observed.query('A == 0')['Y'].mean()
        impute_ate = self.df_observed.query('A == 1')['T'].mean() - self.df_observed.query('A == 0')['T'].mean()
        drop_ate = self.df_observed.query('Delta == 1 & A == 1')['T'].mean() \
                     - self.df_observed.query('Delta == 1 & A == 0')['T'].mean()
        
        ipw_ate_oracle, _ = get_ate(self.df.copy(), self._ipw_psi_oracle, 'ipw_oracle')
        
        print(f'Study index S: {self.S}\nSample size n: {self.n}\nCovariate dimensionality: {self.d}\n***')
        print(f'True ATE: {true_ate:.3f}\nNo-censoring naive ATE: {naive_ate:.3f}\nCensoring-imputed ATE estimate: {impute_ate:.3f}\nCensoring-dropped ATE estimate: {drop_ate:.3f}\n***')
        print(f'Oracle IPW-estimated ATE: {ipw_ate_oracle:.3f}\n')

        if plot: self._plot(save_fig, cov_dim)

    
    def get_df(self):
        return self.df.copy(), self.df_observed.copy()

    
    def get_covs(self):
        return self.cov_list.copy()
    

    def get_oracle_propensity(self, interval):
        # return the oracle propensity scores in a given interval
        return


    def get_oracle_outcome_model(self, interval, treatment, outcome):
        if self.outcome_model == 'survival_function':
            return self._get_oracle_survival_function(self, interval, treatment, outcome)
        elif self.outcome_model == 'AFT':
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

        df['Y0'] = self._sample_tte(X, 'A0')
        df['Y1'] = self._sample_tte(X, 'A1')
        df['C0'] = self._sample_tte(X, 'cen_A0')
        df['C1'] = self._sample_tte(X, 'cen_A1')
        
        df['Y'] =  df['A'] * df['Y1'] + (1 - df['A']) * df['Y0']  # record the realized outcome
        df['C'] =  df['A'] * df['C1'] + (1 - df['A']) * df['C0']  # record the realized outcome

        df['Delta'] = (df['Y'] <= df['C']).astype(int)  # record the censoring indicator
        df['T'] = df['Delta'] * df['Y'] + (1 - df['Delta']) * df['C']   # record the censored time-to-event

        for i in range(len(df)):  # record the oracle survival prob. for the censoring variable
            if df.loc[i, 'A'] == 0:
                df.loc[i, 'sc'] = self._oracle_adj_surv(df.loc[i, 'T'], df.loc[i, self.cov_list], self.cox_args['cen_A0'])
            else:
                df.loc[i, 'sc'] = self._oracle_adj_surv(df.loc[i, 'T'], df.loc[i, self.cov_list], self.cox_args['cen_A1'])

        df_observed = df[self.cov_list + ['S', 'A', 'T', 'Delta']].copy()

        return df, df_observed
    

    def _sample_x(self):
        if self.px_dist == 'Gaussian':
            return np.random.multivariate_normal(np.array(self.px_args['mean']), np.array(self.px_args['cov']), size=self.n)
        elif self.px_dist == 'pre-specified':
            # to-be-implemented
            # requires another parameter in the constructor that is the covariates themselves
            raise NotImplementedError(f'Handle the pre-specification') 
        else:
            raise NotImplementedError(f'Covariate distribution <{self.px_dist}> is not implemented.') 


    def _calc_propensity(self, X):
        if self.prop_fn == 'sigmoid':
            return sigmoid_fn(X, self.prop_args['beta'])
        else:
            raise NotImplementedError(f'Propensity score method <{self.prop_fn}> is not implemented.')      
    

    def _sample_tte(self, X, type):
        if self.hazard_fn == 'weibull':
            return sample_weibull_tte(self.n, self.cox_args[type], X)
        else:
            raise NotImplementedError(f'Baseline hazard model <{self.hazard_fn}> is not implemented.')
        

    def _oracle_adj_surv(self, T, x, args):
        if self.hazard_fn == 'weibull':
            return weibull_oracle_adj_surv(T, x, args)
        else:
            raise NotImplementedError(f'Baseline hazard model <{self.hazard_fn}> is not implemented.') 


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
        sns.histplot(data=self.df, x='C', kde=True, bins='auto', alpha=0.5, label=f'C | S={self.S}, A=0')
        plt.xlim(0, 10)
        plt.title('Potential outcome and censoring variables marginalized over X')
        plt.xlabel('Time')
        plt.ylabel('Density')
        plt.legend()
        if save_fig: plt.savefig(os.path.join(self.fig_save_dir, 'Y0_and_C.png'))
        else: plt.show()

        plt.figure()
        sns.histplot(data=self.df, x='Y1', kde=True, bins='auto', alpha=0.5, label=f'Y(1) | S={self.S}')
        sns.histplot(data=self.df, x='C', kde=True, bins='auto', alpha=0.5, label=f'C | S={self.S}, A=1')
        plt.xlim(0, 10)
        plt.title('Potential outcome and censoring variables marginalized over X')
        plt.xlabel('Time')
        plt.ylabel('Density')
        plt.legend()
        if save_fig: plt.savefig(os.path.join(self.fig_save_dir, 'Y1_and_C.png'))
        else: plt.show()

    
    def _plot_oracle_surv_curves(self, save_fig):
        df_A0 = self._simulate_single_outcome('Y0')  # oracle-censored data for A = 0
        df_A1 = self._simulate_single_outcome('Y1')  # oracle-censored data for A = 1
        df_A0_obs = self.df.query('A == 0')[self.cov_list[1:] + ['T', 'Delta']].copy()  # observed-censored data for A = 0
        df_A1_obs = self.df.query('A == 1')[self.cov_list[1:] + ['T', 'Delta']].copy()  # observed-censored data for A = 1
        df_cen = self.df[self.cov_list[1:] + ['T', 'Delta']].copy()  # original censored data to estimate the surv. fn. for the censoring var.

        t_A0, est_base_surv_A0, _ = coxph_base_surv(df_A0) 
        t_A1, est_base_surv_A1, _ = coxph_base_surv(df_A1)
        t_obs_A0, est_base_surv_obs_A0, _ = coxph_base_surv(df_A0_obs) 
        t_obs_A1, est_base_surv_obs_A1, _ = coxph_base_surv(df_A1_obs) 
        t_cen, est_base_surv_cen, _ = coxph_base_surv(df_cen, flip=True)  # flip=True to get the surv. fn. of the "censoring"

        if self.hazard_fn == 'weibull':
            true_base_surv_A0 = np.exp(-((self.cox_args['A0']['lambda'] * t_A0) ** self.cox_args['A0']['p'])) 
            true_base_surv_A1 = np.exp(-((self.cox_args['A1']['lambda'] * t_A1) ** self.cox_args['A1']['p'])) 
            true_base_surv_cen = np.exp(-((self.cox_args['cen_A0']['lambda'] * t_cen) ** self.cox_args['cen_A0']['p'])) 

        plt.figure()
        plt.plot(t_A0, true_base_surv_A0, label='True baseline S(t)', ls='--', lw=3)
        plt.plot(t_A0, est_base_surv_A0, label='Est. oracle baseline S(t) (used oracle Y0)')
        plt.plot(t_obs_A0, est_base_surv_obs_A0, label='Est. baseline S(t) (used A = 0 only)')
        plt.title('Survival estimates with oracle and observed outcomes for A = 0')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$S(t)$')
        plt.legend()

        if save_fig: plt.savefig(os.path.join(self.fig_save_dir, 'surv_plots_A0.png'))
        else: plt.show()

        plt.figure()
        plt.plot(t_A1, true_base_surv_A1, label='True baseline S(t)', ls='--', lw=3)
        plt.plot(t_A1, est_base_surv_A1, label='Est. oracle baseline S(t) (used oracle Y1)')
        plt.plot(t_obs_A1, est_base_surv_obs_A1, label='Est. baseline S(t) (used A = 1 only)')
        plt.title('Survival estimates with oracle and observed outcomes for A = 1')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$S(t)$')
        plt.legend()

        if save_fig: plt.savefig(os.path.join(self.fig_save_dir, 'surv_plots_A1.png'))
        else: plt.show()

        plt.figure()
        plt.plot(t_cen, est_base_surv_cen, label='Estimated baseline survival C', ls='--', lw=3)
        plt.plot(t_cen, true_base_surv_cen, label='True baseline survival C')
        plt.title('Survival estimate for the censoring variable')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$S(t)$')
        plt.legend()
        
        if save_fig: plt.savefig(os.path.join(self.fig_save_dir, 'surv_plot_cen.png'))
        else: plt.show()


    def _simulate_single_outcome(self, outcome):  # this method returns an oracle dataframe on the censored outcomes for the potential outcome (Y0 or Y1), if it was always observed
        df = self.df.copy()  # exclude X0, as CoxPHFitter throws an error for constant columns
        df['Delta'] = (df[outcome] <= df['C']).astype(int)
        df['T'] = df['Delta'] * df[outcome] + (1 - df['Delta']) *  df['C']

        return df[self.cov_list[1:] + ['T', 'Delta']]


    def _save_csv(self,):
        os.makedirs(self.df_save_dir, exist_ok=True)
        self.df.to_csv(os.path.join(self.df_save_dir, 'df.csv'))
        self.df_observed.to_csv(os.path.join(self.df_save_dir, 'df_observed.csv'))
    
   


## real-world data is systematically (selectively) biased, e.g., access to healthcare
## how much data is enough especially, e.g., in the context of studying rare diseases from observational data
## using different databases can help resolve the "positivity" issue (will have implications in terms of transportability tho..)
