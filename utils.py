import numpy as np
import pandas as pd

def sigmoid_fn(X, beta):
    return 1/(1 + np.exp(- X @ beta))

def sample_weibull_tte(n, args, x):
    '''
    Sample Weibull time-to-event outcomes.
    see: https://web.stanford.edu/~lutian/coursepdf/unit1.pdf
        h(t | x) = p * (lambda_ ** p) * (t ** (p-1)) * exp(beta*x)
        S(t | x) = exp(-((lambda_ * t) ** p) * exp(beta*x))

    @params:
        n: number of samples (integer)
        x: covariates (list)
        args: Cox PH model parameters (dict)
    '''

    C = np.exp(x @ args['beta'])
    U = np.random.uniform(0, 1, n)
    return (-np.log(U) / C) ** (1 / args['p']) * (1 / args['lambda'])


def weibull_oracle_adj_surv(T, x, args):
    '''
    see: https://web.stanford.edu/~lutian/coursepdf/unit1.pdf
        S(t | x) = exp(-((lambda_ * t) ** p) * exp(beta*x))
    '''

    return np.exp(-((args['lambda'] * T) ** args['p']) * np.exp(x @ args['beta']))


def get_ate(df, psi, sig_name):
    '''
    estimate ATE from an instance-wise signal

    @params:
        df: pd.Dataframe
        psi: function that operates on a single row and calculates an instance-wise signal for the CATE (callable)
        sig_name: name of the signal for logging it in the dataframe (string)
    '''
    
    for i in range(len(df)):
        row = df.loc[i]
        df.loc[i, sig_name] = psi(row)

    return np.mean(df[sig_name]), df

