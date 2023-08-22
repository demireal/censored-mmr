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



def write_dict_to_file(dictionary, indent=0, file=None):
    if file is None:
        raise ValueError("File object is required.")

    for key, value in dictionary.items():
        if isinstance(value, dict):
            file.write(f"{' ' * indent}{key}:\n")
            write_dict_to_file(value, indent + 4, file)  # Recursive call for nested dictionaries
        else:
            file.write(f"{' ' * indent}{key}: {value}\n")
            
            
def readme_summary(readme_path, args, 
                      px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r,
                      px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o,):
    
    with open(readme_path, 'w') as file:
        file.write(f'RCT size: {args.rct_size}\n')
        file.write(f'Cov. dim.: {args.dim}\n')
        file.write(f'Max. m: {args.maxm}\n')
        file.write(f'B: {args.B}\n')
        file.write(f'Num exp: {args.num_exp}\n\n')

        file.write('DATA GENERATING PARAMETERS\n\n')

        file.write('-'*20 + '\nRCT\n' + '-'*20 + '\n\n')

        file.write(f'Covariate dist. P(X): {px_dist_r} with\n')
        write_dict_to_file(px_args_r, file=file)

        file.write(f'\nPropensity score P(A=1|X): {prop_fn_r} with\n')
        write_dict_to_file(prop_args_r, file=file)

        file.write(f'\nTime-to-event generation\n')
        write_dict_to_file(tte_params_r, file=file)

        file.write('\n\n' + '-'*20 + '\nOBS\n' + '-'*20 + '\n\n')

        file.write(f'Covariate dist. P(X): {px_dist_o} with\n')
        write_dict_to_file(px_args_o, file=file)

        file.write(f'\nPropensity score P(A=1|X): {prop_fn_o} with\n')
        write_dict_to_file(prop_args_o, file=file)

        file.write(f'\nTime-to-event generation\n')
        write_dict_to_file(tte_params_o, file=file)
            
            
