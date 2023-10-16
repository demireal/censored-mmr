# Benchmarking Observational Studies with Experimental Data under Right-Censoring

### IHDP Experiments

All experiments can be run using the same `main.py` script. We use json config files to run specific experiments.

The arguments are

`--json_path` : the path to the config for data generation and test computation. All tests are in the `exp_configs` directory.

`--CD` : the number of covariates to use

`--UC` : the number of hidden confounders in the treatment response

`--M` : the ratio of the OS sample size with respect to the RCT sample size.

`--signals` : the signals to use for the test. Options include `IPCW`, `CDR`, `CDR-MissG`, `CDR-MissF`, `IPW-Impute`, `IPW-Drop`, `DR-Impute`, `DR-Drop`.

For instance, to run experiments on the IHDP dataset with same potential outcomes and global censoring, with CDR signals:

`python main.py --json_path=exp_configs/ihdp/samePOihdp_GC.json --CD=10 --UC=0 --M=3 --signals=CDR`


### WHI Experiments




