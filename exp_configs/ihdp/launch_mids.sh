#!/bin/bash

python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=IPCW
python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=CDR
python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=CDR-MissG
python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=CDR-MissF
python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=IPW-Impute
python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=IPW-Drop
python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=DR-Impute
python main.py --json_path=ihdp/diffPOihdp-midhighdiff.json --UC=0 --CD=10 --M=3 --signals=DR-Drop

python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=IPCW
python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=CDR
python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=CDR-MissG
python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=CDR-MissF
python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=IPW-Impute
python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=IPW-Drop
python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=DR-Impute
python main.py --json_path=ihdp/diffPOihdp-midlowdiff.json --UC=0 --CD=10 --M=3 --signals=DR-Drop

python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=IPCW
python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=CDR
python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=CDR-MissG
python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=CDR-MissF
python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=IPW-Impute
python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=IPW-Drop
python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=DR-Impute
python main.py --json_path=ihdp/samePOihdp-midhighhc.json --UC=1 --CD=10 --M=3 --signals=DR-Drop

python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=IPCW
python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=CDR
python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=CDR-MissG
python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=CDR-MissF
python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=IPW-Impute
python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=IPW-Drop
python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=DR-Impute
python main.py --json_path=ihdp/samePOihdp-midlowhc.json --UC=1 --CD=10 --M=3 --signals=DR-Drop