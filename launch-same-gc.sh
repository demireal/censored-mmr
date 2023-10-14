#!/usr/bin/env bash

python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=CDR
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=IPCW
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Impute
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Drop
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Impute
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Drop

python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=3 --signals=CDR
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Drop
