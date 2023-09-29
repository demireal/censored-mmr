#!/bin/bash

python3 main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=3 --signals=CDR

python3 main.py --json_path=ihdp/samePOihdp-midlowhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/samePOihdp-midlowhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/samePOihdp-midlowhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/samePOihdp-midlowhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/samePOihdp-midlowhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/samePOihdp-midlowhc_GC.json --UC=1 --CD=10 --M=3 --signals=CDR

python3 main.py --json_path=ihdp/samePOihdp-midhighhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/samePOihdp-midhighhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/samePOihdp-midhighhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/samePOihdp-midhighhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/samePOihdp-midhighhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/samePOihdp-midhighhc_GC.json --UC=1 --CD=10 --M=3 --signals=CDR

python3 main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=3 --signals=CDR