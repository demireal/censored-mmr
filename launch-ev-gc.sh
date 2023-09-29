#!/bin/bash

python3 main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=CDR

python3 main.py --json_path=ihdp/diffPOihdp-midlowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/diffPOihdp-midlowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/diffPOihdp-midlowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/diffPOihdp-midlowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/diffPOihdp-midlowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/diffPOihdp-midlowdiff_GC.json --UC=0 --CD=10 --M=3 --signals=CDR

python3 main.py --json_path=ihdp/diffPOihdp-midhighdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/diffPOihdp-midhighdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/diffPOihdp-midhighdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/diffPOihdp-midhighdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/diffPOihdp-midhighdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/diffPOihdp-midhighdiff_GC.json --UC=0 --CD=10 --M=3 --signals=CDR

python3 main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPCW
python3 main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Impute
python3 main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=3 --signals=IPW-Drop
python3 main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Impute
python3 main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=3 --signals=DR-Drop
python3 main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=3 --signals=CDR