#!/bin/bash

python main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=IPCW
python main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=CDR
python main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Impute
python main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Drop
python main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Impute
python main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Drop
python main.py --json_path=ihdp/samePOihdp_GC.json --UC=0 --CD=10 --M=1 --signals=IPCW_GC

python main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPCW
python main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=1 --signals=CDR
python main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPW-Impute
python main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPW-Drop
python main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=1 --signals=DR-Impute
python main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=1 --signals=DR-Drop
python main.py --json_path=ihdp/samePOihdp-highhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPCW_GC

python main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPCW
python main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=1 --signals=CDR
python main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPW-Impute
python main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPW-Drop
python main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=1 --signals=DR-Impute
python main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=1 --signals=DR-Drop
python main.py --json_path=ihdp/samePOihdp-lowhc_GC.json --UC=1 --CD=10 --M=1 --signals=IPCW_GC

python main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPCW
python main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=1 --signals=CDR
python main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Impute
python main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Drop
python main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Impute
python main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Drop
python main.py --json_path=ihdp/diffPOihdp-highdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPCW_GC

python main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPCW
python main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=1 --signals=CDR
python main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Impute
python main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPW-Drop
python main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Impute
python main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=1 --signals=DR-Drop
python main.py --json_path=ihdp/diffPOihdp-lowdiff_GC.json --UC=0 --CD=10 --M=1 --signals=IPCW_GC