#!/bin/bash


python main.py --save_dir=ArCl_baseline1 --gpu=0 --source_domain=Ar --target_domain=Cl --baseline1
python main.py --save_dir=ArPr_baseline1 --gpu=0 --source_domain=Ar --target_domain=Pr --baseline1
python main.py --save_dir=ArRw_baseline1 --gpu=0 --source_domain=Ar --target_domain=Rw --baseline1
python main.py --save_dir=ClAr_baseline1 --gpu=0 --source_domain=Cl --target_domain=Ar --baseline1
python main.py --save_dir=ClPr_baseline1 --gpu=0 --source_domain=Cl --target_domain=Pr --baseline1
python main.py --save_dir=ClRw_baseline1 --gpu=0 --source_domain=Cl --target_domain=Rw --baseline1
python main.py --save_dir=PrAr_baseline1 --gpu=0 --source_domain=Pr --target_domain=Ar --baseline1
python main.py --save_dir=PrCl_baseline1 --gpu=0 --source_domain=Pr --target_domain=Cl --baseline1
python main.py --save_dir=PrRw_baseline1 --gpu=0 --source_domain=Pr --target_domain=Rw --baseline1
python main.py --save_dir=RwAr_baseline1 --gpu=0 --source_domain=Rw --target_domain=Ar --baseline1
python main.py --save_dir=RwCl_baseline1 --gpu=0 --source_domain=Rw --target_domain=Cl --baseline1
python main.py --save_dir=RwPr_baseline1 --gpu=0 --source_domain=Rw --target_domain=Pr --baseline1