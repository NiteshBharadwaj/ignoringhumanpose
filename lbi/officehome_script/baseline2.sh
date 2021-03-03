#!/bin/bash


python main.py --save_dir=ArCl_baseline2 --gpu=0 --source_domain=Ar --target_domain=Cl --baseline2
python main.py --save_dir=ArPr_baseline2 --gpu=0 --source_domain=Ar --target_domain=Pr --baseline2
python main.py --save_dir=ArRw_baseline2 --gpu=0 --source_domain=Ar --target_domain=Rw --baseline2
python main.py --save_dir=ClAr_baseline2 --gpu=0 --source_domain=Cl --target_domain=Ar --baseline2
python main.py --save_dir=ClPr_baseline2 --gpu=0 --source_domain=Cl --target_domain=Pr --baseline2
python main.py --save_dir=ClRw_baseline2 --gpu=0 --source_domain=Cl --target_domain=Rw --baseline2
python main.py --save_dir=PrAr_baseline2 --gpu=0 --source_domain=Pr --target_domain=Ar --baseline2
python main.py --save_dir=PrCl_baseline2 --gpu=0 --source_domain=Pr --target_domain=Cl --baseline2
python main.py --save_dir=PrRw_baseline2 --gpu=0 --source_domain=Pr --target_domain=Rw --baseline2
python main.py --save_dir=RwAr_baseline2 --gpu=0 --source_domain=Rw --target_domain=Ar --baseline2
python main.py --save_dir=RwCl_baseline2 --gpu=0 --source_domain=Rw --target_domain=Cl --baseline2
python main.py --save_dir=RwPr_baseline2 --gpu=0 --source_domain=Rw --target_domain=Pr --baseline2