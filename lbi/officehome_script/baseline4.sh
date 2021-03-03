#!/bin/bash


python main.py --save_dir=ArCl_baseline4 --gpu=0 --source_domain=Ar --target_domain=Cl --baseline4 --lam=7e-3
python main.py --save_dir=ArPr_baseline4 --gpu=0 --source_domain=Ar --target_domain=Pr --baseline4 --lam=7e-3
python main.py --save_dir=ArRw_baseline4 --gpu=0 --source_domain=Ar --target_domain=Rw --baseline4 --lam=7e-3
python main.py --save_dir=ClAr_baseline4 --gpu=0 --source_domain=Cl --target_domain=Ar --baseline4 --lam=7e-3
python main.py --save_dir=ClPr_baseline4 --gpu=0 --source_domain=Cl --target_domain=Pr --baseline4 --lam=7e-3
python main.py --save_dir=ClRw_baseline4 --gpu=0 --source_domain=Cl --target_domain=Rw --baseline4 --lam=7e-3
python main.py --save_dir=PrAr_baseline4 --gpu=0 --source_domain=Pr --target_domain=Ar --baseline4 --lam=7e-3
python main.py --save_dir=PrCl_baseline4 --gpu=0 --source_domain=Pr --target_domain=Cl --baseline4 --lam=7e-3
python main.py --save_dir=PrRw_baseline4 --gpu=0 --source_domain=Pr --target_domain=Rw --baseline4 --lam=7e-3
python main.py --save_dir=RwAr_baseline4 --gpu=0 --source_domain=Rw --target_domain=Ar --baseline4 --lam=7e-3
python main.py --save_dir=RwCl_baseline4 --gpu=0 --source_domain=Rw --target_domain=Cl --baseline4 --lam=7e-3
python main.py --save_dir=RwPr_baseline4 --gpu=0 --source_domain=Rw --target_domain=Pr --baseline4 --lam=7e-3