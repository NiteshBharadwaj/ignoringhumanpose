#!/bin/bash


python main.py --save_dir=ArCl_ours2 --gpu=0 --source_domain=Ar --target_domain=Cl --ours2 --lam=7e-3
python main.py --save_dir=ArPr_ours2 --gpu=0 --source_domain=Ar --target_domain=Pr --ours2 --lam=7e-3
python main.py --save_dir=ArRw_ours2 --gpu=0 --source_domain=Ar --target_domain=Rw --ours2 --lam=7e-3
python main.py --save_dir=ClAr_ours2 --gpu=0 --source_domain=Cl --target_domain=Ar --ours2 --lam=7e-3
python main.py --save_dir=ClPr_ours2 --gpu=0 --source_domain=Cl --target_domain=Pr --ours2 --lam=7e-3
python main.py --save_dir=ClRw_ours2 --gpu=0 --source_domain=Cl --target_domain=Rw --ours2 --lam=7e-3
python main.py --save_dir=PrAr_ours2 --gpu=0 --source_domain=Pr --target_domain=Ar --ours2 --lam=7e-3
python main.py --save_dir=PrCl_ours2 --gpu=0 --source_domain=Pr --target_domain=Cl --ours2 --lam=7e-3
python main.py --save_dir=PrRw_ours2 --gpu=0 --source_domain=Pr --target_domain=Rw --ours2 --lam=7e-3
python main.py --save_dir=RwAr_ours2 --gpu=0 --source_domain=Rw --target_domain=Ar --ours2 --lam=7e-3
python main.py --save_dir=RwCl_ours2 --gpu=0 --source_domain=Rw --target_domain=Cl --ours2 --lam=7e-3
python main.py --save_dir=RwPr_ours2 --gpu=0 --source_domain=Rw --target_domain=Pr --ours2 --lam=7e-3