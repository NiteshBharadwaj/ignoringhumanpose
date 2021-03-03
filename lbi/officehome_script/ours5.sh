#!/bin/bash


python main.py --save_dir=ArCl_ours5 --gpu=0 --source_domain=Ar --target_domain=Cl --ours5 --lam=7e-3
python main.py --save_dir=ArPr_ours5 --gpu=0 --source_domain=Ar --target_domain=Pr --ours5 --lam=7e-3
python main.py --save_dir=ArRw_ours5 --gpu=0 --source_domain=Ar --target_domain=Rw --ours5 --lam=7e-3
python main.py --save_dir=ClAr_ours5 --gpu=0 --source_domain=Cl --target_domain=Ar --ours5 --lam=7e-3
python main.py --save_dir=ClPr_ours5 --gpu=0 --source_domain=Cl --target_domain=Pr --ours5 --lam=7e-3
python main.py --save_dir=ClRw_ours5 --gpu=0 --source_domain=Cl --target_domain=Rw --ours5 --lam=7e-3
python main.py --save_dir=PrAr_ours5 --gpu=0 --source_domain=Pr --target_domain=Ar --ours5 --lam=7e-3
python main.py --save_dir=PrCl_ours5 --gpu=0 --source_domain=Pr --target_domain=Cl --ours5 --lam=7e-3
python main.py --save_dir=PrRw_ours5 --gpu=0 --source_domain=Pr --target_domain=Rw --ours5 --lam=7e-3
python main.py --save_dir=RwAr_ours5 --gpu=0 --source_domain=Rw --target_domain=Ar --ours5 --lam=7e-3
python main.py --save_dir=RwCl_ours5 --gpu=0 --source_domain=Rw --target_domain=Cl --ours5 --lam=7e-3
python main.py --save_dir=RwPr_ours5 --gpu=0 --source_domain=Rw --target_domain=Pr --ours5 --lam=7e-3