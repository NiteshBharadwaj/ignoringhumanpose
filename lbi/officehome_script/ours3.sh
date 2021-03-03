#!/bin/bash


python main.py --save_dir=ArCl_ours3 --gpu=0 --source_domain=Ar --target_domain=Cl --ours3 --lam=7e-3
python main.py --save_dir=ArPr_ours3 --gpu=0 --source_domain=Ar --target_domain=Pr --ours3 --lam=7e-3
python main.py --save_dir=ArRw_ours3 --gpu=0 --source_domain=Ar --target_domain=Rw --ours3 --lam=7e-3
python main.py --save_dir=ClAr_ours3 --gpu=0 --source_domain=Cl --target_domain=Ar --ours3 --lam=7e-3
python main.py --save_dir=ClPr_ours3 --gpu=0 --source_domain=Cl --target_domain=Pr --ours3 --lam=7e-3
python main.py --save_dir=ClRw_ours3 --gpu=0 --source_domain=Cl --target_domain=Rw --ours3 --lam=7e-3
python main.py --save_dir=PrAr_ours3 --gpu=0 --source_domain=Pr --target_domain=Ar --ours3 --lam=7e-3
python main.py --save_dir=PrCl_ours3 --gpu=0 --source_domain=Pr --target_domain=Cl --ours3 --lam=7e-3
python main.py --save_dir=PrRw_ours3 --gpu=0 --source_domain=Pr --target_domain=Rw --ours3 --lam=7e-3
python main.py --save_dir=RwAr_ours3 --gpu=0 --source_domain=Rw --target_domain=Ar --ours3 --lam=7e-3
python main.py --save_dir=RwCl_ours3 --gpu=0 --source_domain=Rw --target_domain=Cl --ours3 --lam=7e-3
python main.py --save_dir=RwPr_ours3 --gpu=0 --source_domain=Rw --target_domain=Pr --ours3 --lam=7e-3