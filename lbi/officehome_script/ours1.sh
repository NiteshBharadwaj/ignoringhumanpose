#!/bin/bash


python main.py --save_dir=ArCl_ours1 --gpu=0 --source_domain=Ar --target_domain=Cl --ours1
python main.py --save_dir=ArPr_ours1 --gpu=0 --source_domain=Ar --target_domain=Pr --ours1
python main.py --save_dir=ArRw_ours1 --gpu=0 --source_domain=Ar --target_domain=Rw --ours1
python main.py --save_dir=ClAr_ours1 --gpu=0 --source_domain=Cl --target_domain=Ar --ours1
python main.py --save_dir=ClPr_ours1 --gpu=0 --source_domain=Cl --target_domain=Pr --ours1
python main.py --save_dir=ClRw_ours1 --gpu=0 --source_domain=Cl --target_domain=Rw --ours1
python main.py --save_dir=PrAr_ours1 --gpu=0 --source_domain=Pr --target_domain=Ar --ours1
python main.py --save_dir=PrCl_ours1 --gpu=0 --source_domain=Pr --target_domain=Cl --ours1
python main.py --save_dir=PrRw_ours1 --gpu=0 --source_domain=Pr --target_domain=Rw --ours1
python main.py --save_dir=RwAr_ours1 --gpu=0 --source_domain=Rw --target_domain=Ar --ours1
python main.py --save_dir=RwCl_ours1 --gpu=0 --source_domain=Rw --target_domain=Cl --ours1
python main.py --save_dir=RwPr_ours1 --gpu=0 --source_domain=Rw --target_domain=Pr --ours1