#!/bin/bash

python main.py --save_dir=A_D_baseline3 --gpu=0 --source_domain=A --target_domain=D --dataset=office31 --baseline3 --lam=5e-4
python main.py --save_dir=A_W_baseline3 --gpu=0 --source_domain=A --target_domain=W --dataset=office31 --baseline3 --lam=5e-4
python main.py --save_dir=D_A_baseline3 --gpu=0 --source_domain=D --target_domain=A --dataset=office31 --baseline3 --lam=5e-4
python main.py --save_dir=D_W_baseline3 --gpu=0 --source_domain=D --target_domain=W --dataset=office31 --baseline3 --lam=5e-4
python main.py --save_dir=W_A_baseline3 --gpu=0 --source_domain=W --target_domain=A --dataset=office31 --baseline3 --lam=5e-4
python main.py --save_dir=W_D_baseline3 --gpu=0 --source_domain=W --target_domain=D --dataset=office31 --baseline3 --lam=5e-4
