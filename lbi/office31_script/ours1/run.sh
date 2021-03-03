#!/bin/bash

python main.py --save_dir=A_D_ours1 --gpu=0 --source_domain=A --target_domain=D --dataset=office31 --ours1
python main.py --save_dir=A_W_ours1 --gpu=0 --source_domain=A --target_domain=W --dataset=office31 --ours1
python main.py --save_dir=D_A_ours1 --gpu=0 --source_domain=D --target_domain=A --dataset=office31 --ours1
python main.py --save_dir=D_W_ours1 --gpu=0 --source_domain=D --target_domain=W --dataset=office31 --ours1
python main.py --save_dir=W_A_ours1 --gpu=0 --source_domain=W --target_domain=A --dataset=office31 --ours1
python main.py --save_dir=W_D_ours1 --gpu=0 --source_domain=W --target_domain=D --dataset=office31 --ours1
