# Readme

## Learning by ignoring

---

Code for paper Learning by ignoring

### Environment

---

The code is tested on

- PyTorch 1.40
- Python 3

Other dependencies:

- numpy `pip install numpy`
- higher `pip install higher`

### Note

---

- Dataset is located in temp folder.
- Results, saved models, and logs are located in results folder.
- To reproduce the results showed in the paper, the batch size should be 64. However, batch size 64 will take about 11~14gb gpu memory. To reduce the memory usage, you can change batch size to, for example, 32 by adding an argument `—-batch_size=32` .

### Settings

---

**Pretraining Data(Source), Finetuning Data(Target)**

- baseline1: train on target
- baseline2: train on source and target
- baseline3: train on target with regularization from source
- baseline4: train on source and target with regularization from source
- ours1: train on reweighted source and target
- ours2: train on target with regularization from reweighted source
- ours3: train on reweighted source and target with regularization from source
- ours4: train on source and target with regularization from reweighted source
- ours5: train on reweighted source and target with regularization from reweighted source

### OfficeHome

---

- Domains:
    - Art: Ar, Clip Art: Cl, Product: Pr, Real World: Rw
- For example, if you want to run "ours2" with Art as pretraining data and Product as finetuning data.

    ```bash
    python main.py --save_dir=ArPr_ours2 --gpu=0 --source_domain=Ar --target_domain=Pr --ours2 --lam=7e-3
    ```

- To run all domains of a specific setting (e.g. ours1)

    ```bash
    sh officehome_script/ours1.sh 
    ```

### Office31

---

- Domains:
    - Amazon: A, DSLR: D, Webcam: W
- For example, if you want to run "ours2" with Amazon as pretraining data and Webcam as finetuning data.

    ```bash
    python main.py --save_dir=A_W_ours2 --gpu=0 --source_domain=A --target_domain=W --dataset=office31 --ours2 --lam=5e-4
    ```

- To run all domains of a specific setting (e.g. ours1)

    ```bash
    sh office31_script/ours1.sh 
    ```