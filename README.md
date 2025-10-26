# CS6886w_Assignment1
**Systems Engineering for Deep Learning**

---
A notebook file with full end to end implementation which is used for experimental purpose is available - (vgg6_notebook_code)

###  Setup Instructions


# 1. Clone the repository
```bash
git clone https://github.com/Vaishnav-Jayaram/CS6886w_Assignment1.git
```
# 2. Move to the cloned directory
```bash
cd CS6886w_Assignment1
```
# 3. Create a new environment and install dependencies
```bash
# (Use %pip instead of pip if running in Google Colab)
pip install -r requirements.txt
# or in Colab
%pip -q install -r requirements.txt || true
```
# 4. Install additional recommended packages
```bash
pip -q install wandb==0.17.9 matplotlib==3.9.2 rich==13.9.2 -U scikit-learn pyyaml
```
# 5. Login to Weights & Biases (WandB)
```bash
# In Colab cell:

import os, wandb
try:
    wandb.login(timeout=25)
except Exception:
    print("Unable to login")
    pass


# Or in terminal:
wandb login <api_key>
```
# 6. Download and prepare datasets
```bash
python -c "from data import get_datasets, make_loaders; tr,va,te=get_datasets('./data',0.1,42); print(f'Train:{len(tr)} Val:{len(va)} Test:{len(te)}')"
```
# 7. Test the model with the best loaded weights
```bash
python test.py --weights Best.pt --activation gelu --batch_size 128
```
# 8. Baseline Training
```bash
python train.py --activation relu --optimizer nesterov-sgd --batch_size 512 --epochs 30 --lr 0.05 --momentum 0.9 --weight_decay 0.0005 --seed 42 --project vgg6-cifar10
```
# 9. Training using best configuration
```bash
python train.py --activation gelu --optimizer sgd --batch_size 128 --epochs 20 --lr 0.05 --momentum 0.9 --weight_decay 0.0 --seed 2 --project vgg6-cifar10
```
# 10. Run a sweep experiment
```bash
python run_sweep.py --project vgg6-cifar10 --count 22
```
# 11. Online Mode (Iflive W&B dashboard is needed)
```bash
export WANDB_MODE=online
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python run_sweep.py --project vgg6-cifar10 --count 22
```
Link to the wandb Dashboard for the best sweep runs of 22 configurations:
(https://api.wandb.ai/links/ee24d032-iitm-india/bv8iqbhy)
