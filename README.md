# CS6886w_Assignment1
Systems Engineering for Deep Learning

1.Clone the git repo :
'''git clone https://github.com/Vaishnav-Jayaram/CS6886w_Assignment1.git'''
2.Move to the cloned directory:
'''cd CS6886w_Assignment1'''
3. Create a new environment and Install the packages using requirements.txt file or if in colab cell, directly install using below command but with '%' before pip
'''pip -q install -r requirements.txt || true''
4. On a safer side install few packages with the below command:
'''pip -q install wandb==0.17.9 matplotlib==3.9.2 rich==13.9.2 -U scikit-learn pyyaml'''
5. If in colab, try logging into wandb using:
'''
import os, wandb
try:
    wandb.login(timeout=25)
except Exception:
  print("Unable to login")
  pass
'''
or in terminal:
'''wandb login <api_key>'''
6.Data downloading:
'''python -c "from data import get_datasets, make_loaders; tr,va,te=get_datasets('./data',0.1,42); print(f'Train:{len(tr)} Val:{len(va)} Test:{len(te)}')" '''
7. To test the model with the best loaded weights:
'''!python test.py --weights Best.pt --activation gelu --batch_size 128'''
