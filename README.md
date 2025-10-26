# CS6886w_Assignment1
**Systems Engineering for Deep Learning**

---

### 🚀 Setup Instructions


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
# In Colab:
python - <<'EOF'
import os, wandb
try:
    wandb.login(timeout=25)
except Exception:
    print("Unable to login")
    pass
EOF

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
