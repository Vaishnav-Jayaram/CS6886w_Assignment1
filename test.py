import torch, torch.nn as nn
from pathlib import Path
from model import VGG6
from data import get_datasets, make_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, accs = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        accs.append((logits.argmax(dim=1) == yb).float().mean().item())
    import numpy as np
    return float(np.mean(losses)), float(np.mean(accs))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to the .pt file in weights/")
    p.add_argument("--activation", default="relu", help="activation used during training")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    tr, va, te = get_datasets("./data", val_frac=0.1, seed=42)
    _, _, test_loader = make_loaders(tr, va, te, batch_size=args.batch_size)

    model = VGG6(activation=args.activation).to(DEVICE)
    sd = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)

    criterion = nn.CrossEntropyLoss()
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"Loaded: {args.weights}")
    print(f"Test top-1: {te_acc*100:.2f}% | loss: {te_loss:.4f}")