import os, json, datetime, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
import wandb

from model import VGG6
from data import get_datasets, make_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_optimizer(name, params, lr, weight_decay=5e-4, momentum=0.9):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    if name == "nesterov-sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    losses, accs = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train_mode:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item()); accs.append(accuracy_from_logits(logits, yb))
    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, accs = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item()); accs.append(accuracy_from_logits(logits, yb))
    return float(np.mean(losses)), float(np.mean(accs))

def nowstamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def train_one_run(cfg, project="vgg6-cifar10"):
    Path("weights").mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(parents=True, exist_ok=True)

    run_name = f"vgg6_{nowstamp()}"
    with wandb.init(project=project, config=cfg, name=run_name) as run:
        set_seed(cfg["seed"])
        trn, val, tst = get_datasets("./data", val_frac=0.1, seed=cfg["seed"])
        train_loader, val_loader, test_loader = make_loaders(trn, val, tst, batch_size=cfg["batch_size"])

        model = VGG6(activation=cfg["activation"]).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        opt = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                              weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])

        # save config
        run_dir = Path("runs") / f"{run.name or run.id}_{nowstamp()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        best_val, best_path = 0.0, None
        for epoch in range(cfg["epochs"]):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt)
            va_loss, va_acc = evaluate(model, val_loader, criterion)
            wandb.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc*100,
                                  "val_loss": va_loss,   "val_acc": va_acc*100})
            if va_acc > best_val:
                best_val = va_acc
                best_path = Path("weights") / f"{run.name or run.id}_{nowstamp()}_best.pt"
                torch.save(model.state_dict(), best_path)

        if best_path is not None and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        wandb.summary["test_acc"] = te_acc*100
        wandb.summary["test_loss"] = te_loss

        if best_path is not None and best_path.exists():
            art = wandb.Artifact(f"{run.name}_best", type="model",
                                 metadata={"val_acc": float(best_val), "test_acc": float(te_acc)})
            art.add_file(str(best_path)); wandb.log_artifact(art)
            print(f"Saved best weight: {best_path}")
        else:
            print("No best weight saved (unexpected).")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--activation", default="relu")
    p.add_argument("--optimizer",  default="nesterov-sgd")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--lr",         type=float, default=0.05)
    p.add_argument("--momentum",   type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--project",    default="vgg6-cifar10")
    args = p.parse_args()
    cfg = vars(args)
    wandb.login()
    train_one_run(cfg, project=args.project)