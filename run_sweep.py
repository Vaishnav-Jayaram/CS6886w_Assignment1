import os
import argparse
import yaml
import wandb

from train import train_one_run  # reuse the single-run logic

# Default to OFFLINE unless user opts-in to online
if "WANDB_MODE" not in os.environ and "WANDB_API_KEY" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"

def main(args):
    # best-effort login (no prompt)
    try:
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
        elif os.environ.get("WANDB_MODE", "").lower() != "offline":
            wandb.login(anonymous="allow")
        else:
            print("W&B running in OFFLINE mode — no login required.")
    except Exception as e:
        print("Skipping W&B login:", e)

    # Load sweep config
    with open(args.sweep, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_cfg, project=args.project)
    print(f"Sweep created: {sweep_id}")

    def sweep_train():
        train_one_run(wandb.config, project=args.project)

    wandb.agent(sweep_id, function=sweep_train, count=args.count)
    print(f"Completed {args.count} runs for sweep: {sweep_id}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="vgg6-cifar10")
    p.add_argument("--sweep",   default="sweep.yaml")
    p.add_argument("--count",   type=int, default=22, help="number of runs to execute")
    args = p.parse_args()
    main(args)