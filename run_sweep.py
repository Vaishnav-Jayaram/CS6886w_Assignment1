import wandb, yaml, argparse
from train import train_one_run

def main(args):
    wandb.login()
    with open(args.sweep, "r") as f:
        sweep_cfg = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_cfg, project=args.project)
    def agent():
        train_one_run(wandb.config, project=args.project)
    wandb.agent(sweep_id, function=agent, count=args.count)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="vgg6-cifar10")
    p.add_argument("--sweep", default="sweep.yaml")
    p.add_argument("--count", type=int, default=25)
    main(p.parse_args())