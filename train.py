import torch
import random
import copy
import numpy as np
from argparse import ArgumentParser
import yaml
from pathlib import Path
from models.registry import supervisor_dict


def parse_args():
    parser = ArgumentParser(description="Run baseline in a convient framework!")
    parser.add_argument("-m", "--model", help="model name", choices=["GTS", "GWNet", "AGCRN", "STAEformer", "STID", "STWave"])
    parser.add_argument("-c", "--cfg", default="configs/GTS/PEMS08.yaml", help="training config")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")
    parser.add_argument("-s", "--seed", nargs="+", type=int, default=[2026], help="one or more random seeds, e.g. -s 1 2 3")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # 加载配置
    config_path = Path(args.cfg)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(config)
    for s in args.seed:
        cfg = copy.deepcopy(config)
        cfg["seed"] = s
        set_seed(s)
        print(f"\n===== model={args.model} | seed={s} =====")
        # 创建Supervisor并开始训练
        supervisor = supervisor_dict[args.model](**cfg)
        supervisor.train()

        # 善后工作
        del supervisor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
