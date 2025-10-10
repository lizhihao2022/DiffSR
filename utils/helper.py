import os
import yaml
import torch
import shutil
import logging
import numpy as np

from datetime import datetime


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def set_device(cuda, device):
    if cuda is True and torch.cuda.is_available():
        torch.cuda.set_device(device=device)


def load_config(args):
    with open(args['config'], 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    for k, v in config.items():
        args[k] = v
    return args


def save_config(args, saving_path):
    with open(os.path.join(saving_path, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)


def get_dir_path(args, create_dir=True):
    model = args['model']['name']
    dataset = args['data']['name']
    path = args['log']['log_dir']
    date = datetime.now().strftime("%m_%d")
    time = datetime.now().strftime("_%H_%M_%S")
    dir_path = os.path.join(path, dataset, date, model + time)
    if create_dir:
        os.makedirs(dir_path)
    dir_name = date + "_" + model + time
    return dir_path, dir_name


def set_up_logger(args):
    log_dir, dir_name = get_dir_path(args)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(log_dir, "train.log"),
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(log_dir))

    return log_dir, dir_name


def save_code(module, saving_path, with_dir=False, with_path=False):
    os.makedirs(os.path.join(saving_path, 'code'), exist_ok=True)
    
    if with_path:
        src = module
    else:
        if with_dir:
            src = os.path.dirname(module.__file__)
        else:
            src = module.__file__
    dst = os.path.join(saving_path, 'code', os.path.basename(src))
    
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
