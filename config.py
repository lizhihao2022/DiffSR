import argparse

parser = argparse.ArgumentParser(description='GPO')
parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
parser.add_argument("--config", type=str, default="./configs/base.yaml")
