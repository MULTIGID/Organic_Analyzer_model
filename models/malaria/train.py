import argparse
from pathlib import Path

from src.multiclass_runner import train_multiclass
from .data import create_loaders


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on NIH Malaria.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train_multiclass(Path(__file__).with_name("config.yaml"), create_loaders,
                     args.device, args.resume, args.smoke_test)


if __name__ == "__main__":
    main()
