import argparse
from pathlib import Path

from src.multiclass_runner import evaluate_multiclass
from .data import create_loaders


def main():
    parser = argparse.ArgumentParser(description="Evaluate iNaturalist 2021 Full model.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    evaluate_multiclass(Path(__file__).with_name("config.yaml"), create_loaders, args.device)


if __name__ == "__main__":
    main()
