from __future__ import annotations

import platform
import sys


def status(ok: bool, message: str) -> None:
    print(f"[{'OK' if ok else 'ERROR'}] {message}")


def main() -> int:
    print("Organic Analyzer environment check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    version_ok = sys.version_info >= (3, 10)
    status(version_ok, "Python 3.10 or newer")
    try:
        import torch
        from src.config import load_config
        from src.folder_multiclass import class_directories

        status(True, f"PyTorch {torch.__version__}")
        status(torch.cuda.is_available(), "CUDA GPU available")
        if torch.cuda.is_available():
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
        config = load_config("models/inaturalist/config.yaml")
        checkpoint = config.path("paths", "checkpoint")
        status(checkpoint.exists(), f"iNaturalist checkpoint: {checkpoint}")
        try:
            _, classes = class_directories(config.path("data", "train_root"))
            status(len(classes) == 10_000, f"iNaturalist classes: {len(classes)}")
        except (FileNotFoundError, ValueError) as error:
            status(False, str(error))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        status(False, str(error))
        return 1
    return 0 if version_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
