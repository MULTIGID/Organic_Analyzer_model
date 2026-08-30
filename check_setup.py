from __future__ import annotations

import platform
import sys


def status(ok: bool, message: str) -> None:
    marker = "OK" if ok else "ERROR"
    print(f"[{marker}] {message}")


def main() -> int:
    print("Multi-dataset ResNet-50 project environment check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    version_ok = sys.version_info >= (3, 10)
    status(version_ok, "Python 3.10 or newer")
    try:
        import torch

        status(True, f"PyTorch {torch.__version__}")
        status(torch.cuda.is_available(), "CUDA GPU available")
        if torch.cuda.is_available():
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        status(False, "PyTorch is not installed")
        return 1

    try:
        from src.config import load_config
        from src.data import discover_pcam_files

        config = load_config("models/pcam/config.yaml")
        status(True, "models/pcam/config.yaml is valid")
        checkpoint = config.path("paths", "checkpoint")
        status(checkpoint.exists(), f"Model checkpoint: {checkpoint}")
        try:
            files = discover_pcam_files(config.path("data", "root"))
            status(True, "All six PCam HDF5 files were discovered")
            for name, path in files.__dict__.items():
                print(f"      {name}: {path}")
        except (FileNotFoundError, ValueError, KeyError) as error:
            status(False, str(error))

        from src.pbc_data import discover_pbc_splits

        pbc_config = load_config("models/pbc/config.yaml")
        status(True, "models/pbc/config.yaml is valid")
        pbc_checkpoint = pbc_config.path("paths", "checkpoint")
        status(pbc_checkpoint.exists(), f"PBC model checkpoint: {pbc_checkpoint}")
        try:
            splits = discover_pbc_splits(pbc_config.path("data", "root"))
            status(True, "PBC Train, Val and Test splits contain all 8 classes")
            for name, path in splits.items():
                print(f"      {name}: {path}")
        except (FileNotFoundError, ValueError) as error:
            status(False, str(error))

        for module_name in ("malaria", "medmnist", "inaturalist", "nct_crc"):
            module_config = load_config(f"models/{module_name}/config.yaml")
            status(True, f"models/{module_name}/config.yaml is valid")
            module_checkpoint = module_config.path("paths", "checkpoint")
            status(module_checkpoint.exists(), f"{module_name} checkpoint: {module_checkpoint}")

        from src.plantvillage_data import discover_plantvillage

        plant_config = load_config("models/plantvillage/config.yaml")
        status(True, "models/plantvillage/config.yaml is valid")
        plant_checkpoint = plant_config.path("paths", "checkpoint")
        status(plant_checkpoint.exists(), f"PlantVillage checkpoint: {plant_checkpoint}")
        try:
            train, validation, test, classes = discover_plantvillage(
                plant_config.path("data", "root")
            )
            status(True, f"PlantVillage: {len(classes)} classes")
            print(f"      train={len(train)}, validation={len(validation)}, test={len(test)}")
        except (FileNotFoundError, ValueError) as error:
            status(False, str(error))
    except Exception as error:
        status(False, f"Configuration check failed: {error}")
        return 1

    print("\nAn unavailable checkpoint is expected before the first training run.")
    print("CUDA is strongly recommended for full training.")
    return 0 if version_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
