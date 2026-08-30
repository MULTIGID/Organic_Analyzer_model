# Biological Image Analyzer

A research-oriented Streamlit application for classifying biological and medical
images with dedicated ResNet-50 models. The interface supports English and
Ukrainian, CPU inference, and compatible NVIDIA GPUs.

> Model outputs require expert review. This application is not intended for
> clinical diagnosis.

[Українська версія README](README_UA.md)

## Available modules

| Module | Dataset and task | Classes |
|---|---|---:|
| PCam | H&E lymph-node patches; metastatic tissue detection | 2 |
| PBC | Microscopy images of peripheral blood cells | 8 |
| PlantVillage | Healthy and diseased leaves from 14 crops | 38 |
| NIH Malaria | Parasitized and uninfected individual blood cells | 2 |
| PathMNIST (MedMNIST) | Standardized H&E colorectal tissue images | 9 |
| iNaturalist 2021 Mini | Animals, plants, fungi, and other organisms | 10,000 |
| NCT-CRC-HE-100K | Tumor, normal, and other colorectal tissue patches | 9 |

Each module has separate training and evaluation files, configuration,
checkpoints, and results. Switching modules in the interface does not modify any
model checkpoint.

## Features

- ResNet-50 transfer learning with ImageNet initialization;
- mixed-precision training on compatible NVIDIA GPUs;
- early stopping and resumable training;
- separate best and last checkpoints;
- Accuracy, Precision, Recall, F1, and AUC where applicable;
- confusion matrices and training-history charts;
- predicted class, confidence, and top predictions in Streamlit;
- Grad-CAM visualization for all seven models;
- Google search links for predicted classes;
- non-blocking input-image quality checks;
- iNaturalist filtering by Animalia, Plantae, and Fungi;
- responsive light and dark themes;
- English and Ukrainian interface languages.

## System requirements

- Windows 10 or Windows 11;
- 64-bit Python 3.12 or 3.13; Python 3.12 is recommended;
- at least 8 GB RAM; 16 GB is recommended;
- approximately 8 GB of free disk space for the environment and checkpoints;
- an NVIDIA GPU is optional for inference but provides substantial acceleration.

Training datasets are required for training and evaluation, but not for image
analysis with existing checkpoints.

## Installation

Open PowerShell in the project directory containing `app.py`:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a specific CUDA build of PyTorch is required, install it using the command
provided by the official PyTorch installation guide before installing the
remaining requirements.

Verify the environment:

```powershell
python check_setup.py
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Running the application

On Windows, double-click:

```text
start_app.bat
```

Alternatively, run it from a terminal:

```powershell
python -m streamlit run app.py
```

To make the application available to phones or other devices on the local
network:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://LOCAL_IP:8501` on the other device. Windows Firewall must allow
Python or TCP port 8501 on private networks.

## Checkpoints

Checkpoints are not stored in Git because several files exceed GitHub's regular
file-size limit. Place the inference weights in the paths defined by each
module's `config.yaml` file.

- `*_best.pt` is used for inference and evaluation;
- `*_last.pt` stores the latest training state and is used by `--resume`.

The application can display a module whose checkpoint is missing, but that
module cannot perform inference until its `*_best.pt` file is installed.

## Input-image checks

Before inference, the application performs fast heuristic checks for:

- minimum resolution;
- unusual aspect ratio;
- excessively dark or bright images;
- low contrast or insufficient detail;
- possible blur;
- unusual color variation for H&E histology or leaf photography.

Warnings do not block inference. These checks are not an out-of-distribution
detector and do not prove that an image belongs to the expected dataset.

## Configurations and datasets

Dataset paths and training parameters are stored in
`models/<module>/config.yaml`. Forward slashes are recommended in Windows YAML
paths.

| Module | Configuration | Expected data |
|---|---|---|
| PCam | `models/pcam/config.yaml` | Six train/validation/test HDF5 files containing `x` and `y` |
| PBC | `models/pbc/config.yaml` | `Train`, `Val`, and `Test`, each containing eight class folders |
| PlantVillage | `models/plantvillage/config.yaml` | Complete PlantVillage-Dataset repository |
| NIH Malaria | `models/malaria/config.yaml` | `cell_images/Parasitized` and `cell_images/Uninfected` |
| PathMNIST | `models/medmnist/config.yaml` | `pathmnist.npz` |
| iNaturalist | `models/inaturalist/config.yaml` | `train_mini` and `val` containing 10,000 class folders |
| NCT-CRC | `models/nct_crc/config.yaml` | `NCT-CRC-HE-100K` and the independent `CRC-VAL-HE-7K` set |

Datasets are not included in this repository.

## Training

Run a smoke test before starting full training:

```powershell
python -m models.pcam.train --smoke-test --device cuda
python -m models.pbc.train --smoke-test --device cuda
python -m models.plantvillage.train --smoke-test --device cuda
python -m models.malaria.train --smoke-test --device cuda
python -m models.medmnist.train --smoke-test --device cuda
python -m models.inaturalist.train --smoke-test --device cuda
python -m models.nct_crc.train --smoke-test --device cuda
```

For full training, use the same command without `--smoke-test`. To continue
from the last checkpoint, add `--resume`:

```powershell
python -m models.pbc.train --resume --device cuda
```

`--resume` is not required after successfully completed training unless the
configured total number of epochs is increased.

## Evaluation

```powershell
python -m models.pcam.evaluate --device cuda
python -m models.pbc.evaluate --device cuda
python -m models.plantvillage.evaluate --device cuda
python -m models.malaria.evaluate --device cuda
python -m models.medmnist.evaluate --device cuda
python -m models.inaturalist.evaluate --device cuda
python -m models.nct_crc.evaluate --device cuda
```

Results are written to `results/<module>/`, and checkpoints are written to
`checkpoints/<module>/`. PCam stores its main artifacts directly under
`checkpoints/` and `results/` for compatibility with the original layout.

## Main configuration parameters

- `epochs`: maximum total number of epochs;
- `batch_size`: number of images per batch;
- `learning_rate`: optimizer learning rate;
- `weight_decay`: L2 regularization;
- `num_workers`: parallel data-loading processes;
- `patience`: epochs without improvement before early stopping;
- `use_amp`: mixed precision on CUDA.

On Windows, reduce `num_workers` to `0` or `1` if shared-memory or paging-file
errors occur. Reduce `batch_size` if CUDA runs out of memory.

## Project structure

```text
Organic_Analyzer_model/
├── app.py
├── start_app.bat
├── check_setup.py
├── assets/
├── checkpoints/
├── models/
│   ├── pcam/
│   ├── pbc/
│   ├── plantvillage/
│   ├── malaria/
│   ├── medmnist/
│   ├── inaturalist/
│   └── nct_crc/
├── results/
├── src/
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Developer checks

```powershell
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Automated tests do not replace manual validation of every checkpoint on typical
and atypical images.

## Running on another PC

See [RUN_ON_ANOTHER_PC.md](RUN_ON_ANOTHER_PC.md). The repository contains the
application code but does not contain datasets, Python, a virtual environment,
or inference checkpoints.

## Limitations

- Models recognize only the classes present in their respective training sets.
- High softmax confidence does not guarantee a correct prediction.
- Heuristic checks cannot detect every unsupported image type.
- Grad-CAM indicates image regions influencing a prediction; it does not confirm
  pathology localization.
- iNaturalist uses one checkpoint for all 10,000 classes, after which the
  interface filters and renormalizes results for the selected kingdom.
- If the total probability assigned to a selected kingdom is low, conditional
  confidence within that kingdom should not be interpreted as reliable
  recognition.
- The application is not intended for clinical use.
