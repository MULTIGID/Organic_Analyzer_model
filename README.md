# Organic Analyzer Model

[<kbd>English</kbd>](README.md) [<kbd>Українська</kbd>](README_UA.md)

A bilingual Streamlit application for biological-species classification with dedicated ResNet-50 models trained on **iNaturalist 2021 Full** and the image-only labeled subset of **BIOSCAN-5M**. The interface supports animals, insects, plants and fungi, top predictions, confidence indicators, input-quality warnings, Google search links, and Grad-CAM visualization.

Model output is intended for research and education and requires expert review.

## Current datasets

| Status | Dataset | Purpose | Scale |
|---|---|---|---:|
| Current | iNaturalist 2021 Full | Classification of animals, plants, fungi, and other organisms | 10,000 classes; 2,686,843 training images |
| Current | BIOSCAN-5M image-only labeled subset | Insect species classification without DNA input | 11,846 classes; 289,203 training images |

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| Operating system | 64-bit Windows 10/11 or a current 64-bit Linux distribution | 64-bit Windows 11 |
| Python | 3.12 or 3.13, 64-bit | Python 3.12, 64-bit |
| Processor | 4-core x64 processor | 6 or more modern cores |
| Memory | 8 GB RAM | 16 GB RAM or more |
| Storage | 9 GB of free space for the application, environment, and checkpoints | 12 GB or more |
| Graphics | CPU inference is supported | NVIDIA GPU with at least 6 GB VRAM and a compatible CUDA-enabled PyTorch build |
| Network | Required to install Python packages | Not required for local image analysis after installation |

Training datasets are not required to run image analysis. GPU acceleration is optional; without a compatible NVIDIA GPU, the application automatically uses the CPU and processes images more slowly.

## Run the application

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The checkpoints must be stored under `checkpoints/inaturalist` and `checkpoints/bioscan`.

## Project structure

```text
app.py                     Streamlit interface
models/inaturalist/        Training configuration and entry points
models/bioscan/            BIOSCAN configuration, data loader, training and evaluation
src/                       Model, data, inference, taxonomy, and Grad-CAM code
checkpoints/inaturalist/   Best and latest iNaturalist checkpoints
checkpoints/bioscan/       Best/latest BIOSCAN checkpoints and class mapping
results/inaturalist/       Training and evaluation outputs
tests/                     Automated tests
```
