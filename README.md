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

### Windows: installation without terminal commands

1. Install 64-bit Python 3.12 and enable **Add Python to PATH** in the installer.
2. Download or clone the complete repository.
3. Double-click `install.bat`. It creates `.venv` and installs all packages automatically. Run it only once, or again when `requirements.txt` changes.
4. Add the inference files using the structure below.
5. Double-click `start_app.bat` whenever you want to start the application. Keep its console window open while the application is in use; closing it stops the local server.

```text
checkpoints/
├── inaturalist/
│   └── resnet50_inaturalist_best.pt
└── bioscan/
    ├── bioscan_species_resnet50_best.pt
    └── classes.json
```

The application normally opens `http://localhost:8501` automatically. Other devices on the same local network can use `http://COMPUTER_IP:8501` if Windows Firewall permits the connection.

### Manual or Linux launch

Create a virtual environment, install `requirements.txt`, and run `python -m streamlit run app.py`. The BAT files are intended for Windows only.

The training datasets are not needed for inference. Only the three files listed above are required.

## Connection log

The application records the connection time and client IP address once per browser session in `logs/access.log`. Localhost connections are recorded as `local`. The log rotates automatically at 5 MB and retains up to three older files. IP addresses can be personal data, so disclose this logging to users and define an appropriate retention period before making the application publicly accessible.

## Project structure

```text
app.py                     Streamlit interface
install.bat                One-click Windows environment setup
start_app.bat              One-click Windows application launcher
models/inaturalist/        Training configuration and entry points
models/bioscan/            BIOSCAN configuration, data loader, training and evaluation
src/                       Model, data, inference, taxonomy, and Grad-CAM code
checkpoints/inaturalist/   Best and latest iNaturalist checkpoints
checkpoints/bioscan/       Best/latest BIOSCAN checkpoints and class mapping
results/inaturalist/       Training and evaluation outputs
tests/                     Automated tests
```
