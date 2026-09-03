# Organic Analyzer Model

A bilingual Streamlit application for biological-species classification with a ResNet-50 model trained on **iNaturalist 2021 Mini**. The interface supports Animalia, Plantae, and Fungi filtering, top predictions, confidence indicators, input-quality warnings, Google search links, and Grad-CAM visualization.

Model output is intended for research and education and requires expert review.

## Current and planned datasets

| Status | Dataset | Purpose | Scale |
|---|---|---|---:|
| Current | iNaturalist 2021 Mini | Classification of animals, plants, fungi, and other organisms | 10,000 classes; 500,000 training images |
| Planned | BIOSCAN-5M | Insect classification and multimodal taxonomy research | 5,150,808 specimens |

BIOSCAN-5M code and checkpoints will be added after its training pipeline and data layout are finalized.

## Run the application

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The iNaturalist checkpoint must be stored at `checkpoints/inaturalist/resnet50_inaturalist_best.pt`.

## Train and evaluate iNaturalist

Update the dataset paths in `models/inaturalist/config.yaml`, then run:

```powershell
.\.venv\Scripts\python.exe -m models.inaturalist.train --device cuda
.\.venv\Scripts\python.exe -m models.inaturalist.train --resume --device cuda
.\.venv\Scripts\python.exe -m models.inaturalist.evaluate --device cuda
```

Use `--device cpu` without a compatible NVIDIA GPU.

## Project structure

```text
app.py                     Streamlit interface
models/inaturalist/        Training configuration and entry points
src/                       Model, data, inference, taxonomy, and Grad-CAM code
checkpoints/inaturalist/   Best and latest iNaturalist checkpoints
results/inaturalist/       Training and evaluation outputs
tests/                     Automated tests
```

[Українська версія README](README_UA.md) · [Run on another PC](RUN_ON_ANOTHER_PC.md)
