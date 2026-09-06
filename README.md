# Organic Analyzer Model

A bilingual Streamlit application for biological-species classification with dedicated ResNet-50 models trained on **iNaturalist 2021 Full** and the image-only labeled subset of **BIOSCAN-5M**. The interface supports animals, insects, plants and fungi, top predictions, confidence indicators, input-quality warnings, Google search links, and Grad-CAM visualization.

Model output is intended for research and education and requires expert review.

## Current datasets

| Status | Dataset | Purpose | Scale |
|---|---|---|---:|
| Current | iNaturalist 2021 Full | Classification of animals, plants, fungi, and other organisms | 10,000 classes; 2,686,843 training images |
| Current | BIOSCAN-5M image-only labeled subset | Insect species classification without DNA input | 11,846 classes; 289,203 training images |

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

[Українська версія README](README_UA.md) · [Run on another PC](RUN_ON_ANOTHER_PC.md)
