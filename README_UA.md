# Organic Analyzer Model

Двомовний Streamlit-застосунок для класифікації біологічних видів за допомогою ResNet-50, навченої на **iNaturalist 2021 Full**. Інтерфейс підтримує фільтрацію за Animalia, Plantae і Fungi, найімовірніші прогнози, показники впевненості, перевірку якості вхідного зображення, пошук у Google та візуалізацію Grad-CAM.

Результат моделі призначений для досліджень і навчання та потребує експертної перевірки.

## Поточні й заплановані датасети

| Статус | Датасет | Призначення | Масштаб |
|---|---|---|---:|
| Використовується | iNaturalist 2021 Full | Класифікація тварин, рослин, грибів та інших організмів | 10 000 класів; 2 686 843 навчальних зображення |
| Заплановано | BIOSCAN-5M | Класифікація комах і мультимодальні таксономічні дослідження | 5 150 808 зразків |

Код і checkpoints BIOSCAN-5M будуть додані після визначення структури даних та процесу навчання.

## Запуск застосунку

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Checkpoint iNaturalist має бути розміщений у `checkpoints/inaturalist/resnet50_inaturalist_best.pt`.

## Навчання та оцінювання iNaturalist

Спочатку вкажіть шляхи до датасету у `models/inaturalist/config.yaml`, а потім виконайте:

```powershell
.\.venv\Scripts\python.exe -m models.inaturalist.train --device cuda
.\.venv\Scripts\python.exe -m models.inaturalist.train --resume --device cuda
.\.venv\Scripts\python.exe -m models.inaturalist.evaluate --device cuda
```

На комп’ютері без сумісної NVIDIA GPU використовуйте `--device cpu`.

## Структура проєкту

```text
app.py                     Інтерфейс Streamlit
models/inaturalist/        Конфігурація та команди навчання
src/                       Код моделі, даних, inference, таксономії та Grad-CAM
checkpoints/inaturalist/   Найкращий і останній checkpoints iNaturalist
results/inaturalist/       Результати навчання та оцінювання
tests/                     Автоматичні тести
```

[English README](README.md) · [Запуск на іншому ПК](RUN_ON_ANOTHER_PC.md)
