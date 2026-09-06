# Organic Analyzer Model

Двомовний Streamlit-застосунок для класифікації біологічних видів за допомогою окремих моделей ResNet-50, навчених на **iNaturalist 2021 Full** і розміченій частині зображень **BIOSCAN-5M**. Інтерфейс підтримує тварин, комах, рослини та гриби, найімовірніші прогнози, показники впевненості, перевірку якості зображення, пошук у Google та Grad-CAM.

Результат моделі призначений для досліджень і навчання та потребує експертної перевірки.

## Поточні датасети

| Статус | Датасет | Призначення | Масштаб |
|---|---|---|---:|
| Використовується | iNaturalist 2021 Full | Класифікація тварин, рослин, грибів та інших організмів | 10 000 класів; 2 686 843 навчальних зображення |
| Використовується | Розмічена частина зображень BIOSCAN-5M | Класифікація видів комах без даних ДНК | 11 846 класів; 289 203 навчальних зображення |

## Запуск застосунку

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Checkpoints мають бути розміщені в `checkpoints/inaturalist` і `checkpoints/bioscan`.

## Навчання та оцінювання iNaturalist

Спочатку вкажіть шляхи до датасету у `models/inaturalist/config.yaml`, а потім виконайте:

```powershell
.\.venv\Scripts\python.exe -m models.inaturalist.train --device cuda
.\.venv\Scripts\python.exe -m models.inaturalist.train --resume --device cuda
.\.venv\Scripts\python.exe -m models.inaturalist.evaluate --device cuda
```

На комп’ютері без сумісної NVIDIA GPU використовуйте `--device cpu`.

## Навчання та оцінювання BIOSCAN-5M

Спочатку вкажіть шляхи до датасету у `models/bioscan/config.yaml`, а потім виконайте:

```powershell
.\.venv\Scripts\python.exe -m models.bioscan.train
.\.venv\Scripts\python.exe -m models.bioscan.train --resume --epochs 20
.\.venv\Scripts\python.exe -m models.bioscan.evaluate --device cuda
```

Для навчання BIOSCAN використовуються лише зображення й таксономічні мітки видів; ДНК-штрихкоди не використовуються.

## Структура проєкту

```text
app.py                     Інтерфейс Streamlit
models/inaturalist/        Конфігурація та команди навчання
models/bioscan/            Конфігурація, дані, навчання та оцінювання BIOSCAN
src/                       Код моделі, даних, inference, таксономії та Grad-CAM
checkpoints/inaturalist/   Найкращий і останній checkpoints iNaturalist
checkpoints/bioscan/       Checkpoints BIOSCAN і словник класів
results/inaturalist/       Результати навчання та оцінювання
tests/                     Автоматичні тести
```

[English README](README.md) · [Запуск на іншому ПК](RUN_ON_ANOTHER_PC.md)
