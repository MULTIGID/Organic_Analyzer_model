# Biological Image Analyzer

[English](README.md) | [Українська](README_UA.md) | [Installation](RUN_ON_ANOTHER_PC.md)

Дослідницький Streamlit-застосунок для класифікації біологічних і медичних
зображень за допомогою окремих моделей ResNet-50. Інтерфейс підтримує англійську
та українську мови, CPU і сумісні NVIDIA GPU.

> Результат є прогнозом моделі, потребує перевірки фахівцем і не є медичним
> діагнозом.

## Модулі

| Модуль | Дані та завдання | Класи |
|---|---|---:|
| PCam | H&E-фрагменти лімфовузлів; виявлення метастатичної тканини | 2 |
| PBC | Мікроскопічні зображення клітин периферичної крові | 8 |
| PlantVillage | 14 культур, здорове листя та хвороби рослин | 38 |
| NIH Malaria | Окремі заражені та незаражені клітини крові | 2 |
| PathMNIST (MedMNIST) | Стандартизовані H&E-зображення тканин товстої кишки | 9 |
| iNaturalist 2021 Mini | Тварини, рослини, гриби та інші організми | 10 000 |
| NCT-CRC-HE-100K | H&E-фрагменти пухлинної, нормальної та інших тканин товстої кишки | 9 |

Кожен модуль має власні файли навчання, оцінювання, конфігурацію, checkpoints і
результати. Перемикання модуля в інтерфейсі не змінює checkpoint іншої моделі.

## Можливості

- transfer learning ResNet-50 із вагами ImageNet;
- mixed precision на сумісній NVIDIA GPU;
- early stopping і продовження перерваного навчання;
- окремі best і last checkpoints;
- Accuracy, Precision, Recall, F1, AUC для відповідних задач;
- confusion matrix та графіки історії навчання;
- прогноз, упевненість і список найімовірніших класів у Streamlit;
- Grad-CAM для всіх семи моделей;
- посилання Google для пошуку інформації про прогнозовані класи;
- не блокувальні перевірки якості вхідного зображення;
- фільтрація iNaturalist за царствами Animalia, Plantae і Fungi;
- адаптивний інтерфейс, світла і темна теми, українська й англійська мови.

## Системні вимоги

- Windows 10 або Windows 11;
- 64-бітний Python 3.12 або 3.13; рекомендовано Python 3.12;
- щонайменше 8 GB RAM, рекомендовано 16 GB;
- приблизно 8 GB вільного місця для середовища та checkpoints;
- NVIDIA GPU не обов’язкова для аналізу, але значно прискорює його.

Датасети потрібні для навчання й оцінювання, але не потрібні для звичайного
аналізу зображень через готові checkpoints.

## Встановлення

Відкрийте PowerShell у кореневій папці, де знаходиться `app.py`:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Якщо потрібна конкретна CUDA-збірка PyTorch, спочатку встановіть її командою з
офіційного PyTorch Get Started, а потім виконайте встановлення requirements.

Перевірка середовища:

```powershell
python check_setup.py
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Запуск застосунку

Подвійне натискання:

```text
start_app.bat
```

Або через термінал:

```powershell
python -m streamlit run app.py
```

Для доступу з телефону чи іншого пристрою в локальній мережі:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

На іншому пристрої відкрийте `http://LOCAL_IP:8501`. Брандмауер Windows має
дозволяти Python або TCP-порт 8501 у приватній мережі.

## Перевірка завантажених зображень

Перед аналізом застосунок виконує швидкі евристичні перевірки:

- мінімальна роздільна здатність;
- незвичне співвідношення сторін;
- надто темне або надто світле зображення;
- низький контраст і мала кількість деталей;
- ймовірне розмиття;
- незвична колірна мінливість для H&E-гістології або фотографії листка.

Попередження не блокує аналіз. Це не OOD-детектор і не підтвердження того, що
зображення належить потрібному датасету.

## Конфігурації та датасети

Шляхи до даних і параметри знаходяться у `models/<module>/config.yaml`. На
Windows у YAML рекомендовано використовувати `/`.

| Модуль | Конфігурація | Очікувані дані |
|---|---|---|
| PCam | `models/pcam/config.yaml` | шість HDF5-файлів train/validation/test із `x` та `y` |
| PBC | `models/pbc/config.yaml` | `Train`, `Val`, `Test`, у кожній папці 8 класів |
| PlantVillage | `models/plantvillage/config.yaml` | повний репозиторій PlantVillage-Dataset |
| NIH Malaria | `models/malaria/config.yaml` | `cell_images/Parasitized` і `cell_images/Uninfected` |
| PathMNIST | `models/medmnist/config.yaml` | файл `pathmnist.npz` |
| iNaturalist | `models/inaturalist/config.yaml` | `train_mini` і `val` із 10 000 папок класів |
| NCT-CRC | `models/nct_crc/config.yaml` | `NCT-CRC-HE-100K` і незалежний `CRC-VAL-HE-7K` |

Датасети в цей репозиторій і переносний архів не входять.

## Команди навчання

Спочатку рекомендовано виконати smoke test:

```powershell
python -m models.pcam.train --smoke-test --device cuda
python -m models.pbc.train --smoke-test --device cuda
python -m models.plantvillage.train --smoke-test --device cuda
python -m models.malaria.train --smoke-test --device cuda
python -m models.medmnist.train --smoke-test --device cuda
python -m models.inaturalist.train --smoke-test --device cuda
python -m models.nct_crc.train --smoke-test --device cuda
```

Повне навчання виконується тією самою командою без `--smoke-test`. Щоб
продовжити з last checkpoint, додайте `--resume`:

```powershell
python -m models.pbc.train --resume --device cuda
```

`--resume` не потрібен після успішно завершеного навчання, якщо не планується
збільшення загальної кількості епох.

## Команди оцінювання

```powershell
python -m models.pcam.evaluate --device cuda
python -m models.pbc.evaluate --device cuda
python -m models.plantvillage.evaluate --device cuda
python -m models.malaria.evaluate --device cuda
python -m models.medmnist.evaluate --device cuda
python -m models.inaturalist.evaluate --device cuda
python -m models.nct_crc.evaluate --device cuda
```

Результати зберігаються окремо в `results/<module>/`, checkpoints — у
`checkpoints/<module>/`. PCam зберігає основні артефакти без додаткової вкладеної
папки для сумісності з початковою структурою проєкту.

## Основні параметри конфігурації

- `epochs` — максимальна загальна кількість епох;
- `batch_size` — кількість зображень в одному пакеті;
- `learning_rate` — швидкість оновлення ваг;
- `weight_decay` — L2-регуляризація;
- `num_workers` — паралельні процеси читання даних;
- `patience` — кількість епох без покращення до early stopping;
- `use_amp` — mixed precision на CUDA.

На Windows у разі помилок shared memory або paging file зменште `num_workers` до
`0` або `1`. При `CUDA out of memory` зменште `batch_size`.

## Структура проєкту

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

## Тести для розробника

```powershell
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Автоматичні тести не замінюють ручну перевірку кожного checkpoint на типових і
нетипових зображеннях.

## Перенесення на інший ПК

Дивіться `RUN_ON_ANOTHER_PC.md`. Переносний ZIP містить код, assets і inference
checkpoints, але не містить датасети, Python або готове віртуальне середовище.

## Обмеження

- моделі працюють лише з класами відповідного навчального набору;
- висока впевненість softmax не гарантує правильність прогнозу;
- евристичні перевірки не розпізнають усі сторонні типи зображень;
- Grad-CAM показує вплив ділянок на прогноз, але не підтверджує локалізацію патології;
- iNaturalist використовує один checkpoint на всі 10 000 класів, після чого
  інтерфейс фільтрує та нормалізує результати за вибраним царством;
- якщо сумарна ймовірність вибраного царства мала, умовна впевненість усередині
  нього не повинна трактуватися як надійне розпізнавання;
- застосунок не призначений для клінічного використання.
