# Running the application on another Windows PC

The archive contains the application code and inference checkpoints. Training datasets are not required for image analysis.

## Requirements

- Windows 10 or Windows 11
- Python 3.12 or 3.13 (64-bit); Python 3.12 is recommended
- At least 8 GB RAM; 16 GB is recommended
- Approximately 8 GB of free disk space for installation
- An NVIDIA GPU is optional. The application can run on CPU.

## Installation

Open PowerShell in the extracted project directory and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Start locally

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in a browser.

## Start for devices in the local network

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://COMPUTER_LOCAL_IP:8501` on another device. Windows Firewall must allow Python or TCP port 8501 for private networks.

## GPU check

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, the application automatically uses CPU.

---

# Запуск застосунку на іншому комп'ютері Windows

Архів містить код застосунку та checkpoints для розпізнавання. Навчальні датасети для аналізу зображень не потрібні.

1. Встановіть 64-бітний Python 3.12 або 3.13; рекомендовано Python 3.12.
2. Розпакуйте архів.
3. Відкрийте PowerShell у папці проєкту.
4. Створіть `.venv` та встановіть залежності командами з розділу **Installation**.
5. Запустіть `streamlit run app.py`.

Для роботи через локальну мережу використовуйте команду з розділу **Start for devices in the local network**.
