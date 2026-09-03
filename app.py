from __future__ import annotations

import base64
import html
import json
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st

st.set_page_config(
    page_title="Histology ResNet-50 Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

background_path = Path(__file__).resolve().parent / "assets" / "biology_background.png"
background_base64 = base64.b64encode(background_path.read_bytes()).decode("ascii")
background_styles = (
    """
    <style>
    .stApp {
        background-color: light-dark(#edf4fb, #031b35);
        background-image:
            linear-gradient(
                light-dark(rgba(248, 250, 252, 0.76), rgba(2, 15, 31, 0.32)),
                light-dark(rgba(241, 245, 249, 0.84), rgba(2, 15, 31, 0.48))
            ),
            url("data:image/png;base64,__BACKGROUND_IMAGE__");
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stMainBlockContainer"] {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        padding: 2rem;
        color: light-dark(#0f172a, #f8fafc);
        border: 1px solid light-dark(rgba(71, 85, 105, 0.22), rgba(148, 163, 184, 0.22));
        border-radius: 1.25rem;
        background: light-dark(rgba(248, 250, 252, 0.94), rgba(8, 17, 31, 0.90));
        box-shadow: 0 1rem 3rem light-dark(rgba(15, 23, 42, 0.18), rgba(0, 0, 0, 0.38));
        backdrop-filter: blur(7px);
    }
    [data-testid="stSidebar"] > div:first-child {
        color: light-dark(#0f172a, #f8fafc);
        background: light-dark(rgba(248, 250, 252, 0.97), rgba(8, 17, 31, 0.94));
        backdrop-filter: blur(8px);
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid light-dark(
            rgba(71, 85, 105, 0.28),
            rgba(148, 163, 184, 0.24)
        );
    }
    [data-testid="stAlert"] {
        color: light-dark(#0f172a, #f8fafc);
        border-color: light-dark(rgba(37, 99, 235, 0.28), rgba(96, 165, 250, 0.30));
        background: light-dark(rgba(219, 234, 254, 0.92), rgba(15, 45, 74, 0.82));
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    [data-testid="stHeader"] {
        height: 4rem !important;
        min-height: 4rem !important;
        background: light-dark(rgba(248, 250, 252, 0.97), rgba(8, 17, 31, 0.94)) !important;
    }
    [data-testid="stHeader"]::before {
        position: absolute;
        top: 50%;
        left: 4rem;
        max-width: calc(100% - 10.5rem);
        overflow: hidden;
        color: light-dark(#0f172a, #f8fafc);
        font-size: clamp(1rem, 2vw, 1.35rem);
        font-weight: 700;
        line-height: 1.2;
        text-overflow: ellipsis;
        white-space: nowrap;
        transform: translateY(-50%);
        content: var(--app-header-title, "Biological Image Analyzer");
    }
    [data-testid="stDecoration"],
    footer {
        display: none !important;
    }
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 0.4rem !important;
        left: 0.5rem !important;
        z-index: 1000000 !important;
    }
    [data-testid="stSidebarCollapsedControl"] button {
        display: flex !important;
        visibility: visible !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 9rem;
        color: light-dark(#0f172a, #f8fafc);
        border: 1px dashed light-dark(rgba(37, 99, 235, 0.55), rgba(96, 165, 250, 0.62));
        border-radius: 1rem;
        background: light-dark(rgba(241, 245, 249, 0.90), rgba(30, 41, 59, 0.72));
    }
    .module-card, .model-status {
        margin: 0.8rem 0;
        padding: 0.9rem 1rem;
        color: light-dark(#0f172a, #f8fafc);
        border: 1px solid light-dark(rgba(71, 85, 105, 0.20), rgba(148, 163, 184, 0.22));
        border-radius: 0.9rem;
        background: light-dark(rgba(241, 245, 249, 0.90), rgba(30, 41, 59, 0.72));
    }
    .module-card__title {
        margin-bottom: 0.35rem;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .module-card__meta, .model-status__meta {
        color: light-dark(#475569, #cbd5e1);
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .module-card__paragraph + .module-card__paragraph {
        margin-top: 0.75rem;
    }
    .module-card__classes {
        margin-top: 0.75rem;
    }
    .module-card__class-list {
        margin-top: 0.25rem;
        padding-left: 0.15rem;
        line-height: 1.55;
    }
    .model-status__state {
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .model-status--ready .model-status__state { color: #4ade80; }
    .model-status--missing .model-status__state { color: #fbbf24; }
    .prediction-card {
        min-height: 7rem;
        padding: 1rem 1.1rem;
        color: light-dark(#0f172a, #f8fafc);
        border: 1px solid light-dark(rgba(37, 99, 235, 0.30), rgba(96, 165, 250, 0.35));
        border-radius: 0.9rem;
        background: light-dark(rgba(241, 245, 249, 0.90), rgba(30, 41, 59, 0.72));
    }
    .prediction-card__label {
        margin-bottom: 0.4rem;
        color: light-dark(#475569, #cbd5e1);
        font-size: 0.92rem;
    }
    .prediction-card__value {
        color: light-dark(#0f172a, #f8fafc);
        font-size: clamp(1.35rem, 3vw, 2.25rem);
        font-weight: 650;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .prediction-card__path {
        margin-top: 0.55rem;
        color: light-dark(#64748b, #94a3b8);
        font-size: 0.82rem;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }
    .app-loader {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.8rem;
        min-height: 5rem;
        color: light-dark(#475569, #94a3b8);
        font-size: 1rem;
    }
    .app-loader__icon {
        display: inline-block;
        font-size: 2rem;
        animation: microscope-pulse 1.1s ease-in-out infinite;
    }
    @keyframes microscope-pulse {
        0%, 100% { transform: scale(0.92); opacity: 0.55; }
        50% { transform: scale(1.08); opacity: 1; }
    }
    @media (prefers-reduced-motion: reduce) {
        .app-loader__icon { animation: none; }
    }
    @media (max-width: 768px) {
        .stApp { background-attachment: scroll; }
        [data-testid="stHeader"]::before {
            left: 4rem;
            max-width: calc(100% - 8rem);
            font-size: 1rem;
        }
        [data-testid="stMainBlockContainer"] {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            padding: 1.15rem;
            border-radius: 0.9rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 7rem;
        }
        h1 { font-size: 2rem !important; }
    }
    </style>
    """
).replace("__BACKGROUND_IMAGE__", background_base64)
st.markdown(
    background_styles,
    unsafe_allow_html=True,
)

page_loader = st.empty()
page_loader.markdown(
    """
    <div class="app-loader" role="status" aria-live="polite">
        <span class="app-loader__icon" aria-hidden="true">🔬</span>
        <span>Loading the analyzer... / Завантаження аналізатора...</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load the ML stack only after the page styles and custom loader reach the
# browser. Importing PyTorch can otherwise leave Streamlit's default loader
# visible for several seconds during a cold start.
import torch  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from src.config import load_config  # noqa: E402
from src.input_validation import validate_module_input  # noqa: E402
from src.multiclass_inference import MulticlassPredictor  # noqa: E402
from src.taxonomy import (  # noqa: E402
    INATURALIST_DOMAIN_CLASS_COUNTS,
    filter_inaturalist_probabilities,
)
from src.utils import resolve_device  # noqa: E402

TEXT = {
    "EN": {
        "title": "Biological Image Analyzer",
        "subtitle": "ResNet-50 classification platform",
        "caption": "Research prototype for biological species classification",
        "disclaimer": "For research and education only. The result is a model prediction and requires expert review.",
        "settings": "Analysis settings", "language": "Language / Мова", "model": "Model",
        "category": "Domain",
        "animals": "Animals", "plants": "Plants", "mushrooms": "Mushrooms",
        "ready": "Model ready", "training_required": "Training required",
        "classes": "classes", "best_accuracy": "Best validation accuracy",
        "checkpoint_updated": "Checkpoint date", "architecture": "ResNet-50",
        "checkpoint_version": "Checkpoint version", "training_epoch": "Training epoch",
        "dataset": "Dataset",
        "taxonomy_path": "Taxonomy", "top_predictions": "Top predictions",
        "category_probability": "Selected-domain probability",
        "filtered_confidence": "Confidence within selected domain",
        "category_mismatch": "The image probably does not belong to the selected domain: {domain}.",
        "google_search": "Search in Google",
        "device": "Device",
        "inaturalist_task": "Recognizes 10,000 species of animals, plants, fungi and other organisms from iNaturalist 2021 Mini.",
        "inaturalist_input": "Upload a clear nature photograph in which the organism is the main subject.",
        "checkpoint": "The {module} checkpoint was not found. Train it first with `{command}`.",
        "checkpoint_damaged": "The {module} checkpoint could not be loaded. The file may be damaged or incompatible. Restore a verified checkpoint and try again.",
        "upload": "Upload an image for {module}", "upload_hint": "Upload one image to begin.",
        "bad_image": "The uploaded file could not be read as an image.",
        "file_too_large": "The uploaded file is larger than 20 MB. Choose a smaller image.",
        "small_image": "The image is too small for reliable analysis.",
        "uploaded": "Uploaded image", "analyze": "Analyze image",
        "running": "Running {module} ResNet-50 analysis...", "gradcam": "Grad-CAM explanation",
        "gradcam_intensity": "Grad-CAM intensity",
        "gradcam_intensity_help": "Controls the visibility of highlighted image regions.",
        "positive_probability": "Positive-class probability", "predicted_class": "Predicted class",
        "confidence": "Model confidence", "positive": "Positive", "negative": "Negative",
        "probability": "Probability", "high": "High", "moderate": "Moderate", "low": "Low",
        "unreliable": "Input checks found unusual properties. Treat the prediction as unreliable.",
        "warning_too_small": "The image resolution is low for reliable analysis.",
        "warning_square_aspect": "This model was trained mainly on square image patches; the aspect ratio is unusual.",
        "warning_extreme_aspect": "The image has an unusually wide or tall aspect ratio.",
        "warning_too_dark": "The image is unusually dark and important details may be lost.",
        "warning_too_bright": "The image is unusually bright and important details may be lost.",
        "warning_low_variation": "The image contains very little visual detail or contrast.",
        "warning_low_color_histology": "The color variation is unusual for an H&E histology image.",
        "warning_low_color_leaf": "The color variation is unusual for a plant leaf photograph.",
        "warning_possibly_blurry": "The image may be blurred or out of focus.",
        "review": "The result requires expert review and must not be used to guide treatment.",
    },
    "УКР": {
        "title": "Аналізатор біологічних зображень",
        "subtitle": "Платформа класифікації ResNet-50",
        "caption": "Дослідницький прототип класифікації біологічних видів",
        "disclaimer": "Лише для досліджень і навчання. Результат є прогнозом моделі та потребує перевірки фахівцем.",
        "settings": "Налаштування аналізу", "language": "Мова / Language", "model": "Модель",
        "category": "Напрям",
        "animals": "Тварини", "plants": "Рослини", "mushrooms": "Гриби",
        "ready": "Модель готова", "training_required": "Потрібне навчання",
        "classes": "класів", "best_accuracy": "Найкраща валідаційна точність",
        "checkpoint_updated": "Дата checkpoint", "architecture": "ResNet-50",
        "checkpoint_version": "Версія checkpoint", "training_epoch": "Епоха навчання",
        "dataset": "Датасет",
        "taxonomy_path": "Таксономія", "top_predictions": "Найімовірніші класи",
        "category_probability": "Імовірність вибраного напряму",
        "filtered_confidence": "Впевненість у межах вибраного напряму",
        "category_mismatch": "Зображення, ймовірно, не належить до вибраного напряму: {domain}.",
        "google_search": "Знайти в Google",
        "device": "Пристрій",
        "inaturalist_task": "Розпізнає 10 000 видів тварин, рослин, грибів та інших організмів із iNaturalist 2021 Mini.",
        "inaturalist_input": "Завантажте чітку фотографію з природи, де організм є головним об’єктом кадру.",
        "checkpoint": "Checkpoint {module} не знайдено. Спочатку виконайте `{command}`.",
        "checkpoint_damaged": "Checkpoint {module} не вдалося завантажити. Файл може бути пошкодженим або несумісним. Відновіть перевірений checkpoint і повторіть спробу.",
        "upload": "Завантажте зображення для {module}",
        "upload_hint": "Завантажте одне зображення.",
        "bad_image": "Завантажений файл не вдалося прочитати як зображення.",
        "file_too_large": "Розмір завантаженого файла перевищує 20 МБ. Виберіть менше зображення.",
        "small_image": "Зображення замале для надійного аналізу.",
        "uploaded": "Завантажене зображення", "analyze": "Проаналізувати зображення",
        "running": "Виконується аналіз {module} за допомогою ResNet-50...",
        "gradcam": "Пояснення Grad-CAM", "positive_probability": "Імовірність позитивного класу",
        "gradcam_intensity": "Інтенсивність Grad-CAM",
        "gradcam_intensity_help": "Керує видимістю виділених ділянок зображення.",
        "predicted_class": "Прогнозований клас", "confidence": "Впевненість моделі",
        "positive": "Позитивний", "negative": "Негативний",
        "probability": "Імовірність",
        "high": "Висока", "moderate": "Середня", "low": "Низька",
        "unreliable": "Перевірка виявила незвичні властивості. Вважайте прогноз ненадійним.",
        "warning_too_small": "Роздільна здатність зображення замала для надійного аналізу.",
        "warning_square_aspect": "Модель навчалася переважно на квадратних фрагментах; співвідношення сторін незвичне.",
        "warning_extreme_aspect": "Зображення має незвично витягнуте співвідношення сторін.",
        "warning_too_dark": "Зображення надто темне, тому важливі деталі можуть бути втрачені.",
        "warning_too_bright": "Зображення надто світле, тому важливі деталі можуть бути втрачені.",
        "warning_low_variation": "Зображення містить дуже мало візуальних деталей або контрасту.",
        "warning_low_color_histology": "Колірна мінливість незвична для H&E-гістологічного зображення.",
        "warning_low_color_leaf": "Колірна мінливість незвична для фотографії листка рослини.",
        "warning_possibly_blurry": "Зображення може бути розмитим або не у фокусі.",
        "review": "Результат потребує експертної перевірки й не може визначати лікування.",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_NAMES = {
    "iNaturalist mini": "models/inaturalist/config.yaml",
}
TRAIN_COMMANDS = {
    "iNaturalist mini": "python models/inaturalist/train.py",
}
DOMAIN_MODELS = {
    "animals": ("iNaturalist mini",),
    "plants": ("iNaturalist mini",),
    "mushrooms": ("iNaturalist mini",),
}
MODULE_TASK_KEYS = {
    "iNaturalist mini": "inaturalist",
}
MODULE_ICONS = {
    "iNaturalist mini": "🦋",
}

MODULE_DISPLAY_NAMES = {"EN": {}, "УКР": {}}


def module_display_name(module: str, language: str) -> str:
    return MODULE_DISPLAY_NAMES[language].get(module, module)

MODULE_CLASS_DESCRIPTIONS = {"EN": {}, "УКР": {}}

CHECKPOINT_VERSIONS = {module: "V1.0" for module in MODULE_ICONS}
MODULE_DATASETS = {
    "iNaturalist mini": "iNaturalist 2021 Mini",
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def checkpoint_summary(
    config, checkpoint_path: Path
) -> tuple[float | None, str | None, int | None]:
    history_path = config.path("paths", "results_dir") / "training_history.json"
    best_accuracy = None
    training_epoch = None
    if history_path.exists():
        try:
            history_data = json.loads(history_path.read_text(encoding="utf-8"))
            if history_data.get("best_accuracy") is not None:
                best_accuracy = float(history_data["best_accuracy"])
            elif history_data.get("history"):
                best_accuracy = max(
                    float(row["validation_accuracy"])
                    for row in history_data["history"]
                    if row.get("validation_accuracy") is not None
                )
            history = history_data.get("history") or []
            if history:
                training_epoch = len(history)
        except (OSError, ValueError, TypeError, KeyError):
            best_accuracy = None
    updated = checkpoint_path.stat().st_mtime if checkpoint_path.exists() else None
    if updated is None:
        return best_accuracy, None, training_epoch
    from datetime import datetime
    return (
        best_accuracy,
        datetime.fromtimestamp(updated).strftime("%d.%m.%Y %H:%M"),
        training_epoch,
    )


def load_predictor_safely(loader, module: str, text: dict[str, str], *args):
    try:
        return loader(*args)
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        st.error(text["checkpoint_damaged"].format(module=module))
        return None


@st.cache_resource
def load_multiclass_predictor(
    config_path: str, checkpoint_mtime: float
) -> MulticlassPredictor:
    del checkpoint_mtime
    config = load_config(config_path)
    return MulticlassPredictor(
        config.path("paths", "checkpoint"), resolve_device("auto"),
        int(config.section("data")["image_size"]),
    )


def readable_class(class_name: str, language: str) -> str:
    return class_name.replace("___", " — ").replace("_", " ")


def prediction_class_details(
    class_name: str, module: str, language: str
) -> tuple[str, str | None]:
    if module != "iNaturalist mini":
        return readable_class(class_name, language), None
    taxonomy = class_name.split("_")
    if len(taxonomy) == 8 and taxonomy[0].isdigit():
        species_name = f"{taxonomy[6]} {taxonomy[7]}"
        taxonomy_path = " › ".join(taxonomy[1:6])
        return species_name, f"ID {taxonomy[0]} · {taxonomy_path}"
    return readable_class(class_name, language), None


def render_prediction_card(
    class_name: str, module: str, language: str, text: dict[str, str],
    label: str | None = None,
) -> None:
    display_name, taxonomy_path = prediction_class_details(class_name, module, language)
    card_label = label or text["predicted_class"]
    path_html = (
        f'<div class="prediction-card__path">{html.escape(text["taxonomy_path"])}: '
        f'{html.escape(taxonomy_path)}</div>'
        if taxonomy_path else ""
    )
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-card__label">{html.escape(card_label)}</div>
            <div class="prediction-card__value">{html.escape(display_name)}</div>
            {path_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_predictions(
    probabilities: dict[str, float], module: str, language: str,
    text: dict[str, str], limit: int = 10,
) -> None:
    st.markdown(f"#### {text['top_predictions']}")
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:limit]
    for rank, (class_name, probability) in enumerate(ranked, start=1):
        display_name, _ = prediction_class_details(class_name, module, language)
        result_column, search_column = st.columns([4, 1], vertical_alignment="center")
        with result_column:
            st.markdown(f"**{rank}. {display_name}** — {probability:.1%}")
            st.progress(float(probability))
        with search_column:
            search_url = f"https://www.google.com/search?q={quote_plus(display_name)}"
            st.link_button(text["google_search"], search_url, use_container_width=True)


with st.sidebar:
    language = st.radio("Language / Мова", ("EN", "УКР"), horizontal=True)
    text = TEXT[language]
    st.header(text["settings"])
    domain = st.radio(
        text["category"],
        tuple(DOMAIN_MODELS),
        format_func=lambda key: text[key],
    )
    module = st.radio(
        text["model"],
        DOMAIN_MODELS[domain],
        format_func=lambda name: module_display_name(name, language),
    )
    display_module = module_display_name(module, language)
    gradcam_intensity = st.slider(
        text["gradcam_intensity"],
        min_value=20,
        max_value=80,
        value=55,
        step=5,
        help=text["gradcam_intensity_help"],
    ) / 100.0

    config_path = PROJECT_ROOT / CONFIG_NAMES[module]
    config = load_config(config_path)
    checkpoint_path = config.path("paths", "checkpoint")
    task_key = MODULE_TASK_KEYS[module]
    class_count = int(config.section("model").get("num_classes", 2))
    if module == "iNaturalist mini":
        class_count = INATURALIST_DOMAIN_CLASS_COUNTS[domain]
    best_accuracy, checkpoint_updated, training_epoch = checkpoint_summary(
        config, checkpoint_path
    )

    listed_classes = MODULE_CLASS_DESCRIPTIONS[language].get(module, ())
    classes_html = ""
    if listed_classes:
        class_rows = "".join(
            f"<div>• {html.escape(class_name)}</div>" for class_name in listed_classes
        )
        classes_html = (
            f'<div class="module-card__classes"><strong>{text["classes"].capitalize()}:</strong>'
            f'<div class="module-card__class-list">{class_rows}</div></div>'
        )
    card_html = (
        '<div class="module-card">'
        f'<div class="module-card__title">{MODULE_ICONS[module]} {display_module}</div>'
        '<div class="module-card__meta">'
        f'<div class="module-card__paragraph">{text[f"{task_key}_task"]}</div>'
        f'<div class="module-card__paragraph">{text[f"{task_key}_input"]}</div>'
        f'{classes_html}'
        f'<div class="module-card__paragraph">{class_count} {text["classes"]} · '
        f'{text["architecture"]}</div>'
        '</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
    status_class = "ready" if checkpoint_path.exists() else "missing"
    status_text = text["ready"] if checkpoint_path.exists() else text["training_required"]
    accuracy_line = (
        f"{text['best_accuracy']}: {best_accuracy * 100:.2f}%<br>"
        if best_accuracy is not None else ""
    )
    updated_line = (
        f"{text['checkpoint_updated']}: {checkpoint_updated}<br>"
        if checkpoint_updated else ""
    )
    epoch_line = (
        f"{text['training_epoch']}: {training_epoch}<br>" if training_epoch else ""
    )
    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(
        f"""
        <div class="model-status model-status--{status_class}">
            <div class="model-status__state">● {status_text}</div>
            <div class="model-status__meta">{accuracy_line}{updated_line}{epoch_line}
            {text['checkpoint_version']}: {CHECKPOINT_VERSIONS[module]}<br>
            {text['dataset']}: {MODULE_DATASETS[module]}<br>
            {text['device']}: {device_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

header_title = json.dumps(text["title"], ensure_ascii=False)
st.markdown(
    f"<style>.stApp {{ --app-header-title: {header_title}; }}</style>",
    unsafe_allow_html=True,
)
st.caption(text["subtitle"])
st.info(text["disclaimer"])
page_loader.empty()

if not checkpoint_path.exists():
    command = TRAIN_COMMANDS[module]
    st.error(text["checkpoint"].format(module=display_module, command=command))
    st.stop()
st.markdown(
    f"### {MODULE_ICONS[module]} {text['upload'].format(module=display_module)}"
)
uploaded = st.file_uploader(
    text["upload"].format(module=display_module),
    type=("png", "jpg", "jpeg", "tif", "tiff"),
    key=f"uploader-{module}",
    label_visibility="collapsed",
)
if uploaded is None:
    st.write(text["upload_hint"])
    st.stop()
if uploaded.size > MAX_UPLOAD_BYTES:
    st.error(text["file_too_large"])
    st.stop()
try:
    image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error(text["bad_image"])
    st.stop()
warning_codes = validate_module_input(image, module)
warnings = [text[f"warning_{code}"] for code in warning_codes]
for warning in warnings:
    st.warning(warning)
left, right = st.columns(2)
with left:
    st.subheader(text["uploaded"])
    st.image(image, use_container_width=True)

if st.button(text["analyze"], type="primary", use_container_width=True):
    with st.spinner(text["running"].format(module=display_module)):
        predictor = load_predictor_safely(
            load_multiclass_predictor, display_module, text,
            str(config_path), checkpoint_path.stat().st_mtime,
        )
        if predictor is None:
            st.stop()
        prediction = predictor.predict(image)
        result_probabilities, kingdom_probability = filter_inaturalist_probabilities(
            prediction.probabilities, domain
        )
        result_class, result_confidence = max(
            result_probabilities.items(), key=lambda item: item[1]
        )
        heatmap = predictor.grad_cam(
            image, result_class, intensity=gradcam_intensity
        )
    st.subheader(f"{display_module} — {text['predicted_class']}")
    a, b, c = st.columns([3, 1, 1])
    with a:
        render_prediction_card(result_class, module, language, text)
    b.metric(text["filtered_confidence"], f"{result_confidence:.1%}")
    c.metric(text["category_probability"], f"{kingdom_probability:.1%}")
    if kingdom_probability < 0.5:
        st.warning(text["category_mismatch"].format(domain=text[domain]))
    render_top_predictions(result_probabilities, module, language, text)
    if heatmap is not None:
        with right:
            st.subheader(text["gradcam"])
            st.image(heatmap, use_container_width=True)
    if warnings:
        st.warning(text["unreliable"])
    st.caption(text["review"])
