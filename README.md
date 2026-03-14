🏗️ Архитектура
┌─────────────────────────────────────────────┐
│              Входной видеопоток              │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  🎯 YOLOv11 Detection (yolo_batch_main_mot)  │
│  • Пакетная обработка кадров                 │
│  • 8 алгоритмов NMS на выбор                 │
│  • Координаты + confidence для каждого bbox  │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  ⚛️ Quantum Refiner (quantum_refiner.py)     │
│  • Classical backbone: ResNet18 (заморожен)  │
│  • Quantum layer: 10 кубитов, angle encoding │
│  • Fusion: классические + квантовые фичи     │
│  • Output: 4 класса с вероятностями          │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  🎨 Визуализация (infer_pennyline_yolo.py)   │
│  • Отображение bbox с квантовыми метками     │
│  • Цветовая индикация уверенности            │
│  • Реальное время через OpenCV               │
└─────────────────────────────────────────────┘


🔬 Гибридная модель
# Классическая часть
ResNet18 → Linear(512→128) → BN → ReLU → Dropout
              ↓
         Linear(128→64) → BN → ReLU → Dropout
              ↓
         Linear(64→10) → Tanh  # Вход в квантовый слой

# Квантовая часть (PennyLane)
┌─ Angle Encoding: RY(2*arctan(x_i)) на каждом кубите
├─ Entanglement: CNOT между соседними кубитами  
├─ Variational Ansatz: обучаемые RY(θ) параметры
└─ Measurement: <PauliZ> → 10 выходов

# Финальная классификация
[64 классических + 10 квантовых] → Linear(74→32) → ReLU → Linear(32→4) → Softmax


🎯 Основные задачи:

    🔍 Обнаружение объектов в реальном времени с помощью YOLOv11-26
    ⚛️ Уточнение класса объекта через гибридную квантово-классическую модель
    🚁 Классификация: drone | bird | plane | background
    ⚡ Параллельная обработка через multiprocessing


🗂️ Структура репозитория

📦 yolo_and_QNN/
├── 📄 quantum_refiner.py          # Ядро: HybridModel + QuantumLayer + QuantumRefiner
├── 📄 infer_pennyline_yolo.py     # Обработчик: Quantum_batches + QuantumVisualizer
├── 📄 infer_yolo_QNN.py           # 🚀 Точка входа: оркестрация всего пайплайна
├── 📄 yolo_batch_main_mot.py      # Детектор: YOLO + 8 NMS алгоритмов
├── 📄 init_Yolo_for_sahi_batches_v2.py  # Визуализация и сохранение результатов
├── 📄 requirements.txt           
├── 📄 README.md                   

🚀 Быстрый старт
1. Клонирование и установка
  git clone https://github.com/EVNN304/yolo_and_QNN.git
  cd yolo_and_QNN
  
  # Создайте виртуальное окружение (рекомендуется)
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  # или
  venv\Scripts\activate     # Windows
  
  Шаг 2: Установи зависимости
    # Вариант A: Установка из requirements.txt
    pip install -r requirements.txt

    # Вариант B: Пошаговая установка (рекомендуется для контроля)
    # 1. PyTorch с CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # 2. PennyLane с GPU
    pip install pennylane pennylane-lightning pennylane-lightning-gpu

    # 3. Остальные зависимости
    pip install ultralytics opencv-python opencv-contrib-python Pillow numpy typing-extensions

2. Подготовка моделей
  # YOLOv11-26 веса
  YOLO_MODEL_PATH = "/path/to/best_yolo11x_288x288_batch_64.pt"
  
  # Квантовая модель
  QUANTUM_MODEL_PATH = "/path/to/drones_model_3.pth"
3. Запуск
  python infer_yolo_QNN.py

⚙️ Настройка параметров

  # Пути к данным
  path_video = "/path/to/video.webm"  # или 0 для веб-камеры
  crop_w, crop_h = 288, 288           # Размер патчей для YOLO
  conf_threshold = 0.6                # Порог уверенности детекции
  
  # Классы
  class_map = {
      0: "drone", 
      1: "bird", 
      2: "plane", 
      3: "background"
  }
  cl = Yolo_batches(...)
  cl.set_nms_type("classic")  # Доступные варианты:
    # "classic"   — стандартный NMS (быстрый)
    # "soft"      — Soft-NMS (плавное подавление)
    # "wbf"       — Weighted Boxes Fusion
    # "diou"      — DIoU-NMS (учёт расстояния центров)
    # "adaptive"  — Adaptive NMS (динамический порог)
    # "cluster"   — Cluster NMS
    # "nmm"       — Non-Maximum Merge
    # "greedynmm" — Greedy NMM

🧪 Тестирование

  from quantum_refiner import QuantumRefiner

  refiner = QuantumRefiner(
    model_path="drones_model_3.pth",
    class_map={0: "drone", 1: "bird", 2: "plane", 3: "background"}
)


🤝 Лицензия
MIT License

Copyright (c) 2026 EVNN304

Permission is hereby granted...






Если вы используете этот проект в исследовании, пожалуйста, сошлитесь на него:


@misc{hqnn-detect2026,
  author = {EVNN304},
  title = {HQNN-Detect: Hybrid Quantum-Classical Object Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/EVNN304/yolo_and_QNN}
}
  
