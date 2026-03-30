# HQNN-Detect: Hybrid Quantum-Classical Object Detection

A real-time object detection and classification refinement system based on **YOLOv11** and a hybrid quantum-classical neural network.

---

## System Architecture

The data processing pipeline is divided into three main stages:

### 1. Object Detection (YOLOv11)
Utilizes the **yolo_batch_main_mot** module for primary object detection.
*   **Batch Frame Processing:** Implemented to significantly increase performance.
*   **8 NMS Algorithms:** Support for various Non-Maximum Suppression (NMS) methods.
*   **Coordinates and Confidence:** Outputs Bounding Boxes and confidence scores for each detected object.

### 2. Quantum Refinement (Quantum Refiner)
The **quantum_refiner.py** module performs secondary classification on the detected objects.
*   **Classical Backbone:** Uses a pre-trained and frozen **ResNet18** model.
*   **Quantum Layer:** A **10-qubit** circuit utilizing **angle encoding**.
*   **Fusion:** Integration of classical features with quantum characteristics of the object.
*   **Result:** Precise probability distribution across 4 target classes.

### 3. Visualization and Output
The **infer_pennyline_yolo.py** module handles the display of final results.
*   **Label Overlay:** Bounding boxes updated with quantum-refined class labels.
*   **Color Indication:** Visual representation of classification confidence.
*   **Real-time Processing:** Video stream processing powered by the **OpenCV** library.

---

## Hybrid Model Technical Specification

### Classical Component
Sequence of layers for feature extraction:
*   **ResNet18** (Feature extractor).
*   **Linear(512→128)** — Batch Normalization, ReLU, Dropout.
*   **Linear(128→64)** — Batch Normalization, ReLU, Dropout.
*   **Linear(64→10)** — **Tanh** activation to prepare data for the quantum layer.

### Quantum Component (PennyLane)
Quantum computing logic:
*   **Angle Encoding:** Application of **RY(2*arctan(x_i))** to each qubit.
*   **Entanglement:** Utilizing **CNOT** gates between adjacent qubits to create quantum entanglement.
*   **Variational Ansatz:** Trainable **RY(theta)** parameters.
*   **Measurement:** Obtaining 10 outputs via **PauliZ** expectation values.

### Final Classification
*   **Concatenation:** Merging **64 classical** and **10 quantum** features.
*   **Linear(74→32)** — ReLU activation.
*   **Linear(32→4)** — **Softmax** activation to determine the final class.

---

## Key Functional Objectives

*   **Detection:** Real-time object detection using YOLOv11.
*   **Refinement:** Enhanced classification accuracy via the hybrid model.
*   **Categories:** Target classification: **drone**, **bird**, **plane**, and **background**.
*   **Optimization:** Parallel processing via **multiprocessing**.

---

## Repository Structure

*   `quantum_refiner.py` — Implementation of the hybrid model, quantum layer, and refinement logic.
*   `infer_pennyline_yolo.py` — Tools for batch visualization of results.
*   `infer_yolo_QNN.py` — Main script for pipeline orchestration.
*   `yolo_batch_main_mot.py` — Object detector with support for multiple NMS algorithms.
*   `requirements.txt` — List of necessary libraries and dependencies.

---

## Quick Start

### 1. Cloning and Setup
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/EVNN304/yolo_and_QNN.git
cd yolo_and_QNN
```

Create a virtual environment (recommended):
```bash
# For Linux / Mac
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
**Option A: Fast installation from requirements.txt**
```bash
pip install -r requirements.txt
```

**Option B: Step-by-step installation (recommended for CUDA control)**
```bash
# 1. PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. PennyLane with GPU support
pip install pennylane pennylane-lightning pennylane-lightning-gpu

# 3. Additional dependencies
pip install ultralytics opencv-python opencv-contrib-python Pillow numpy typing-extensions
```

### 3. Model Preparation
Set the correct paths to your model weights in the script:
```python
# YOLOv11-26 weights
YOLO_MODEL_PATH = "/path/to/best_yolo11x_288x288_batch_64.pt"

# Quantum model weights
QUANTUM_MODEL_PATH = "/path/to/drones_model_3.pth"
```

### 4. Run the Pipeline
```bash
python infer_yolo_QNN.py
```

---

## Parameter Configuration

```python
# Data paths
path_video = "/path/to/video.webm"  # or 0 for webcam
crop_w, crop_h = 288, 288           # YOLO patch size
conf_threshold = 0.6                # Detection confidence threshold

# Classes
class_map = {
    0: "drone", 
    1: "bird", 
    2: "plane", 
    3: "background"
}

# NMS Configuration
cl = Yolo_batches(...)
cl.set_nms_type("classic")  # Available options:
    # "classic"   — standard NMS (fast)
    # "soft"      — Soft-NMS (smooth suppression)
    # "wbf"       — Weighted Boxes Fusion
    # "diou"      — DIoU-NMS (distance-based)
    # "adaptive"  — Adaptive NMS
    # "cluster"   — Cluster NMS
    # "nmm"       — Non-Maximum Merge
    # "greedynmm" — Greedy NMM
```

---

## Testing

```python
from quantum_refiner import QuantumRefiner

refiner = QuantumRefiner(
    model_path="drones_model_3.pth",
    class_map={0: "drone", 1: "bird", 2: "plane", 3: "background"}
)
```

---

## Examples

Below are examples of the hybrid quantum-classical system performing real-time aircraft detection and classification.

### Object Detection (Left: Standard YOLO / Right: Hybrid-Quantum Refinement)
*Initial object detection performed by YOLOv11 followed by Quantum Refiner*

#### Frame 1:
![Object Detection](examples/Снимок%20экрана%20от%202026-03-15%2000-40-11.png)
#### Frame 2:
![Quantum Refiner](examples/Снимок%20экрана%20от%202026-03-15%2000-43-00.png)
#### Frame 3:
![Quantum Refiner](examples/Снимок%20экрана%20от%202026-03-15%2000-42-12.png)
#### Frame 4:
![Quantum Refiner](examples/Снимок%20экрана%20от%202026-03-15%2000-41-36.png)
#### Frame 5:
![Quantum Refiner](examples/Снимок%20экрана%20от%202026-03-15%2000-40-50.png)
#### Frame 6:
![Final Result](examples/Снимок%20экрана%20от%202026-03-15%2000-43-42.png)

---

## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@misc{hqnn-detect2026,
  author = {EVNN304},
  title = {HQNN-Detect: Hybrid Quantum-Classical Object Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/EVNN304/yolo_and_QNN}
}
```
