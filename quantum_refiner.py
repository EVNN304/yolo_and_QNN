import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional
import cv2
import warnings

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)
N_FEATURES = 10
N_QUBITS = N_FEATURES

# Используем lightning.gpu если есть, иначе default.qubit
try:
    qml_dev = qml.device("lightning.gpu", wires=N_QUBITS)
except:
    qml_dev = qml.device("default.qubit", wires=N_QUBITS)




class QuantumLayer(nn.Module):

    def __init__(self, n_qubits: int, n_features: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.weights = nn.Parameter(torch.randn(3 * n_qubits) * 0.01)

        @qml.qnode(qml_dev, interface="torch", diff_method="parameter-shift")
        def quantum_circuit(inputs: torch.Tensor, params: torch.Tensor) -> List[torch.Tensor]:
            # Angle encoding
            for i in range(n_qubits):
                if i < len(inputs):
                    angle = 2 * torch.atan(inputs[i])
                    qml.RY(angle, wires=i)
                else:
                    qml.RY(0.0, wires=i)

            # Entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Variational layers
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = quantum_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_norm = torch.tanh(x)
        outputs = []

        for i in range(batch_size):
            sample_features = x_norm[i]

            if sample_features.shape[0] < self.n_features:
                padded_features = torch.zeros(self.n_features).to(x.device)
                padded_features[:sample_features.shape[0]] = sample_features
                sample_features = padded_features
            elif sample_features.shape[0] > self.n_features:
                sample_features = sample_features[:self.n_features]

            try:
                quantum_output = self.quantum_circuit(sample_features, self.weights)
                output_tensor = torch.stack(quantum_output).to(x.device).float()
                outputs.append(output_tensor)
            except Exception as e:
                print(f"Ошибка в квантовом слое для образца {i}: {e}")
                outputs.append(torch.zeros(self.n_qubits).to(x.device))

        return torch.stack(outputs)


class HybridModel(nn.Module):


    def __init__(self, num_classes: int, n_features: int = N_FEATURES):
        super().__init__()

        print(f"Создание гибридной модели с {num_classes} классами")
        print(f"Используется {N_QUBITS} кубитов")

        self.classical_backbone = models.resnet18(weights=None)

        for param in self.classical_backbone.parameters():
            param.requires_grad = False

        # Размораживаем только последние слои
        for param in self.classical_backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.classical_backbone.fc.parameters():
            param.requires_grad = True

        num_ftrs = self.classical_backbone.fc.in_features  # 512 для ResNet18
        self.classical_backbone.fc = nn.Identity()

        self.classical_features = nn.Sequential(
            nn.Linear(num_ftrs, 128),  # [128, 512]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  # [64, 128]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.quantum_input_layer = nn.Sequential(
            nn.Linear(64, n_features),
            nn.Tanh()
        )

        print(f"Создание квантового слоя с {N_QUBITS} кубитами...")
        self.quantum_layer = QuantumLayer(N_QUBITS, n_features)

        quantum_output_size = N_QUBITS

        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + quantum_output_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(32, num_classes)

        self._initialize_weights()
        print("Модель создана")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_features = self.classical_backbone(x)
        classical_features = self.classical_features(backbone_features)
        quantum_input = self.quantum_input_layer(classical_features)

        quantum_input = torch.nan_to_num(quantum_input, nan=0.0, posinf=1.0, neginf=-1.0)

        quantum_output = self.quantum_layer(quantum_input)
        combined = torch.cat([classical_features, quantum_output], dim=1)
        fused_features = self.fusion_layer(combined)
        output = self.classifier(fused_features)

        temperature = 2.0
        return output / temperature


class QuantumRefiner:
    """Высокоуровневый интерфейс для квантовой классификации"""

    def __init__(self, model_path: str, class_map: Dict[int, str], num_classes: int = 4):
        self.class_map = class_map
        self.num_classes = num_classes

        self.model = HybridModel(num_classes=num_classes, n_features=N_FEATURES)

        try:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(DEVICE).eval()

            print(f"✓ QuantumRefiner loaded | {DEVICE} | classes: {list(class_map.values())}")

        except Exception as e:
            print(f"⚠️ Warning loading model: {e}")
            self.model.to(DEVICE).eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop
        pil_img = Image.fromarray(crop_rgb)
        return self.transform(pil_img).unsqueeze(0)

    def classify(self, crop: np.ndarray) -> Dict:
        with torch.no_grad():
            input_tensor = self._preprocess(crop).to(DEVICE)
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, 1)

            return {
                'class_id': idx.item(),
                'class_name': self.class_map.get(idx.item(), 'Unknown'),
                'confidence': conf.item(),
            }