# quantum_batch_main.py
import multiprocessing as mp
import time
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional

from quantum_refiner import HybridModel, QuantumRefiner, DEVICE


class Quantum_batches():
    def __init__(
            self,
            q_from_yolo: mp.Queue,
            q_out: mp.Queue,
            conf_threshold: float = 0.6,
            path_model: str = "/home/grida/PycharmProjects/HQNN_class/drones_model_3.pth",
            class_map: Optional[Dict[int, str]] = None,
            num_classes: int = 4,
            verbose: bool = True
    ):
        self.q_from_yolo = q_from_yolo
        self.q_out = q_out
        self.conf_threshold = conf_threshold
        self.path_model = path_model
        self.class_map = class_map if class_map else {}
        self.num_classes = num_classes
        self.verbose = verbose

        self.imgsize = 224
        self.batch_size = 1
        self.half_flag = False

        self.processed_count = 0
        self.start_time = time.time()

        self.model: Optional[QuantumRefiner] = None

    def set_path_qnn_model(self, val:str):
        self.path_model = val

    def set_size_inp_layers(self, val: int):
        self.imgsize = val

    def set_half_flag(self, val: bool):
        self.half_flag = val

    def get_size_inp_layers(self):
        return self.imgsize

    def set_path_model(self, val: str):
        self.path_model = val

    def set_conf_model(self, val: float):
        self.conf_threshold = val

    def set_classes_names(self, val: dict):
        self.class_map = val

    def set_verbose(self, val: bool):
        self.verbose = val

    def set_batch_size(self, val: int):
        self.batch_size = val

    def load_model(self):
        try:
            if not self.class_map:
                self.class_map = {0: "drone", 1: "bird", 2: "plane", 3: "background"}

            self.model = QuantumRefiner(
                model_path=self.path_model,
                class_map=self.class_map,
                num_classes=self.num_classes
            )
            print(f"[QUANTUM] Модель загружена на {DEVICE}")
            return self.model
        except Exception as e:
            print(f"[QUANTUM] Error_load_model: {e.args}")
            return None

    def process_start(self, daemon: bool = True):
        prc_main = mp.Process(target=self.main_func, args=(), daemon=daemon)
        prc_main.start()
        return prc_main

    def classify_crop(self, frame: np.ndarray, bbox: List[int]) -> Dict:
        if self.model is None:
            return {'class_id': -1, 'class_name': 'Unknown', 'confidence': 0.0}

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return {'class_id': -1, 'class_name': 'Empty', 'confidence': 0.0}

        try:
            result = self.model.classify(crop)
            return result
        except Exception as e:
            if self.verbose:
                print(f"[QUANTUM] Classify error: {e}")
            return {'class_id': -1, 'class_name': 'Error', 'confidence': 0.0}

    def main_func(self):
        model = self.load_model()

        if model is None:
            print("[QUANTUM] Не удалось загрузить модель, выход")
            return

        fr_c = 0
        start_time = time.time()

        print(f"[QUANTUM] Process started (PID: {mp.current_process().pid})")

        while True:
            if not self.q_from_yolo.empty():
                try:
                    data = self.q_from_yolo.get()

                    if data is None:
                        break

                    frame = data[0]
                    predictions = data[1]

                    fr_c += 1
                    refined_predictions = []

                    for pred in predictions:
                        x_cnt, y_cnt, w, h = pred[2], pred[3], pred[4], pred[5]
                        x1 = int(x_cnt - w / 2)
                        y1 = int(y_cnt - h / 2)
                        x2 = int(x_cnt + w / 2)
                        y2 = int(y_cnt + h / 2)
                        yolo_cls = pred[0]
                        yolo_conf = pred[1]

                        bbox = [x1, y1, x2, y2]

                        q_result = self.classify_crop(frame, bbox)

                        refined_pred = {
                            'bbox': bbox,
                            'bbox_centered': (x_cnt, y_cnt, w, h),
                            'quantum_cls': q_result['class_id'],
                            'quantum_class_name': q_result['class_name'],
                            'quantum_conf': q_result['confidence'],
                        }

                        refined_predictions.append(refined_pred)

                    if not self.q_out.empty():
                        try:
                            self.q_out.get_nowait()
                        except:
                            pass

                    self.q_out.put([frame, refined_predictions])

                    current_time = time.time()
                    fps = 1.0 / (current_time - start_time) if fr_c > 0 else 0.0
                    start_time = current_time

                    if self.verbose and fr_c % 10 == 0:
                        print(f"[QUANTUM] QNN_FPS:{fps:.2f} | Processed:{fr_c}")

                    self.processed_count = fr_c

                except Exception as e:
                    print(f"[QUANTUM] Errr_prc_quantum_batch: {e.args}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"[QUANTUM] Process stopped. Total processed: {self.processed_count}")


class QuantumVisualizer():
    def __init__(
            self,
            q_results: mp.Queue,
            window_name: str = "Quantum Classification Results",
            window_size: tuple = (1280, 720),
            colors: Optional[Dict] = None
    ):
        self.q_results = q_results
        self.window_name = window_name
        self.window_size = window_size
        self.colors = colors or {
            'yolo_bbox': (0, 255, 0),
            'quantum_label_bg': (0, 0, 0),
            'quantum_label_fg': (255, 255, 255),
            'low_confidence': (0, 165, 255),
        }
        self.running = True

    def draw_quantum_label(
            self,
            frame: np.ndarray,
            bbox: List[int],
            quantum_result: Dict
    ) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)

        class_name = quantum_result.get('quantum_class_name', 'Unknown')
        conf = quantum_result.get('quantum_conf', 0.0) * 100

        label = f"{class_name} {conf:.1f}%"

        text_y = y1 - 10
        if text_y < 30:
            text_y = y1 + 25

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame,
            (x1, text_y - text_h - baseline - 5),
            (x1 + text_w + 10, text_y + baseline + 5),
            self.colors['quantum_label_bg'],
            cv2.FILLED
        )

        text_color = self.colors['quantum_label_fg']
        if conf < 60:
            text_color = self.colors['low_confidence']

        cv2.putText(
            frame, label, (x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['yolo_bbox'], 2)

        return frame

    def process_start(self, daemon: bool = True):
        prc = mp.Process(target=self.main_func, args=(), daemon=daemon)
        prc.start()

    def main_func(self):
        print(f"[VISUALIZER] Started (PID: {mp.current_process().pid})")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)

        while self.running:
            try:
                if not self.q_results.empty():
                    data = self.q_results.get()

                    if data is None:
                        break

                    frame = data[0]
                    predictions = data[1]

                    for pred in predictions:
                        bbox = pred['bbox']

                        frame = self.draw_quantum_label(
                            frame=frame,
                            bbox=bbox,
                            quantum_result=pred
                        )

                    cv2.imshow(self.window_name, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("[VISUALIZER] Exit signal received")
                        self.running = False
                        break

                else:
                    cv2.waitKey(1)

            except Exception as e:
                print(f"[VISUALIZER] Error: {e.args}")
                import traceback
                traceback.print_exc()
                continue

        cv2.destroyAllWindows()
        print("[VISUALIZER] Stopped")