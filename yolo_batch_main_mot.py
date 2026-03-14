import multiprocessing as mp
import time

import torchvision.ops as ops
import torch
import numpy as np
from ultralytics import YOLO
from add_scripts.toolset import Detection_centered

ALGORITHMS = [
    {"name": "Classic NMS", "key": "classic", "params": {"iou_threshold": 0.15}},
    {"name": "Soft-NMS", "key": "soft", "params": {"iou_thr": 0.5, "sigma": 0.5, "score_threshold": 0.1}},
    {"name": "WBF", "key": "wbf", "params": {"iou_thr": 0.5, "skip_box_thr": 0.0001}},
    {"name": "DIoU-NMS", "key": "diou", "params": {"iou_threshold": 0.5, "beta": 1.0}},
    {"name": "Adaptive NMS", "key": "adaptive", "params": {"iou_thr_min": 0.3, "iou_thr_max": 0.7}},
    {"name": "Cluster NMS", "key": "cluster", "params": {"iou_threshold": 0.5}},
    {"name": "Non-Maximum Merge", "key": "nmm", "params": {"iou_threshold": 0.15, "merge_method": "weighted"}},
    {"name": "Greedy NMM", "key": "greedynmm", "params": {"iou_threshold": 0.05, "score_threshold": 0.01}},
]

class Yolo_batches():
    def __init__(self,q_in:mp.Queue, q_out:mp.Queue, q_names:mp.Queue, q_to_qnn:mp.Queue, conf=0.6, path_model="/home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt", nms_type="classic"):
        self.q_in = q_in
        self.q_names = q_names
        self.q_out = q_out
        self.q_to_qnn = q_to_qnn
        self.conf = conf
        self.path_model = path_model
        self.clases_names = {}
        self.nms_type = nms_type
        self.nms_params = self._get_default_params(nms_type)
        self.imgsize = 640
        self.half_flag = True
        self.verbose = True

    def set_size_inp_layers(self, val:int):
        self.imgsize = val

    def set_half_flag(self, val:bool):
        self.half_flag = val

    def get_size_inp_layers(self):
        return self.imgsize

    def _get_default_params(self, nms_type):
        for alg in ALGORITHMS:
            if alg["key"] == nms_type:
                return alg["params"]
        return {}

    def set_nms_type(self, nms_type: str):
        if any(alg["key"] == nms_type for alg in ALGORITHMS):
            self.nms_type = nms_type
            self.nms_params = self._get_default_params(nms_type)
        else:
            raise ValueError(f"Unknown NMS type: {nms_type}")

    def set_nms_params(self, **params):
        self.nms_params.update(params)


    def set_path_model(self, val:str):
        self.path_model = val

    def set_conf_model(self, val:float):
        self.conf = val

    def set_classes_names(self, val:dict):
        self.clases_names = val
        self.q_names.put(self.clases_names)

    def set_verbose(self, val:bool):
        self.verbose = val

    def apply_nms(self, all_boxes_glob):
        if not all_boxes_glob:
            return []

        if self.nms_type == "classic":
            return self._nms_classic(all_boxes_glob, **self.nms_params)
        elif self.nms_type == "soft":
            return self._nms_soft(all_boxes_glob, **self.nms_params)
        elif self.nms_type == "wbf":
            return self._nms_wbf(all_boxes_glob, **self.nms_params)
        elif self.nms_type == "diou":
            return self._nms_diou(all_boxes_glob, **self.nms_params)
        elif self.nms_type == "nmm":
            return self._nms_nmm(all_boxes_glob, **self.nms_params)
        elif self.nms_type == "greedynmm":
            return self._nms_greedynmm(all_boxes_glob, **self.nms_params)
        else:
            print(f"Unknown NMS type: {self.nms_type}")
            return all_boxes_glob

    def load_model(self):
        try:
            return YOLO(self.path_model)
        except Exception as e:
            print(f"Error_load_model: {e.args}")
            return None

    def process_start(self, daemon=True):

        prc_main = mp.Process(target=self.main_func, args=(), daemon=daemon)
        prc_main.start()

    def _nms_classic(self, all_boxes_glob, iou_threshold=0.5):
        if not all_boxes_glob:
            return []

        # all_boxes_glob: [[cls, conf, x1, y1, x2, y2], ...]
        boxes = torch.tensor([[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob])
        scores = torch.tensor([conf for _, conf, _, _, _, _ in all_boxes_glob])

        keep_indices = ops.nms(boxes, scores, iou_threshold)

        return [all_boxes_glob[i] for i in keep_indices.tolist()]




    def main_func(self):
        model = self.load_model()
        self.set_classes_names(model.names)

        if model == None:
            exit(0)
        start_time = time.time()
        fr_c = 0
        while True:
            if (not self.q_in.empty()):
                try:
                    lst_batch_img, cropp_cord = self.q_in.get()

                    fr_c += 1
                    results = model(lst_batch_img[::], conf=self.conf,  imgsz=self.imgsize, half=self.half_flag, verbose=self.verbose, augment=False, visualize=False) # batch=len(lst_batch_img)
                    all_data = []
                    predictions = []
                    predictions_qaunts = []
                    for i, r in enumerate(results):
                        if r.boxes is not None:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            cls = r.boxes.cls.cpu().numpy()

                            x_offset, y_offset = cropp_cord[i]

                            boxes_glob = boxes.copy()
                            boxes_glob[:, [0, 2]] += x_offset  # x1, x2
                            boxes_glob[:, [1, 3]] += y_offset  # y1, y2
                            data = np.column_stack([cls, confs, boxes_glob])  # (N, 6)


                            all_data.append(data)

                    if all_data:
                        all_boxes_glob = np.vstack(all_data).tolist()
                        all_boxes_glob = self.apply_nms(all_boxes_glob)
                        predictions = self.filter_and_shift(all_boxes_glob)
                        for i, k in enumerate(predictions):
                            x_cnt, y_cnt = k.get_int_center()
                            w, h = k.get_wh()
                            predictions_qaunts.append([k.obj_class, k.p, x_cnt, y_cnt, w, h])


                    if self.q_out.empty():
                        self.q_out.put([lst_batch_img[-1], predictions])
                    if self.q_to_qnn.empty():
                        self.q_to_qnn.put([lst_batch_img[-1], predictions_qaunts])

                    current_time = time.time()

                    fps = 1.0 / (current_time - start_time) if fr_c > 0 else 0.0  # Мгновенный FPS
                    start_time = current_time
                    print(f"PREDS:YOLO_FPS:{fps}")
                except Exception as e:
                    print(f"Errr_prc_yolo_batch:{e.args}")



    def _nms_greedynmm(self, all_boxes_glob, iou_threshold=0.05, score_threshold=0.01):
        if not all_boxes_glob:
            return []

        boxes = np.array([[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob])
        scores = np.array([conf for _, conf, _, _, _, _ in all_boxes_glob])
        classes = np.array([cls for cls, _, _, _, _, _ in all_boxes_glob])

        keep = []
        merged = np.zeros(len(scores), dtype=bool)

        for i in range(len(scores)):
            if merged[i]:
                continue
            if scores[i] < score_threshold:
                continue

            idxs = np.where(classes == classes[i])[0]
            idxs = idxs[scores[idxs] >= score_threshold]
            idxs = idxs[~merged[idxs]]

            if len(idxs) == 1:
                keep.append(i)
                merged[i] = True
                continue

            current_boxes = boxes[idxs]
            current_scores = scores[idxs]

            x1, y1, x2, y2 = current_boxes[:, 0], current_boxes[:, 1], current_boxes[:, 2], current_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            xx1 = np.maximum(x1[:, None], x1[None, :])
            yy1 = np.maximum(y1[:, None], y1[None, :])
            xx2 = np.minimum(x2[:, None], x2[None, :])
            yy2 = np.minimum(y2[:, None], y2[None, :])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            union = areas[:, None] + areas[None, :] - inter
            iou = inter / (union + 1e-9)

            to_merge = iou >= iou_threshold
            merged_local = np.zeros(len(idxs), dtype=bool)

            for j in range(len(idxs)):
                if merged_local[j]:
                    continue
                merged_local[to_merge[j]] = True

                sub_boxes = current_boxes[to_merge[j]]
                sub_scores = current_scores[to_merge[j]]

                x1_avg = np.average(sub_boxes[:, 0], weights=sub_scores)
                y1_avg = np.average(sub_boxes[:, 1], weights=sub_scores)
                x2_avg = np.average(sub_boxes[:, 2], weights=sub_scores)
                y2_avg = np.average(sub_boxes[:, 3], weights=sub_scores)

                merged_idx = idxs[j]
                all_boxes_glob[merged_idx][2] = x1_avg
                all_boxes_glob[merged_idx][3] = y1_avg
                all_boxes_glob[merged_idx][4] = x2_avg
                all_boxes_glob[merged_idx][5] = y2_avg

            merged[idxs[merged_local]] = True
            keep.append(merged_idx)

        return [all_boxes_glob[i] for i in keep]

    def _nms_nmm(self, all_boxes_glob, iou_threshold=0.15, merge_method="weighted"):
        if not all_boxes_glob:
            return []

        boxes = np.array([[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob])
        scores = np.array([conf for _, conf, _, _, _, _ in all_boxes_glob])
        classes = np.array([cls for cls, _, _, _, _, _ in all_boxes_glob])

        keep = []
        merged = np.zeros(len(scores), dtype=bool)

        for i in range(len(scores)):
            if merged[i]:
                continue

            idxs = np.where(classes == classes[i])[0]
            idxs = idxs[~merged[idxs]]

            if len(idxs) == 1:
                keep.append(i)
                merged[i] = True
                continue

            current_boxes = boxes[idxs]
            current_scores = scores[idxs]

            x1, y1, x2, y2 = current_boxes[:, 0], current_boxes[:, 1], current_boxes[:, 2], current_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            xx1 = np.maximum(x1[:, None], x1[None, :])
            yy1 = np.maximum(y1[:, None], y1[None, :])
            xx2 = np.minimum(x2[:, None], x2[None, :])
            yy2 = np.minimum(y2[:, None], y2[None, :])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            union = areas[:, None] + areas[None, :] - inter
            iou = inter / (union + 1e-9)

            to_merge = iou >= iou_threshold
            merged_local = np.zeros(len(idxs), dtype=bool)

            for j in range(len(idxs)):
                if merged_local[j]:
                    continue
                merged_local[to_merge[j]] = True

                sub_boxes = current_boxes[to_merge[j]]
                sub_scores = current_scores[to_merge[j]]

                if merge_method == "weighted":
                    x1_avg = np.average(sub_boxes[:, 0], weights=sub_scores)
                    y1_avg = np.average(sub_boxes[:, 1], weights=sub_scores)
                    x2_avg = np.average(sub_boxes[:, 2], weights=sub_scores)
                    y2_avg = np.average(sub_boxes[:, 3], weights=sub_scores)
                else:
                    x1_avg = np.mean(sub_boxes[:, 0])
                    y1_avg = np.mean(sub_boxes[:, 1])
                    x2_avg = np.mean(sub_boxes[:, 2])
                    y2_avg = np.mean(sub_boxes[:, 3])

                merged_idx = idxs[j]
                all_boxes_glob[merged_idx][2] = x1_avg
                all_boxes_glob[merged_idx][3] = y1_avg
                all_boxes_glob[merged_idx][4] = x2_avg
                all_boxes_glob[merged_idx][5] = y2_avg

            merged[idxs[merged_local]] = True
            keep.append(merged_idx)

        return [all_boxes_glob[i] for i in keep]

    def _nms_soft(self, all_boxes_glob, iou_thr=0.5, sigma=0.5, score_threshold=0.1):
        if not all_boxes_glob:
            return []

        boxes = np.array([[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob])
        scores = np.array([conf for _, conf, _, _, _, _ in all_boxes_glob])
        classes = np.array([cls for cls, _, _, _, _, _ in all_boxes_glob])

        keep = []
        while len(scores) > 0:
            idx_max = np.argmax(scores)
            keep.append(idx_max)

            xx1 = np.maximum(boxes[idx_max][0], boxes[:, 0])
            yy1 = np.maximum(boxes[idx_max][1], boxes[:, 1])
            xx2 = np.minimum(boxes[idx_max][2], boxes[:, 2])
            yy2 = np.minimum(boxes[idx_max][3], boxes[:, 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            area_current = (boxes[idx_max][2] - boxes[idx_max][0]) * (boxes[idx_max][3] - boxes[idx_max][1])
            area_rest = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            union = area_current + area_rest - inter
            iou = inter / (union + 1e-9)

            scores = scores * np.exp(- (iou ** 2) / sigma)

            mask = scores > score_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

        return [all_boxes_glob[i] for i in keep]

    def _nms_diou(self, all_boxes_glob, iou_threshold=0.5, beta=1.0):
        if not all_boxes_glob:
            return []

        boxes = np.array([[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob])
        scores = np.array([conf for _, conf, _, _, _, _ in all_boxes_glob])
        classes = np.array([cls for cls, _, _, _, _, _ in all_boxes_glob])

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []
        while len(scores) > 0:
            idx_max = np.argmax(scores)
            keep.append(idx_max)

            xx1 = np.maximum(x1[idx_max], x1)
            yy1 = np.maximum(y1[idx_max], y1)
            xx2 = np.minimum(x2[idx_max], x2)
            yy2 = np.minimum(y2[idx_max], y2)

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            union = areas[idx_max] + areas - inter
            iou = inter / (union + 1e-9)

            center_x1 = (x1[idx_max] + x2[idx_max]) / 2
            center_y1 = (y1[idx_max] + y2[idx_max]) / 2
            center_x2 = (x1 + x2) / 2
            center_y2 = (y1 + y2) / 2
            center_dist = (center_x1 - center_x2)**2 + (center_y1 - center_y2)**2

            c_x1 = np.minimum(x1[idx_max], x1)
            c_y1 = np.minimum(y1[idx_max], y1)
            c_x2 = np.maximum(x2[idx_max], x2)
            c_y2 = np.maximum(y2[idx_max], y2)
            diag_dist = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2

            diou = iou - (center_dist / (diag_dist + 1e-9))**beta

            mask = diou <= iou_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

        return [all_boxes_glob[i] for i in keep]

    def _nms_wbf(self, all_boxes_glob, iou_thr=0.5, skip_box_thr=0.0001):
        if not all_boxes_glob:
            return []

        cls_groups = {}
        for box in all_boxes_glob:
            cls = box[0]
            if cls not in cls_groups:
                cls_groups[cls] = []
            cls_groups[cls].append(box)

        result = []
        for cls, boxes in cls_groups.items():
            boxes = np.array(boxes)
            scores = boxes[:, 1]
            mask = scores >= skip_box_thr
            boxes = boxes[mask]
            if len(boxes) == 0:
                continue

            bboxes = boxes[:, 2:6]
            scores = boxes[:, 1]

            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            keep = []
            while len(scores) > 0:
                idx_max = np.argmax(scores)
                keep.append(idx_max)

                xx1 = np.maximum(x1[idx_max], x1)
                yy1 = np.maximum(y1[idx_max], y1)
                xx2 = np.minimum(x2[idx_max], x2)
                yy2 = np.minimum(y2[idx_max], y2)

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h

                union = areas[idx_max] + areas - inter
                iou = inter / (union + 1e-9)

                mask = iou <= iou_thr
                x1 = x1[mask]
                y1 = y1[mask]
                x2 = x2[mask]
                y2 = y2[mask]
                scores = scores[mask]
                areas = areas[mask]

            result.extend([all_boxes_glob[i] for i in keep])

        return result


    def filter_and_shift(self, detections):
        predictions = []
        for i, k in enumerate(detections):
            x_cnt, y_cnt = int(k[2] + ((k[4] - k[2])/2)), int(k[3] + ((k[5] - k[3])/2))
            predictions.append(Detection_centered((float(x_cnt), float(y_cnt), float((k[4] - k[2])), float((k[5] - k[3]))), int(k[0]), k[1]))
        return predictions
