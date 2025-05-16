import onnxruntime as ort
import numpy as np
import cv2
import os

YOLOFACE8_MODEL_PATH = r'D:/visiovox-fusion-platform/models/face_detection/yolov8n-face.onnx'
YOLOFACE8_INPUT_SIZE = 640
YOLOFACE8_CONF_THRESH = 0.2
YOLOFACE8_NMS_THRESH = 0.4

class YoloFace8Detector:
    def __init__(self, model_path=YOLOFACE8_MODEL_PATH, conf_thresh=YOLOFACE8_CONF_THRESH, nms_thresh=YOLOFACE8_NMS_THRESH):
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def preprocess(self, image):
        h0, w0 = image.shape[:2]
        scale = YOLOFACE8_INPUT_SIZE / max(h0, w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        img = cv2.resize(image, (nw, nh))
        new_img = np.zeros((YOLOFACE8_INPUT_SIZE, YOLOFACE8_INPUT_SIZE, 3), dtype=np.uint8)
        new_img[:nh, :nw, :] = img
        img = new_img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, scale, nh, nw

    def nms(self, boxes, scores, thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect_face(self, image):
        h0, w0 = image.shape[:2]
        img, scale, nh, nw = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: img})
        outputs = np.squeeze(outputs[0]).T  # (N, 20)
        if outputs.shape[1] < 5:
            print('[YOLOFACE8 DEBUG] Saída inesperada do modelo:', outputs.shape)
            return None
        bbox_raw, score_raw, kps_raw, *_ = np.split(outputs, [4, 5], axis=1)
        keep_indices = np.where(score_raw > self.conf_thresh)[0]
        if not keep_indices.any():
            print('[YOLOFACE8 DEBUG] Nenhuma detecção válida encontrada!')
            return None
        bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]
        # Converter para (x1, y1, x2, y2)
        x1 = bbox_raw[:, 0] - bbox_raw[:, 2] / 2
        y1 = bbox_raw[:, 1] - bbox_raw[:, 3] / 2
        x2 = bbox_raw[:, 0] + bbox_raw[:, 2] / 2
        y2 = bbox_raw[:, 1] + bbox_raw[:, 3] / 2
        bboxes = np.stack((x1, y1, x2, y2), axis=-1)
        scores = score_raw.flatten()
        # NMS
        keep = self.nms(bboxes, scores, self.nms_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        # Ajustar para o tamanho original
        bboxes = bboxes / scale
        # Selecionar a maior detecção
        if len(bboxes) == 0:
            print('[YOLOFACE8 DEBUG] Nenhuma bbox após NMS!')
            return None
        best = np.argmax(scores)
        x1, y1, x2, y2 = bboxes[best].astype(int)
        # Clampar para dentro da imagem
        x1 = max(0, min(x1, w0-1))
        y1 = max(0, min(y1, h0-1))
        x2 = max(0, min(x2, w0-1))
        y2 = max(0, min(y2, h0-1))
        return (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h) 