import io
import os
import json
import zipfile
from functools import lru_cache
from typing import Any, Dict, List

import gdown
import numpy as np
from PIL import Image
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from app.core.settings import (
    MODEL_BUNDLE_URL,
    MODEL_BUNDLE_ID,
    MODEL_WORKDIR,
    MODEL_PATH,
    CONFIG_PATH,
    CLASSES_PATH,
    MODEL_DEVICE,
    NUM_CLASSES,
    SCORE_THRESH_TEST,
    MIN_SIZE_TEST,
    MAX_SIZE_TEST,
    TOP_K,
)

class DetectronFoodModel:
    def __init__(self):
        self._ensure_bundle()
        self.class_names = self._load_class_names()
        self.cfg = self._build_cfg()
        self.predictor = DefaultPredictor(self.cfg)

    def _download_bundle(self, zip_path: str):
        if MODEL_BUNDLE_ID:
            gdown.download(id=MODEL_BUNDLE_ID, output=zip_path, quiet=False)
            return

        if MODEL_BUNDLE_URL:
            gdown.download(url=MODEL_BUNDLE_URL, output=zip_path, quiet=False, fuzzy=True)
            return

        raise RuntimeError("Debes definir MODEL_BUNDLE_URL o MODEL_BUNDLE_ID en el .env")

    def _ensure_bundle(self):
        os.makedirs(MODEL_WORKDIR, exist_ok=True)

        if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
            return

        zip_path = os.path.join(MODEL_WORKDIR, "model_bundle.zip")

        if not os.path.exists(zip_path):
            self._download_bundle(zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(MODEL_WORKDIR)

        self._normalize_extracted_files()

        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("No se encontró model_final.pth dentro del bundle.")
        if not os.path.exists(CONFIG_PATH):
            raise RuntimeError("No se encontró config_baseline.yaml dentro del bundle.")

    def _normalize_extracted_files(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
            return

        found_model = None
        found_config = None
        found_classes = None

        for root, _, files in os.walk(MODEL_WORKDIR):
            for name in files:
                full = os.path.join(root, name)
                if name == "model_final.pth":
                    found_model = full
                elif name == "config_baseline.yaml":
                    found_config = full
                elif name == "classes.json":
                    found_classes = full

        if found_model and found_model != MODEL_PATH:
            os.replace(found_model, MODEL_PATH)
        if found_config and found_config != CONFIG_PATH:
            os.replace(found_config, CONFIG_PATH)
        if found_classes and found_classes != CLASSES_PATH:
            os.replace(found_classes, CLASSES_PATH)

    def _load_class_names(self) -> List[str]:
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _resolve_device(self) -> str:
        if MODEL_DEVICE == "cpu":
            return "cpu"
        if MODEL_DEVICE == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_cfg(self):
        cfg = get_cfg()

        if os.path.exists(CONFIG_PATH):
            cfg.merge_from_file(CONFIG_PATH)
        else:
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )

        cfg.MODEL.WEIGHTS = MODEL_PATH
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST

        # Mantener el espíritu del entrenamiento, pero con valores seguros para inferencia
        cfg.INPUT.FORMAT = "BGR"
        cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
        cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST

        cfg.MODEL.DEVICE = self._resolve_device()

        return cfg

    def _image_bytes_to_bgr(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")

        image_np = np.asarray(image, dtype=np.uint8)

        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("La imagen debe tener 3 canales (RGB).")

        # RGB -> BGR y contigua en memoria
        image_bgr = np.ascontiguousarray(image_np[:, :, ::-1], dtype=np.uint8)

        return image_bgr

    def _is_noise_label(self, name: str) -> bool:
        name = name.strip()
        return name == "seg-eAln" or name.isdigit()

    def predict(self, image_bytes: bytes, top_k: int = TOP_K) -> Dict[str, Any]:
        image_bgr = self._image_bytes_to_bgr(image_bytes)

        if not isinstance(image_bgr, np.ndarray):
            raise TypeError(f"image_bgr no es ndarray, sino {type(image_bgr)}")

        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8, copy=False)

        if not image_bgr.flags["C_CONTIGUOUS"]:
            image_bgr = np.ascontiguousarray(image_bgr)

        outputs = self.predictor(image_bgr)

        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            return {
                "detections": [],
                "summary_text": "No se detectaron alimentos.",
            }

        detections = []
        has_masks = instances.has("pred_masks")

        for i in range(len(instances)):
            class_id = int(instances.pred_classes[i])
            score = float(instances.scores[i])

            raw_name = (
                self.class_names[class_id]
                if self.class_names and class_id < len(self.class_names)
                else f"class_{class_id}"
            )

            # Filtrar clases basura de tu export
            if self._is_noise_label(raw_name):
                continue

            # Filtrar predicciones débiles
            if score < 0.60:
                continue

            area_pct = None
            if has_masks:
                mask = instances.pred_masks[i].numpy()
                area_pct = round(float(mask.sum()) / float(mask.size) * 100.0, 2)

            detections.append({
                "class_id": class_id,
                "class_name": raw_name,
                "score": round(score, 4),
                "area_pct": area_pct,
            })

        detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:top_k]

        if not detections:
            return {
                "detections": [],
                "summary_text": "No se detectaron alimentos relevantes.",
            }

        summary_parts = []
        for d in detections:
            if d["area_pct"] is not None:
                summary_parts.append(
                    f'{d["class_name"]} (score={d["score"]}, area={d["area_pct"]}%)'
                )
            else:
                summary_parts.append(
                    f'{d["class_name"]} (score={d["score"]})'
                )

        return {
            "detections": detections,
            "summary_text": " | ".join(summary_parts),
        }


@lru_cache(maxsize=1)
def get_model() -> DetectronFoodModel:
    return DetectronFoodModel()