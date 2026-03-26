import os

MODEL_BUNDLE_URL = os.getenv("MODEL_BUNDLE_URL", "")
MODEL_BUNDLE_ID = os.getenv("MODEL_BUNDLE_ID", "")
MODEL_WORKDIR = os.getenv("MODEL_WORKDIR", "/home/user/app/model_data")

MODEL_PATH = os.path.join(MODEL_WORKDIR, "model_final.pth")
CONFIG_PATH = os.path.join(MODEL_WORKDIR, "config_baseline.yaml")
CLASSES_PATH = os.path.join(MODEL_WORKDIR, "classes.json")

MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")  # auto | cpu | cuda

# Tomados de tu config_baseline.yaml como base
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "206"))
SCORE_THRESH_TEST = float(os.getenv("SCORE_THRESH_TEST", "0.55"))
MIN_SIZE_TEST = int(os.getenv("MIN_SIZE_TEST", "800"))
MAX_SIZE_TEST = int(os.getenv("MAX_SIZE_TEST", "1333"))

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))