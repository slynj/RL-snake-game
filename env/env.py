import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "log")
MODEL_DIR_PATH = os.path.join(BASE_DIR, "best_model")
MODEL_PATH = os.path.join(MODEL_DIR_PATH, "best_model.zip")
