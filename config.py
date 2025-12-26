import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(RESULTS_DIR, 'model')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 5

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
SEED = 42

CATEGORIES = ['flat', 'houseorplot', 'landparcel', 'commercial unit', 'others']
LABEL2ID = {label: idx for idx, label in enumerate(CATEGORIES)}
ID2LABEL = {idx: label for idx, label in enumerate(CATEGORIES)}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'