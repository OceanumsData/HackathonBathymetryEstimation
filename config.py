import torch

DATA_DIR = 'data/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_RAW = False
AUTO_ROTATE = False