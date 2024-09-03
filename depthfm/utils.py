import numpy as np
from PIL import Image

def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb
