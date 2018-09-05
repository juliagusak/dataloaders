import torch
import numpy as np

IMG = 'img'
LABEL = 'label'

def tensor_to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()
    
def numpy_one_hot(label, num_classes=2):
    label = np.eye(num_classes)[label]
    return label