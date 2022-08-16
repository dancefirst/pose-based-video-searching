import numpy as np

def extract_feature_svd(pose):
    U, s, V = np.linalg.svd(pose, full_matrices = True)
    return s

