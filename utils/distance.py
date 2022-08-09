from scipy.spatial.distance import cosine
import numpy as np

def cosine_dist_xy(pose1, pose2):
    # input shape = (N, 2)
    if len(pose1) != len(pose2):
        which_is_bigger = 0 if len(pose1) > len(pose2) else 1
        target_len = len([pose1, pose2][1 - which_is_bigger])
        pose1, pose2 = pose1[:target_len], pose2[:target_len]

    dist = []
    for p1, p2 in zip(pose1, pose2):
        try:
            dist.append(cosine(p1, p2))
        except:
            dist.append(cosine(p1.flatten(), p2.flatten()))

    return np.mean(dist)

def cosine_dist_xy_flatten(pose1, pose2):
    # input shape = (N, 2)
    if len(pose1) != len(pose2):
        which_is_bigger = 0 if len(pose1) > len(pose2) else 1
        target_len = len([pose1, pose2][1 - which_is_bigger])
        pose1, pose2 = pose1[:target_len], pose2[:target_len]

    return cosine(pose1.flatten(), pose2.flatten())